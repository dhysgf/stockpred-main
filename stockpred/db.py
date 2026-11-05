"""SQLite 持久层：日线、K 线指标、买卖信号、分钟线、实时预测、拐点、大模型设置。

为什么单独一层
--------------
1. **可回查**：实时预测必须能事后对账（预测价 vs 真实价），否则"命中率"只能靠嘴说。
   每一条预测都落盘，到期由引擎回填真实价与误差，页面上的命中率/MAE 全部来自这张表。
2. **可积累**：日内季节性、在线岭回归、拐点时间分布都需要历史分钟线；
   这些数据靠每天运行自然累积，不依赖任何付费数据源。
3. **单文件、零服务**：SQLite + WAL，`data/stockpred.db` 一个文件就是全部数据，
   拷走即可复现，不引入数据库服务。

并发约定
--------
`http.server` 是多线程的，实时引擎自己也有后台线程。这里用 thread-local 连接，
写操作走短事务 + `BEGIN IMMEDIATE`，并把 busy_timeout 设为 5 秒；
写入失败只记录、不抛给调用方（行情链路不能因为落库失败而中断）。
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime

import config

_LOCK = threading.RLock()
_LOCAL = threading.local()
_INITED = {"done": False}

SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_bars (
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL,
    volume REAL, amount REAL,
    pct_chg REAL, amplitude REAL, turnover REAL,
    source TEXT DEFAULT 'sina',
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS daily_indicators (
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    ma5 REAL, ma10 REAL, ma20 REAL, ma60 REAL,
    rsi REAL, macd REAL, macd_hist REAL, boll_pctb REAL,
    kdj_k REAL, kdj_d REAL, kdj_j REAL,
    atr_norm REAL, cci REAL, vol_ratio REAL, mom5 REAL, mom20 REAL,
    mfi REAL, obv_slope5 REAL, up_vol_ratio REAL, pv_corr REAL,
    pat_score REAL, pat_bull REAL, pat_bear REAL,
    energy REAL, energy_label TEXT, body_ratio REAL, close_pos REAL,
    vol_score REAL, position20 REAL,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS daily_signals (
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    score REAL, advice TEXT, action TEXT,
    confidence REAL, agreement REAL,
    signals_json TEXT, buy_json TEXT, sell_json TEXT, risk_json TEXT,
    turn_json TEXT, kline_json TEXT,
    created_at TEXT,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS daily_forecast (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    base_date TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    target_date TEXT,
    base_close REAL,
    pred_close REAL,
    lo REAL, hi REAL,
    prob_up REAL,
    direction TEXT,
    actual_close REAL,
    err_pct REAL,
    settled_at TEXT,
    created_at TEXT,
    UNIQUE (symbol, base_date, horizon)
);

CREATE TABLE IF NOT EXISTS intraday_bars (
    symbol TEXT NOT NULL,
    ts TEXT NOT NULL,
    trade_date TEXT,
    open REAL, high REAL, low REAL, close REAL,
    volume REAL, amount REAL,
    PRIMARY KEY (symbol, ts)
);
CREATE INDEX IF NOT EXISTS idx_intraday_date ON intraday_bars(symbol, trade_date);

CREATE TABLE IF NOT EXISTS live_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    ts TEXT NOT NULL,
    target_ts TEXT,
    horizon_min INTEGER,
    price REAL,
    pred_price REAL,
    pred_bps REAL,
    lo REAL, hi REAL,
    direction TEXT,
    conf REAL,
    sigma_bps REAL,
    experts_json TEXT,
    realized_price REAL,
    realized_bps REAL,
    err_bps REAL,
    mode TEXT DEFAULT 'live',
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_live_symbol_ts ON live_predictions(symbol, ts);
CREATE INDEX IF NOT EXISTS idx_live_pending ON live_predictions(symbol, target_ts, realized_price);

CREATE TABLE IF NOT EXISTS expert_state (
    symbol TEXT NOT NULL,
    name TEXT NOT NULL,
    weight REAL,
    n INTEGER DEFAULT 0,
    mae_bps REAL,
    hit REAL,
    pred_bps REAL,
    updated_at TEXT,
    PRIMARY KEY (symbol, name)
);

CREATE TABLE IF NOT EXISTS turn_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    trade_date TEXT,
    ts TEXT,
    kind TEXT,
    price REAL,
    source TEXT,
    conf REAL,
    note TEXT,
    hit INTEGER,
    created_at TEXT,
    UNIQUE (symbol, ts, kind, source)
);

CREATE TABLE IF NOT EXISTS llm_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    provider TEXT,
    model TEXT,
    purpose TEXT,
    ok INTEGER,
    latency REAL,
    note TEXT
);

CREATE TABLE IF NOT EXISTS settings (
    k TEXT PRIMARY KEY,
    v TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS rank_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    market TEXT,
    total INTEGER,
    rows_json TEXT,
    UNIQUE (ts, market)
);
CREATE INDEX IF NOT EXISTS idx_rank_market ON rank_snapshots(market, ts);
"""


def db_path() -> str:
    p = (getattr(config, "LIVE", {}) or {}).get("db_path") or "data/stockpred.db"
    if not os.path.isabs(p):
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), p)
    return p


def _connect() -> sqlite3.Connection:
    p = db_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    c = sqlite3.connect(p, timeout=8.0, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    c.execute("PRAGMA busy_timeout=5000")
    return c


def conn() -> sqlite3.Connection:
    c = getattr(_LOCAL, "conn", None)
    if c is None:
        c = _connect()
        _LOCAL.conn = c
        if not _INITED["done"]:
            init(c)
    return c


def init(c: sqlite3.Connection | None = None):
    c = c or conn()
    with _LOCK:
        c.executescript(SCHEMA)
        c.commit()
        _INITED["done"] = True


@contextmanager
def tx():
    """短事务。异常只回滚，不向外抛（调用方是实时链路，不能被存储拖垮）。"""
    c = conn()
    try:
        c.execute("BEGIN IMMEDIATE")
        yield c
        c.commit()
    except Exception:
        try:
            c.rollback()
        except Exception:
            pass
        raise


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _exec(sql: str, rows: list[tuple], quiet: bool = True) -> int:
    if not rows:
        return 0
    try:
        with tx() as c:
            c.executemany(sql, rows)
        return len(rows)
    except Exception as e:
        if not quiet:
            raise
        print(f"[db] 写入失败（已忽略，不影响行情链路）：{e}")
        return 0


def _query(sql: str, args: tuple = ()) -> list[dict]:
    try:
        c = conn()
        cur = c.execute(sql, args)
        return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        print(f"[db] 查询失败：{e}")
        return []


# =====================================================================
# 日线 / 指标 / 信号
# =====================================================================
def upsert_daily_bars(symbol: str, rows: list[dict], source: str = "sina") -> int:
    data = []
    for r in rows:
        data.append((
            symbol, str(r.get("date"))[:10],
            _f(r.get("open")), _f(r.get("high")), _f(r.get("low")), _f(r.get("close")),
            _f(r.get("volume")), _f(r.get("amount")),
            _f(r.get("pct_chg")), _f(r.get("amplitude")), _f(r.get("turnover")), source,
        ))
    return _exec(
        "INSERT INTO daily_bars(symbol,date,open,high,low,close,volume,amount,"
        "pct_chg,amplitude,turnover,source) VALUES(?,?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(symbol,date) DO UPDATE SET open=excluded.open,high=excluded.high,"
        "low=excluded.low,close=excluded.close,volume=excluded.volume,amount=excluded.amount,"
        "pct_chg=excluded.pct_chg,amplitude=excluded.amplitude,turnover=excluded.turnover",
        data)


_IND_COLS = [
    "ma5", "ma10", "ma20", "ma60", "rsi", "macd", "macd_hist", "boll_pctb",
    "kdj_k", "kdj_d", "kdj_j", "atr_norm", "cci", "vol_ratio", "mom5", "mom20",
    "mfi", "obv_slope5", "up_vol_ratio", "pv_corr",
    "pat_score", "pat_bull", "pat_bear",
    "energy", "energy_label", "body_ratio", "close_pos", "vol_score", "position20",
]


def upsert_daily_indicators(symbol: str, rows: list[dict]) -> int:
    cols = ["symbol", "date"] + _IND_COLS
    sql = (f"INSERT INTO daily_indicators({','.join(cols)}) VALUES({','.join('?' * len(cols))}) "
           f"ON CONFLICT(symbol,date) DO UPDATE SET "
           + ",".join(f"{c}=excluded.{c}" for c in _IND_COLS))
    data = []
    for r in rows:
        vals = [symbol, str(r.get("date"))[:10]]
        for c in _IND_COLS:
            v = r.get(c)
            vals.append(v if isinstance(v, str) or v is None else _f(v))
        data.append(tuple(vals))
    return _exec(sql, data)


def upsert_daily_signals(symbol: str, date: str, decision: dict | None,
                         trade_points: dict | None = None, turn: dict | None = None,
                         kline_meta: dict | None = None) -> int:
    d = decision or {}
    tp = trade_points or {}
    return _exec(
        "INSERT INTO daily_signals(symbol,date,score,advice,action,confidence,agreement,"
        "signals_json,buy_json,sell_json,risk_json,turn_json,kline_json,created_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(symbol,date) DO UPDATE SET score=excluded.score,advice=excluded.advice,"
        "action=excluded.action,confidence=excluded.confidence,agreement=excluded.agreement,"
        "signals_json=excluded.signals_json,buy_json=excluded.buy_json,sell_json=excluded.sell_json,"
        "risk_json=excluded.risk_json,turn_json=excluded.turn_json,kline_json=excluded.kline_json",
        [(symbol, date, _f(d.get("score")), str(d.get("advice") or ""), str(d.get("action") or ""),
          _f(d.get("confidence")), _f(d.get("agreement")),
          _j(d.get("signals")), _j(tp.get("buy_points")), _j(tp.get("sell_points")),
          _j(tp.get("risk")), _j(turn), _j(kline_meta), _now())])


def upsert_daily_forecast(symbol: str, base_date: str, horizon: int, target_date: str,
                          base_close: float, pred_close: float, lo: float, hi: float,
                          prob_up: float | None, direction: str) -> int:
    return _exec(
        "INSERT INTO daily_forecast(symbol,base_date,horizon,target_date,base_close,pred_close,"
        "lo,hi,prob_up,direction,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(symbol,base_date,horizon) DO UPDATE SET target_date=excluded.target_date,"
        "base_close=excluded.base_close,pred_close=excluded.pred_close,lo=excluded.lo,hi=excluded.hi,"
        "prob_up=excluded.prob_up,direction=excluded.direction,created_at=excluded.created_at",
        [(symbol, base_date, int(horizon), target_date, _f(base_close), _f(pred_close),
          _f(lo), _f(hi), None if prob_up is None else _f(prob_up), str(direction), _now())])


def settle_daily_forecast(symbol: str) -> int:
    """把已经到期的日线预测回填真实收盘价与误差（按交易日对齐，不用日历天数）。"""
    rows = _query(
        "SELECT f.id, f.base_date, f.horizon, f.target_date, f.pred_close, b.date AS actual_date, "
        "b.close AS actual_close FROM daily_forecast f "
        "JOIN daily_bars b ON b.symbol=f.symbol AND b.date=f.target_date "
        "WHERE f.symbol=? AND f.actual_close IS NULL", (symbol,))
    if not rows:
        return 0
    data = []
    for r in rows:
        if not r["actual_close"] or not r["pred_close"]:
            continue
        err = (r["actual_close"] / r["pred_close"] - 1) * 100
        data.append((r["actual_close"], round(err, 4), _now(), r["id"]))
    return _exec("UPDATE daily_forecast SET actual_close=?, err_pct=?, settled_at=? WHERE id=?", data)


# =====================================================================
# 分钟线
# =====================================================================
def upsert_intraday_bars(symbol: str, bars: list[dict]) -> int:
    data = []
    for b in bars:
        ts = str(b.get("ts") or b.get("time") or "")
        if not ts:
            continue
        data.append((symbol, ts, str(b.get("trade_date") or ts[:10]),
                     _f(b.get("open")), _f(b.get("high")), _f(b.get("low")), _f(b.get("close")),
                     _f(b.get("volume")), _f(b.get("amount"))))
    return _exec(
        "INSERT INTO intraday_bars(symbol,ts,trade_date,open,high,low,close,volume,amount) "
        "VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(symbol,ts) DO UPDATE SET "
        "high=excluded.high,low=excluded.low,close=excluded.close,volume=excluded.volume,"
        "amount=excluded.amount", data)


def intraday_bars(symbol: str, trade_date: str | None = None, limit: int = 600) -> list[dict]:
    if trade_date:
        rows = _query("SELECT * FROM intraday_bars WHERE symbol=? AND trade_date=? "
                      "ORDER BY ts ASC LIMIT ?", (symbol, trade_date, limit))
    else:
        rows = _query("SELECT * FROM intraday_bars WHERE symbol=? ORDER BY ts DESC LIMIT ?",
                      (symbol, limit))
        rows.reverse()
    return rows


def intraday_dates(symbol: str, limit: int = 15) -> list[str]:
    rows = _query("SELECT DISTINCT trade_date FROM intraday_bars WHERE symbol=? "
                  "ORDER BY trade_date DESC LIMIT ?", (symbol, limit))
    return [r["trade_date"] for r in rows if r["trade_date"]]


# =====================================================================
# 实时预测（含结算）
# =====================================================================
def add_live_prediction(symbol: str, ts: str, target_ts: str, horizon_min: int,
                        price: float, pred_price: float, pred_bps: float,
                        lo: float, hi: float, direction: str, conf: float,
                        sigma_bps: float, experts: list[dict], mode: str = "live") -> int:
    try:
        with tx() as c:
            cur = c.execute(
                "INSERT INTO live_predictions(symbol,ts,target_ts,horizon_min,price,pred_price,"
                "pred_bps,lo,hi,direction,conf,sigma_bps,experts_json,mode,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (symbol, ts, target_ts, int(horizon_min), _f(price), _f(pred_price), _f(pred_bps),
                 _f(lo), _f(hi), str(direction), _f(conf), _f(sigma_bps), _j(experts), mode, _now()))
            return int(cur.lastrowid)
    except Exception as e:
        print(f"[db] 预测落库失败：{e}")
        return 0


def pending_predictions(symbol: str, now_ts: str, limit: int = 200) -> list[dict]:
    return _query(
        "SELECT * FROM live_predictions WHERE symbol=? AND realized_price IS NULL "
        "AND target_ts IS NOT NULL AND target_ts<=? ORDER BY target_ts ASC LIMIT ?",
        (symbol, now_ts, limit))


def settle_prediction(pid: int, realized_price: float, realized_bps: float, err_bps: float):
    _exec("UPDATE live_predictions SET realized_price=?, realized_bps=?, err_bps=? "
          "WHERE id=?", [(_f(realized_price), _f(realized_bps), _f(err_bps), int(pid))])


def recent_live_predictions(symbol: str, limit: int = 240) -> list[dict]:
    rows = _query("SELECT * FROM live_predictions WHERE symbol=? ORDER BY id DESC LIMIT ?",
                  (symbol, limit))
    rows.reverse()
    return rows


def live_predictions_on(symbol: str, trade_date: str, horizon_min: int | None = None,
                        limit: int = 5000) -> list[dict]:
    sql = ("SELECT * FROM live_predictions WHERE symbol=? AND substr(ts,1,10)=? ")
    args: list = [symbol, trade_date]
    if horizon_min:
        sql += "AND horizon_min=? "
        args.append(int(horizon_min))
    sql += "ORDER BY id ASC LIMIT ?"
    args.append(limit)
    return _query(sql, tuple(args))


def live_stats(symbol: str, since: str | None = None, limit: int = 4000) -> dict:
    """实时预测的样本外成绩：整体与分周期 MAE / 方向命中率 / 区间覆盖率。"""
    sql = ("SELECT horizon_min, pred_bps, realized_bps, err_bps, lo, hi, price, pred_price, "
           "realized_price FROM live_predictions WHERE symbol=? AND realized_price IS NOT NULL ")
    args: list = [symbol]
    if since:
        sql += "AND ts>=? "
        args.append(since)
    sql += "ORDER BY id DESC LIMIT ?"
    args.append(limit)
    rows = _query(sql, tuple(args))
    out: dict = {"n": len(rows), "by_h": {}, "mae_bps": None, "hit": None,
                 "cover": None, "bias_bps": None}
    if not rows:
        return out
    by_h: dict[int, list[dict]] = {}
    for r in rows:
        by_h.setdefault(int(r["horizon_min"] or 0), []).append(r)
    for h, rs in sorted(by_h.items()):
        errs = [abs(float(r["err_bps"] or 0)) for r in rs]
        hits = [1 if (float(r["pred_bps"] or 0) > 0) == (float(r["realized_bps"] or 0) > 0) else 0
                for r in rs if abs(float(r["pred_bps"] or 0)) > 1e-9]
        cover = [1 if float(r["lo"] or 0) <= float(r["realized_price"] or 0) <= float(r["hi"] or 0)
                 else 0 for r in rs]
        # |误差| 的 80 分位：直接按"能否覆盖八成样本"来标定区间，比假设正态可靠
        srt = sorted(errs)
        q80 = srt[min(len(srt) - 1, int(math.ceil(0.8 * len(srt))) - 1)] if srt else None
        out["by_h"][h] = {
            "n": len(rs),
            "mae_bps": round(sum(errs) / len(errs), 2),
            "rmse_bps": round((sum(e * e for e in errs) / len(errs)) ** 0.5, 2),
            "q80_bps": round(q80, 2) if q80 is not None else None,
            "hit": round(sum(hits) / len(hits), 4) if hits else None,
            "cover": round(sum(cover) / len(cover), 4),
            "bias_bps": round(sum(float(r["err_bps"] or 0) for r in rs) / len(rs), 2),
        }
    ph = int((getattr(config, "LIVE", {}) or {}).get("primary_horizon", 5))
    main = out["by_h"].get(ph) or (list(out["by_h"].values())[0] if out["by_h"] else {})
    out.update({"mae_bps": main.get("mae_bps"), "hit": main.get("hit"),
                "cover": main.get("cover"), "bias_bps": main.get("bias_bps")})
    return out


# =====================================================================
# 专家状态 / 拐点 / 大模型日志 / 设置
# =====================================================================
def load_expert_state(symbol: str) -> dict[str, dict]:
    rows = _query("SELECT * FROM expert_state WHERE symbol=?", (symbol,))
    return {r["name"]: r for r in rows}


def save_expert_state(symbol: str, states: list[dict]):
    _exec(
        "INSERT INTO expert_state(symbol,name,weight,n,mae_bps,hit,pred_bps,updated_at) "
        "VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(symbol,name) DO UPDATE SET "
        "weight=excluded.weight,n=excluded.n,mae_bps=excluded.mae_bps,hit=excluded.hit,"
        "pred_bps=excluded.pred_bps,updated_at=excluded.updated_at",
        [(symbol, s["name"], _f(s.get("weight")), int(s.get("n") or 0),
          _f(s.get("mae_bps")), _f(s.get("hit")), _f(s.get("pred_bps")), _now())
         for s in states])


def add_turn_point(symbol: str, trade_date: str, ts: str, kind: str, price: float,
                   source: str = "detected", conf: float = 0.0, note: str = "") -> int:
    return _exec(
        "INSERT INTO turn_points(symbol,trade_date,ts,kind,price,source,conf,note,created_at) "
        "VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(symbol,ts,kind,source) DO UPDATE SET "
        "price=excluded.price,conf=excluded.conf,note=excluded.note",
        [(symbol, trade_date, ts, kind, _f(price), source, _f(conf), note, _now())])


def turn_points(symbol: str, trade_date: str | None = None, limit: int = 60) -> list[dict]:
    if trade_date:
        return _query("SELECT * FROM turn_points WHERE symbol=? AND trade_date=? "
                      "ORDER BY ts ASC LIMIT ?", (symbol, trade_date, limit))
    rows = _query("SELECT * FROM turn_points WHERE symbol=? ORDER BY ts DESC LIMIT ?",
                  (symbol, limit))
    rows.reverse()
    return rows


def log_llm(provider: str, model: str, purpose: str, ok: bool, latency: float, note: str = ""):
    _exec("INSERT INTO llm_log(ts,provider,model,purpose,ok,latency,note) VALUES(?,?,?,?,?,?,?)",
          [(_now(), provider, model, purpose, int(bool(ok)), _f(latency), note[:300])])


def llm_logs(limit: int = 30) -> list[dict]:
    return _query("SELECT * FROM llm_log ORDER BY id DESC LIMIT ?", (limit,))


def set_setting(k: str, v) -> None:
    _exec("INSERT INTO settings(k,v,updated_at) VALUES(?,?,?) ON CONFLICT(k) DO UPDATE SET "
          "v=excluded.v,updated_at=excluded.updated_at",
          [(k, v if isinstance(v, str) else json.dumps(v, ensure_ascii=False), _now())])


def get_setting(k: str, default=None):
    rows = _query("SELECT v FROM settings WHERE k=?", (k,))
    if not rows:
        return default
    v = rows[0]["v"]
    if isinstance(v, str) and (v.startswith("{") or v.startswith("[")):
        try:
            return json.loads(v)
        except Exception:
            return v
    return v


def get_settings(keys: list[str]) -> dict:
    out = {}
    for k in keys:
        out[k] = get_setting(k)
    return out


# =====================================================================
# 日线历史（给页面画对比图 / 日内季节性统计）
# =====================================================================
def daily_history(symbol: str, limit: int = 60) -> list[dict]:
    rows = _query("SELECT * FROM daily_bars WHERE symbol=? ORDER BY date DESC LIMIT ?",
                  (symbol, limit))
    rows.reverse()
    return rows


def daily_forecast_history(symbol: str, horizon: int | None = None, limit: int = 60) -> list[dict]:
    sql = "SELECT * FROM daily_forecast WHERE symbol=? "
    args: list = [symbol]
    if horizon:
        sql += "AND horizon=? "
        args.append(int(horizon))
    sql += "ORDER BY base_date DESC LIMIT ?"
    args.append(limit)
    rows = _query(sql, tuple(args))
    rows.reverse()
    return rows


def daily_signal_history(symbol: str, limit: int = 30) -> list[dict]:
    rows = _query("SELECT * FROM daily_signals WHERE symbol=? ORDER BY date DESC LIMIT ?",
                  (symbol, limit))
    rows.reverse()
    return rows


def minute_seasonality(symbol: str, days: int = 10) -> dict[int, dict]:
    """按「日内第几分钟」统计未来 5 分钟的平均收益（bps）与样本数。

    用途：season 专家。样本不足时返回空字典 → 该专家自动从加权中剔除，
    而不是编一个 0 当作"中性"参与投票。
    """
    rows = _query(
        "SELECT ts, close, trade_date FROM intraday_bars WHERE symbol=? "
        "ORDER BY ts ASC", (symbol,))
    if not rows:
        return {}
    acc: dict[int, list[float]] = {}
    by_day: dict[str, list[dict]] = {}
    for r in rows:
        by_day.setdefault(r["trade_date"] or r["ts"][:10], []).append(r)
    days_used = sorted(by_day)[-days:]
    for d in days_used:
        bars = by_day[d]
        closes = [float(b["close"] or 0) for b in bars]
        for i, b in enumerate(bars):
            if i + 5 >= len(closes):
                break
            if not closes[i] or not closes[i + 5]:
                continue
            minute = _minute_index(str(b["ts"]))
            if minute < 0:
                continue
            r_bps = (closes[i + 5] / closes[i] - 1) * 1e4
            acc.setdefault(minute, []).append(r_bps)
    out = {}
    for m, vals in acc.items():
        if len(vals) < 3:
            continue
        out[m] = {"mean_bps": sum(vals) / len(vals), "n": len(vals)}
    return out


def _minute_index(ts: str) -> int:
    """HH:MM → 交易分钟序号（0~239）；非交易时段返回 -1。"""
    try:
        hm = ts[11:16] if len(ts) >= 16 else ts
        h, m = int(hm[:2]), int(hm[3:5])
    except Exception:
        return -1
    mins = h * 60 + m
    am0, am1 = 9 * 60 + 30, 11 * 60 + 30
    pm0, pm1 = 13 * 60, 15 * 60
    if am0 <= mins <= am1:
        return mins - am0
    if pm0 <= mins <= pm1:
        return 120 + (mins - pm0)
    return -1


def reset_symbol(symbol: str) -> dict:
    """清空某只票的实时数据（调试用）。日线/分钟线保留，只清预测与专家状态。"""
    out = {}
    for t in ("live_predictions", "expert_state", "turn_points"):
        try:
            with tx() as c:
                cur = c.execute(f"DELETE FROM {t} WHERE symbol=?", (symbol,))
                out[t] = cur.rowcount
        except Exception as e:
            out[t] = f"失败：{e}"
    return out


def _f(x, default=None):
    try:
        if x is None or x == "":
            return default
        v = float(x)
        if v != v:      # NaN
            return default
        return v
    except Exception:
        return default


def _j(x) -> str | None:
    if x is None:
        return None
    try:
        return json.dumps(x, ensure_ascii=False, default=str)
    except Exception:
        return None


# =====================================================================
# 涨幅榜快照（东方财富爬虫）
# =====================================================================
def save_rank_snapshot(market: str, rows: list[dict], total: int = 0) -> bool:
    """每分钟最多一份快照（同分钟覆盖）；内容与上一份完全相同时直接跳过。

    收盘后榜单是冻结的，若不去重，历史表会被几十份一模一样的快照塞满。
    """
    ts = time.strftime("%Y-%m-%d %H:%M")
    payload = _j(rows)
    try:
        with tx() as c:
            prev = c.execute(
                "SELECT ts, rows_json FROM rank_snapshots WHERE market=? "
                "ORDER BY ts DESC LIMIT 1", (market,)).fetchone()
            if prev and prev["rows_json"] == payload:
                # 数据没变：不加新行，但把最后一份的时间戳更新到当前分钟，
                # 这样历史表读起来是"这份数据一直有效到 XX:XX"，而不是一串重复。
                c.execute("UPDATE rank_snapshots SET ts=?, total=? WHERE ts=? AND market=?",
                          (ts, int(total or 0), prev["ts"], market))
                return False
            c.execute(
                "INSERT INTO rank_snapshots(ts,market,total,rows_json) VALUES(?,?,?,?) "
                "ON CONFLICT(ts,market) DO UPDATE SET total=excluded.total,"
                "rows_json=excluded.rows_json",
                (ts, market, int(total or 0), payload))
        return True
    except Exception as e:
        print(f"[db] 榜单快照写入失败：{e}")
        return False


def rank_history(market: str, limit: int = 48) -> list[dict]:
    rows = _query("SELECT ts,market,total,rows_json FROM rank_snapshots WHERE market=? "
                  "ORDER BY ts DESC LIMIT ?", (market, limit))
    out = []
    for r in rows:
        try:
            out.append({"ts": r["ts"], "total": r["total"],
                        "rows": json.loads(r["rows_json"] or "[]")})
        except Exception:
            continue
    return out


def db_info() -> dict:
    p = db_path()
    size = os.path.getsize(p) if os.path.exists(p) else 0
    counts = {}
    for t in ("daily_bars", "daily_indicators", "daily_signals", "daily_forecast",
              "intraday_bars", "live_predictions", "turn_points", "expert_state"):
        rows = _query(f"SELECT COUNT(*) AS n FROM {t}")
        counts[t] = rows[0]["n"] if rows else 0
    return {"path": p, "size_kb": round(size / 1024, 1), "counts": counts,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
