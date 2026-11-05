"""实时行情层：秒级快照、1 分钟 K 线、tick 缓冲与微观结构特征。

数据源（全部免 token）
----------------------
- 秒级快照 + 五档盘口：``hq.sinajs.cn``（GBK，需 Referer），与网页行情刷新同源；
- 当日 1 分钟 K 线：``web.ifzq.gtimg.cn/appstock/app/minute/query``
  （返回 ``HHMM 现价 累计量(手) 累计额(元)``，逐条差分即得每分钟 OHLCV）。

时间口径
--------
A 股连续竞价 09:30–11:30、13:00–15:00，共 240 分钟。本模块内所有"加 N 分钟"
都用 :func:`add_trading_minutes`，跨午休会正确跳到 13:00，避免出现
"11:28 预测 5 分钟后 = 11:33（不存在的行情）"这种对不上账的时间戳。
"""
from __future__ import annotations

import json
import math
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta

UA = {"User-Agent": "Mozilla/5.0"}
AM_OPEN, AM_CLOSE = 9 * 60 + 30, 11 * 60 + 30
PM_OPEN, PM_CLOSE = 13 * 60, 15 * 60
TOTAL_MINUTES = 240


def normalize(symbol: str) -> str:
    s = (symbol or "").strip().lower()
    if s.startswith(("sh", "sz", "bj")) and len(s) == 8:
        return s
    if s.isdigit() and len(s) == 6:
        return ("sh" if s[0] == "6" else ("bj" if s[0] in "48" else "sz")) + s
    return s


def now() -> datetime:
    return datetime.now()


def is_open(dt: datetime | None = None) -> bool:
    dt = dt or now()
    if dt.weekday() >= 5:
        return False
    m = dt.hour * 60 + dt.minute
    return (AM_OPEN <= m <= AM_CLOSE) or (PM_OPEN <= m <= PM_CLOSE)


def minute_index(dt: datetime) -> int:
    """交易日内的分钟序号 0..239；不在连续竞价时段返回 -1。"""
    m = dt.hour * 60 + dt.minute
    if AM_OPEN <= m <= AM_CLOSE:
        return m - AM_OPEN
    if PM_OPEN <= m <= PM_CLOSE:
        return 120 + (m - PM_OPEN)
    return -1


def index_to_clock(idx: int) -> str:
    idx = max(0, min(TOTAL_MINUTES - 1, int(idx)))
    m = (AM_OPEN + idx) if idx < 120 else (PM_OPEN + idx - 120)
    return f"{m // 60:02d}:{m % 60:02d}"


def add_trading_minutes(dt: datetime, minutes: int) -> datetime:
    """在交易时段内前进 N 分钟（跨午休自动跳段）。"""
    idx = minute_index(dt)
    if idx < 0:
        # 非交易时段：按"下一个交易日开盘"的语义推进（用于回放/收盘后展示）
        base = dt.replace(hour=9, minute=30, second=0, microsecond=0)
        idx = 0
    else:
        base = dt
    tgt = idx + int(minutes)
    d = 0
    while tgt >= TOTAL_MINUTES:
        tgt -= TOTAL_MINUTES
        d += 1
    day = base + timedelta(days=d)
    m = (AM_OPEN + tgt) if tgt < 120 else (PM_OPEN + tgt - 120)
    return day.replace(hour=m // 60, minute=m % 60, second=0, microsecond=0)


def _http(url: str, timeout: float = 6.0, encoding: str = "utf-8",
          referer: str | None = None) -> str:
    h = dict(UA)
    if referer:
        h["Referer"] = referer
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode(encoding, "ignore")


# =====================================================================
# 快照
# =====================================================================
@dataclass
class Quote:
    symbol: str = ""
    name: str = ""
    ts: str = ""              # "YYYY-MM-DD HH:MM:SS"
    price: float = 0.0
    prev_close: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    volume: float = 0.0       # 股
    amount: float = 0.0       # 元
    bids: list = field(default_factory=list)
    asks: list = field(default_factory=list)
    bid_vol: float = 0.0
    ask_vol: float = 0.0
    weibi: float = 0.0
    delay: float = 0.0

    @property
    def pct(self) -> float:
        return (self.price / self.prev_close - 1) * 100 if self.prev_close else 0.0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "name": self.name, "ts": self.ts,
            "price": self.price, "prev_close": self.prev_close, "open": self.open,
            "high": self.high, "low": self.low, "volume": self.volume,
            "amount": self.amount, "pct": round(self.pct, 3),
            "weibi": round(self.weibi, 2), "bid_vol": self.bid_vol,
            "ask_vol": self.ask_vol,
            "bids": self.bids[:5], "asks": self.asks[:5],
        }


def fetch_quote(symbol: str) -> Quote:
    s = normalize(symbol)
    t0 = time.time()
    try:
        raw = _http(f"https://hq.sinajs.cn/list={s}", timeout=5, encoding="gbk",
                    referer="https://finance.sina.com.cn")
        payload = raw.split('"')[1]
        p = payload.split(",")
        if len(p) < 32:
            raise ValueError("字段不足")
    except Exception:
        return Quote(symbol=s)

    def f(x, d=0.0):
        try:
            return float(x)
        except Exception:
            return d

    def pairs(start: int) -> list:
        out = []
        for i in range(start, start + 10, 2):
            if i + 1 < len(p):
                v, pr = f(p[i]), f(p[i + 1])
                if v > 0 and pr > 0:
                    out.append([v, pr])
        return out

    bids, asks = pairs(10), pairs(20)
    bv = sum(x[0] for x in bids)
    av = sum(x[0] for x in asks)
    return Quote(
        symbol=s, name=p[0], ts=f"{p[30]} {p[31]}",
        price=f(p[3]), prev_close=f(p[2]), open=f(p[1]), high=f(p[4]), low=f(p[5]),
        volume=f(p[8]), amount=f(p[9]), bids=bids, asks=asks,
        bid_vol=bv, ask_vol=av,
        weibi=round((bv - av) / (bv + av) * 100, 2) if (bv + av) else 0.0,
        delay=round(time.time() - t0, 3),
    )


# =====================================================================
# 1 分钟 K 线
# =====================================================================
def fetch_minute_bars(symbol: str) -> tuple[list[dict], str]:
    """当日 1 分钟 K 线。返回 (bars, trade_date)。

    腾讯接口只给「分钟末价 + 累计量额」，故用前收/前一分种价合成 OHLC：
    open=上一分钟收盘（首根用当日开盘），high/low 取相邻两价极值，volume 用累计差分。
    这足以支撑动量/VWAP/区间位置/量能类特征；不要用它做精确形态识别。
    """
    s = normalize(symbol)
    url = f"https://web.ifzq.gtimg.cn/appstock/app/minute/query?code={s}"
    try:
        j = json.loads(_http(url, timeout=6))
        d = (j.get("data") or {}).get(s) or {}
        raw = ((d.get("data") or {}).get("data") or [])
        trade_date = str((d.get("data") or {}).get("date") or "")
        prec = 0.0
        qt = d.get("qt")
        if isinstance(qt, dict):
            row = qt.get(s) or []
            if isinstance(row, list) and len(row) > 4:
                try:
                    prec = float(row[4])
                except Exception:
                    prec = 0.0
        if not raw:
            return [], trade_date
        if len(trade_date) == 8:
            trade_date = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
        bars: list[dict] = []
        prev_price = None
        prev_vol = 0.0
        prev_amt = 0.0
        for line in raw:
            parts = line.split()
            if len(parts) < 3:
                continue
            hm, price, cum_vol = parts[0], float(parts[1]), float(parts[2])
            cum_amt = float(parts[3]) if len(parts) > 3 else 0.0
            idx = _hm_index(hm)
            if idx < 0:
                continue
            vol = max(0.0, cum_vol - prev_vol)      # 手
            amt = max(0.0, cum_amt - prev_amt)
            prev_vol, prev_amt = cum_vol, cum_amt
            op = prev_price if prev_price is not None else (price or prec)
            hi = max(op, price)
            lo = min(op, price)
            bars.append({
                "ts": f"{trade_date} {hm[:2]}:{hm[2:4]}:00",
                "trade_date": trade_date,
                "minute": idx,
                "open": round(op, 3), "high": round(hi, 3),
                "low": round(lo, 3), "close": round(price, 3),
                "volume": round(vol * 100, 0),          # 手 → 股
                "amount": round(amt, 2),
            })
            prev_price = price
        return bars, trade_date
    except Exception:
        return [], ""


def _hm_index(hm: str) -> int:
    try:
        h, m = int(hm[:2]), int(hm[2:4])
    except Exception:
        return -1
    mins = h * 60 + m
    # 09:25 集合竞价、15:00 之后的盘后成交：不计入连续竞价分钟序列
    if 9 * 60 + 25 <= mins < AM_OPEN:
        return -1
    if AM_OPEN <= mins <= AM_CLOSE:
        return mins - AM_OPEN
    if PM_OPEN <= mins <= PM_CLOSE:
        return 120 + (mins - PM_OPEN)
    return -1


# =====================================================================
# tick 缓冲
# =====================================================================
class TickBuffer:
    """最近 N 秒的成交价快照，用来算主动买卖差（OFI）与秒级斜率。

    新浪 hq 只给累计成交量，不给逐笔方向，因此用「价格上行记买、下行记卖」的
    tick rule 近似：这是业界常用的无逐笔数据替代方案，方向解释力弱于真实逐笔，
    在页面上必须如实标注。
    """

    def __init__(self, keep_seconds: int = 600):
        self.keep = keep_seconds
        self.ticks: list[tuple[float, float, float]] = []   # (epoch, price, cum_volume)

    def add(self, ts_epoch: float, price: float, cum_volume: float):
        if price <= 0:
            return
        self.ticks.append((ts_epoch, float(price), float(cum_volume)))
        cut = ts_epoch - self.keep
        if len(self.ticks) > 4000 or (self.ticks and self.ticks[0][0] < cut):
            self.ticks = [t for t in self.ticks if t[0] >= cut][-4000:]

    def ofi(self, seconds: int = 60) -> tuple[float, int]:
        """返回 (买卖不平衡 ∈[-1,1], 样本数)。"""
        if len(self.ticks) < 3:
            return 0.0, 0
        cut = self.ticks[-1][0] - seconds
        seq = [t for t in self.ticks if t[0] >= cut]
        if len(seq) < 3:
            seq = self.ticks[-3:]
        buy = sell = 0.0
        for (_, p0, v0), (_, p1, v1) in zip(seq, seq[1:]):
            dv = max(0.0, v1 - v0)
            if dv <= 0:
                continue
            if p1 > p0:
                buy += dv
            elif p1 < p0:
                sell += dv
        tot = buy + sell
        if tot <= 0:
            return 0.0, len(seq)
        return (buy - sell) / tot, len(seq)

    def slope_bps(self, seconds: int = 60) -> tuple[float, int]:
        """最小二乘斜率（bps/分钟）与样本数。"""
        if len(self.ticks) < 3:
            return 0.0, 0
        cut = self.ticks[-1][0] - seconds
        seq = [t for t in self.ticks if t[0] >= cut]
        if len(seq) < 3:
            return 0.0, 0
        t0, p0 = seq[0][0], seq[0][1]
        xs = [(t[0] - t0) / 60.0 for t in seq]
        ys = [math.log(max(t[1], 1e-9) / max(p0, 1e-9)) * 1e4 for t in seq]
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs)
        if den <= 1e-12:
            return 0.0, n
        beta = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
        return beta, n

    def last(self) -> tuple[float, float]:
        return (self.ticks[-1][1], self.ticks[-1][2]) if self.ticks else (0.0, 0.0)

    def price_near(self, epoch: float, tol: float = 90.0) -> float | None:
        """取最接近某时刻的成交价（用于结算到期预测）。"""
        if not self.ticks:
            return None
        best = min(self.ticks, key=lambda t: abs(t[0] - epoch))
        if abs(best[0] - epoch) > tol:
            return None
        return best[1]


# =====================================================================
# 特征
# =====================================================================
def compute_features(bars: list[dict], quote: Quote | None = None,
                     ticks: TickBuffer | None = None,
                     season: dict | None = None) -> dict:
    """把「1 分钟 K 线 + 快照 + tick」压成一组可解释的微观特征。

    所有收益量都用 bps（1bp=0.01%）；缺失值返回 None，由上层专家决定是否退出，
    **绝不填 0 假装中性**（填 0 会让"没有数据"看起来像"市场没方向"）。
    """
    out: dict = {"n_bars": len(bars or []), "valid": False}
    if not bars or len(bars) < 5:
        return out
    closes = [float(b["close"]) for b in bars]
    vols = [float(b.get("volume") or 0) for b in bars]
    price = float(quote.price) if (quote and quote.price > 0) else closes[-1]
    out["price"] = price
    out["last_ts"] = bars[-1]["ts"]
    out["minute"] = int(bars[-1].get("minute") or 0)

    def ret_bps(k: int) -> float | None:
        if len(closes) <= k or not closes[-1 - k]:
            return None
        return (price / closes[-1 - k] - 1) * 1e4

    out["ret1"] = ret_bps(1)
    out["ret3"] = ret_bps(3)
    out["ret5"] = ret_bps(5)
    out["ret10"] = ret_bps(10)

    # 分钟波动（bps）：只用已完成的 1 分钟收益，避免把 tick 噪声当波动
    rets = []
    for a, b in zip(closes[-31:-1], closes[-30:]):
        if a > 0:
            rets.append(math.log(b / a) * 1e4)
    if len(rets) >= 8:
        mu = sum(rets) / len(rets)
        sd = math.sqrt(sum((r - mu) ** 2 for r in rets) / (len(rets) - 1))
        out["sigma_min"] = max(0.6, sd)
    else:
        out["sigma_min"] = max(1.0, abs(out.get("ret1") or 0.0) * 0.8)

    # VWAP（当日累计额/累计量；腾讯的 volume 已换算成股）
    amt = sum(float(b.get("amount") or 0) for b in bars)
    vol = sum(vols)
    if amt > 0 and vol > 0:
        out["vwap"] = amt / vol
        out["vwap_dev"] = (price / out["vwap"] - 1) * 1e4
    else:
        out["vwap"] = None
        out["vwap_dev"] = None

    highs = [float(b.get("high") or b["close"]) for b in bars]
    lows = [float(b.get("low") or b["close"]) for b in bars]
    day_hi, day_lo = max(highs), min(lows)
    if quote and quote.high > 0:
        day_hi = max(day_hi, quote.high)
    if quote and quote.low > 0:
        day_lo = min(day_lo, quote.low)
    out["day_high"], out["day_low"] = day_hi, day_lo
    rng = day_hi - day_lo
    out["range_pos"] = (price - day_lo) / rng if rng > 0 else 0.5
    out["day_range_bps"] = rng / price * 1e4 if price else 0.0

    # 量能：近 5 分钟 vs 近 30 分钟
    v5 = sum(vols[-5:]) / 5
    v30 = sum(vols[-30:]) / max(1, len(vols[-30:]))
    out["vol_ratio"] = (v5 / v30) if v30 > 0 else None
    # 上涨分钟的量占比（近 20 分钟）
    up = sum(v for a, b, v in zip(closes[-21:-1], closes[-20:], vols[-20:]) if b > a)
    tot = sum(vols[-20:])
    out["up_vol_ratio"] = (up / tot) if tot > 0 else None

    if quote and quote.bid_vol + quote.ask_vol > 0:
        out["depth_imb"] = (quote.bid_vol - quote.ask_vol) / (quote.bid_vol + quote.ask_vol)
        out["weibi"] = quote.weibi
        out["depth_total"] = quote.bid_vol + quote.ask_vol
        if quote.bids and quote.asks:
            out["spread_bps"] = (quote.asks[0][1] - quote.bids[0][1]) / price * 1e4
    if ticks:
        ofi, n = ticks.ofi(60)
        out["ofi"], out["ofi_n"] = ofi, n
        sl, sn = ticks.slope_bps(45)
        out["tick_slope"], out["tick_n"] = sl, sn

    if season and out.get("minute") is not None:
        s = season.get(int(out["minute"]))
        if s:
            out["season_bps"] = s["mean_bps"]
            out["season_n"] = s["n"]

    out["open_gap_bps"] = None
    if quote and quote.prev_close and quote.open:
        out["open_gap_bps"] = (quote.open / quote.prev_close - 1) * 1e4
    out["day_ret_bps"] = ((price / quote.prev_close - 1) * 1e4
                          if quote and quote.prev_close else None)
    out["valid"] = True
    return out


def session_state(dt: datetime | None = None) -> dict:
    dt = dt or now()
    idx = minute_index(dt)
    return {
        "now": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "open": is_open(dt),
        "minute": idx,
        "clock": index_to_clock(idx) if idx >= 0 else dt.strftime("%H:%M"),
        "remaining": max(0, TOTAL_MINUTES - idx) if idx >= 0 else 0,
    }
