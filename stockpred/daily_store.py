"""日线入库：真实 K 线、K 线指标/能量、买卖信号、多周期预测 → SQLite。

落库内容（用户视角："把每天真实的股价、买卖转手信号、K 线信息都存下来"）
------------------------------------------------------------------------
- ``daily_bars``       真实 OHLCV + 涨跌幅/振幅（来源：新浪日线）
- ``daily_indicators`` MA/RSI/MACD/BOLL/ATR/MFI/OBV/形态分 + K 线能量（energy.py）
- ``daily_signals``    六+二路联合决策、买卖点区间、拐点概率（每日一份快照）
- ``daily_forecast``   1/5/7/30 日目标价与区间；到期后回填真实收盘并计算误差
                       → 这张表就是"一天预测 vs 一天真实"对比图的数据源

设计约束：入库失败绝不能让分析失败 —— 全部包在 try 里，页面/CLI 只会看到
一条 `[db]` 开头的提示，不会中断。
"""
from __future__ import annotations

import math

from . import db
from .energy import candle_energy, kline_meta


def _f(x, default=None):
    try:
        if x is None or x == "":
            return default
        v = float(x)
        return default if (v != v or math.isinf(v)) else v
    except Exception:
        return default


def _df_rows(df, limit: int | None = None) -> list[dict]:
    if df is None or len(df) == 0:
        return []
    sub = df.tail(limit) if limit else df
    out = []
    for _, r in sub.iterrows():
        d = {}
        for k, v in r.items():
            try:
                if hasattr(v, "isoformat"):
                    d[k] = str(v)[:10]
                elif v != v:            # NaN
                    d[k] = None
                else:
                    d[k] = v
            except Exception:
                d[k] = None
        out.append(d)
    return out


def store_from_analysis(res: dict) -> dict:
    """把 analyze() 的结果落库。返回写入统计（给页面/CLI 展示）。"""
    out = {"bars": 0, "indicators": 0, "signals": 0, "forecast": 0, "settled": 0}
    symbol = res.get("symbol", "")
    if not symbol:
        return out
    kline = res.get("kline")
    tech = res.get("tech")

    # 1) 真实日线
    try:
        rows = []
        prev_close = None
        for r in _df_rows(kline):
            close = _f(r.get("close"))
            hi, lo = _f(r.get("high")), _f(r.get("low"))
            pct = (close / prev_close - 1) * 100 if (close and prev_close) else None
            amp = (hi - lo) / prev_close * 100 if (hi and lo and prev_close) else None
            rows.append({
                "date": r.get("date"), "open": r.get("open"), "high": hi, "low": lo,
                "close": close, "volume": r.get("volume"),
                "pct_chg": round(pct, 3) if pct is not None else None,
                "amplitude": round(amp, 3) if amp is not None else None,
            })
            prev_close = close
        out["bars"] = db.upsert_daily_bars(symbol, rows)
    except Exception as e:
        print(f"[db] 日线入库失败：{e}")

    # 2) 技术指标 + K 线能量（逐行算，能量是逐日可回溯的）
    try:
        ind_rows = []
        for r in _df_rows(tech):
            e = candle_energy(r.get("open"), r.get("high"), r.get("low"), r.get("close"),
                              vol_ratio=_f(r.get("vol_ratio"), 1.0) or 1.0,
                              atr_norm=_f(r.get("atr_norm"), 0.02) or 0.02)
            ind_rows.append({
                "date": r.get("date"),
                "ma5": _f(r.get("ma5")), "ma10": _f(r.get("ma10")), "ma20": _f(r.get("ma20")),
                "rsi": _f(r.get("rsi")), "macd": _f(r.get("macd")),
                "macd_hist": _f(r.get("macd_hist")), "boll_pctb": _f(r.get("boll_pctb")),
                "atr_norm": _f(r.get("atr_norm")), "cci": _f(r.get("cci")),
                "vol_ratio": _f(r.get("vol_ratio")), "mom5": _f(r.get("mom5")),
                "mom20": _f(r.get("mom20")), "mfi": _f(r.get("mfi")),
                "obv_slope5": _f(r.get("obv_slope5")), "up_vol_ratio": _f(r.get("up_vol_ratio")),
                "pv_corr": _f(r.get("pv_corr")),
                "pat_score": _f(r.get("pat_score")), "pat_bull": _f(r.get("pat_bull")),
                "pat_bear": _f(r.get("pat_bear")),
                "energy": e["energy"], "energy_label": e["label"],
                "body_ratio": e["body_ratio"], "close_pos": e["close_pos"],
            })
        out["indicators"] = db.upsert_daily_indicators(symbol, ind_rows)
    except Exception as e:
        print(f"[db] 指标入库失败：{e}")

    # 3) 决策 / 买卖点 / 拐点快照
    try:
        last = _df_rows(tech, 1)
        meta = {}
        if last:
            r = last[0]
            meta = kline_meta(r, vol_ratio=_f(r.get("vol_ratio"), 1.0) or 1.0,
                              atr_norm=_f(r.get("atr_norm"), 0.02) or 0.02)
        date = str((kline["date"].iloc[-1]))[:10] if kline is not None and len(kline) else ""
        decision = res.get("decision") or None
        if decision and decision.get("signals"):
            out["signals"] = db.upsert_daily_signals(
                symbol, date, decision, res.get("trade_points"),
                res.get("turn"), meta)
        else:
            # 没有真实决策（典型场景：--sync-db 只同步 K 线/指标）时**绝不写空壳行**。
            # 旧实现会插一条 score=NULL、signals_json='[]' 的记录，实时侧的 model 专家
            # 读到"有记录但无信号"，只能报"该日线结论里没有 model 信号"，
            # 且会覆盖掉此前真实分析生成的结论。宁可不写，让下游诚实报"尚无日线结论"。
            out["signals"] = 0
    except Exception as e:
        print(f"[db] 信号入库失败：{e}")

    # 4) 多周期预测（目标日按预测线末行日期对齐真实交易日）
    try:
        preds = res.get("preds") or {}
        base_close = _f(res.get("last_close"))
        n = 0
        for h, df in preds.items():
            if df is None or len(df) == 0:
                continue
            lastrow = df.iloc[-1]
            target = str(lastrow.get("date") or "")[:10]
            band = _f(lastrow.get("band_ret"), 0.0) or 0.0
            pc = _f(lastrow.get("pred_close"))
            if not target or not pc:
                continue
            n += db.upsert_daily_forecast(
                symbol, str(res.get("last_date")), int(h), target, base_close, pc,
                pc * (1 - band), pc * (1 + band),
                _f(lastrow.get("prob_up")), str(lastrow.get("pred_dir") or ""))
        out["forecast"] = n
    except Exception as e:
        print(f"[db] 预测入库失败：{e}")

    # 5) 到期的旧预测回填真实收盘
    try:
        out["settled"] = db.settle_daily_forecast(symbol)
    except Exception as e:
        print(f"[db] 预测结算失败：{e}")
    return out


def sync_symbol(symbol: str, datalen: int = 700, source: str = "sina") -> dict:
    """独立同步：只抓日线 + 算指标入库（不训练模型，秒级）。

    给 CLI（``python run.py --sync-db 600519``）和定时任务用。
    """
    out = {"bars": 0, "indicators": 0, "settled": 0}
    try:
        from .data_fetcher import fetch_kline
        from .features import add_technicals
        kline = fetch_kline(symbol, datalen=datalen, source=source)
        if kline is None or len(kline) == 0:
            return out
        tech = add_technicals(kline)
        res = {"symbol": symbol, "kline": kline, "tech": tech,
               "last_date": str(kline["date"].iloc[-1].date()),
               "last_close": float(kline["close"].iloc[-1]),
               "decision": None, "trade_points": None, "turn": None, "preds": {}}
        out.update({k: v for k, v in store_from_analysis(res).items()
                    if k in ("bars", "indicators", "settled")})
    except Exception as e:
        print(f"[db] 同步失败：{e}")
    return out
