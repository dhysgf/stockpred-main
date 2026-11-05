"""日内拐点：预测"今天的高点/低点大概在什么时间、什么价位"，以及下一个局部拐点。

两个不同的问题，必须分开回答
----------------------------
1. **今日极值（high/low）是否已经出现**：这是"剩余时段还能不能创新高/新低"的概率题。
   用随机游走反射原理近似：设剩余 R 分钟的累计波动 σ√R，价格距前高 d，
   则触及概率 ≈ 2·(1−Φ(d/(σ√R)))。这个近似假设无漂移，
   所以必须把集成模型给的分钟漂移 μ 加进去，否则单边趋势日会系统性低估突破概率。
2. **下一个局部拐点在哪**：这是"动能何时衰竭"的时点题。
   用日内 RSI + 连续同向 bar 数 + 量能衰减做规则判断，输出"几分钟后 / 大概什么价"。

时间分布来自本机积累的分钟线库（每天运行会自动增多）；
样本不足时退回经验先验（A 股日内高低点常见于开盘 30 分钟与尾盘 30 分钟），
并在返回值里标注 ``profile_source``，绝不让先验冒充统计结果。
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

from . import db
from .realtime import index_to_clock, minute_index

# 经验先验：日内第几分钟出现当日最高/最低的相对权重（每 30 分钟一段）
_PRIOR_HIGH = {0: 0.30, 1: 0.18, 2: 0.10, 3: 0.10, 4: 0.10, 5: 0.10, 6: 0.14, 7: 0.18}
_PRIOR_LOW = {0: 0.28, 1: 0.18, 2: 0.12, 3: 0.10, 4: 0.10, 5: 0.10, 6: 0.12, 7: 0.16}


def _cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _parse_ts(ts) -> datetime | None:
    """'%Y-%m-%d %H:%M:%S' → datetime；解析不了就返回 None（调用方退回墙钟）。"""
    if not ts:
        return None
    try:
        return datetime.strptime(str(ts), "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def intraday_rsi(bars: list[dict], n: int = 14) -> float:
    if len(bars) < n + 1:
        return 50.0
    closes = [float(b["close"]) for b in bars]
    gains = losses = 0.0
    for a, b in zip(closes[-n - 1:-1], closes[-n:]):
        d = b - a
        gains += max(0.0, d)
        losses += max(0.0, -d)
    if losses <= 0:
        return 85.0 if gains > 0 else 50.0
    rs = (gains / n) / (losses / n)
    return 100 - 100 / (1 + rs)


def detect_turns(bars: list[dict], win: int = 3, min_swing_bps: float = 12.0) -> list[dict]:
    """已确认的日内局部拐点（左右各 win 根，且反向幅度 ≥ min_swing_bps）。"""
    if len(bars) < 2 * win + 2:
        return []
    closes = [float(b["close"]) for b in bars]
    out = []
    for i in range(win, len(bars) - win):
        c = closes[i]
        if c <= 0:
            continue
        left = closes[i - win:i]
        right = closes[i + 1:i + 1 + win]
        if c >= max(left) and c >= max(right):
            swing = (c - min(right)) / c * 1e4
            if swing >= min_swing_bps:
                out.append({"ts": bars[i]["ts"], "price": round(c, 2), "kind": "顶",
                            "swing_bps": round(swing, 1)})
        elif c <= min(left) and c <= min(right):
            swing = (max(right) - c) / c * 1e4
            if swing >= min_swing_bps:
                out.append({"ts": bars[i]["ts"], "price": round(c, 2), "kind": "底",
                            "swing_bps": round(swing, 1)})
    return out[-12:]


def time_profile(symbol: str, days: int = 15) -> tuple[dict, str]:
    """统计历史每日最高/最低出现的分钟分布。返回 ({'high': {minute: w}, 'low': ...}, 来源说明)。"""
    rows = db.intraday_bars(symbol, limit=days * 260)
    by_day: dict[str, list[dict]] = {}
    for r in rows:
        by_day.setdefault(str(r.get("trade_date") or str(r["ts"])[:10]), []).append(r)
    highs, lows = [], []
    for d, bars in by_day.items():
        bars = sorted(bars, key=lambda b: str(b["ts"]))
        if len(bars) < 60:
            continue
        ci = max(range(len(bars)), key=lambda i: float(bars[i]["close"]))
        li = min(range(len(bars)), key=lambda i: float(bars[i]["close"]))
        highs.append(int(bars[ci].get("minute") or minute_from_ts(str(bars[ci]["ts"]))))
        lows.append(int(bars[li].get("minute") or minute_from_ts(str(bars[li]["ts"]))))
    if len(highs) < 3:
        return {}, "经验先验（本机仅 %d 天分钟线，样本不足）" % len(highs)

    def hist(vals: list[int]) -> dict[int, float]:
        cnt: dict[int, int] = {}
        for m in vals:
            cnt[int(m) // 15 * 15] = cnt.get(int(m) // 15 * 15, 0) + 1
        tot = sum(cnt.values()) or 1
        return {k: v / tot for k, v in cnt.items()}

    return {"high": hist(highs), "low": hist(lows)}, f"本机 {len(highs)} 天分钟线统计"


def minute_from_ts(ts: str) -> int:
    try:
        h, m = int(ts[11:13]), int(ts[14:16])
    except Exception:
        return 0
    mins = h * 60 + m
    if 9 * 60 + 30 <= mins <= 11 * 60 + 30:
        return mins - (9 * 60 + 30)
    if 13 * 60 <= mins <= 15 * 60:
        return 120 + mins - 13 * 60
    return 0


def _window(profile: dict[int, float] | None, prior: dict[int, float],
            now_minute: int) -> tuple[str, float]:
    """在剩余时段内取概率质量最集中的 40 分钟窗口。"""
    weights = profile or {k * 30: v for k, v in prior.items()}
    remaining = [(m, w) for m, w in sorted(weights.items())
                 if m >= max(0, now_minute - 5) and m < 240]
    if not remaining:
        start = min(239, now_minute)
        return f"{index_to_clock(start)}-{index_to_clock(min(239, start + 30))}", 0.0
    tot = sum(w for _, w in remaining) or 1.0
    best_start, best_mass = remaining[0][0], -1.0
    for m, _ in remaining:
        mass = sum(w for mm, w in remaining if m <= mm < m + 40) / tot
        if mass > best_mass:
            best_mass, best_start = mass, m
    start = int(best_start)
    return f"{index_to_clock(start)}-{index_to_clock(min(239, start + 39))}", round(best_mass, 3)


def predict_extremes(features: dict, session: dict, pred_bps: float,
                     sigma_min: float, profile: dict | None = None) -> dict:
    """预测今日剩余时段的最高/最低价与时间窗。"""
    p = float(features.get("price") or 0)
    if p <= 0:
        return {}
    now_min = int(session.get("minute") if session.get("minute") is not None else features.get("minute") or 0)
    remain = max(1, 240 - now_min)
    sigma = max(0.8, float(sigma_min or 3.0)) * math.sqrt(remain)    # 剩余时段总波动（bps）
    mu = float(pred_bps or 0) / 5.0                                  # bps/分钟
    drift = mu * remain

    day_hi = float(features.get("day_high") or p)
    day_lo = float(features.get("day_low") or p)

    def side(level: float, kind: str) -> dict:
        if kind == "高":
            d = (level / p - 1) * 1e4
            # 已在前高附近：仍可能被突破，给一个不可忽略的续创新高概率
            if d <= 2.0:
                prob_done = 0.30
                exp_add = drift + 0.8 * sigma
            else:
                z = (d - drift) / max(1e-6, sigma)
                prob_done = 1 - min(1.0, 2 * (1 - _cdf(max(0.0, z))))
                exp_add = drift + 0.8 * sigma if drift > 0 else 0.35 * sigma
            exp_price = max(level, p * (1 + exp_add / 1e4))
            lo = min(exp_price, max(level, p * (1 + (exp_add - 0.9 * sigma) / 1e4)))
            hi = max(exp_price, p * (1 + (exp_add + 0.9 * sigma) / 1e4))
        else:
            d = (1 - level / p) * 1e4
            if d <= 2.0:
                prob_done = 0.30
                exp_add = -drift + 0.8 * sigma
            else:
                z = (d + drift) / max(1e-6, sigma)
                prob_done = 1 - min(1.0, 2 * (1 - _cdf(max(0.0, z))))
                exp_add = -drift + 0.8 * sigma if drift < 0 else 0.35 * sigma
            exp_price = min(level, p * (1 - exp_add / 1e4))
            lo = min(exp_price, p * (1 - (exp_add + 0.9 * sigma) / 1e4))
            hi = max(exp_price, p * (1 - (exp_add - 0.9 * sigma) / 1e4))
        win, mass = _window((profile or {}).get("high" if kind == "高" else "low"),
                            _PRIOR_HIGH if kind == "高" else _PRIOR_LOW, now_min)
        return {
            "price": round(exp_price, 2), "lo": round(lo, 2), "hi": round(hi, 2),
            "prob_done": round(max(0.02, min(0.98, prob_done)), 3),
            "window": win, "window_mass": mass,
            "level": round(level, 2),
        }

    return {
        "high": side(day_hi, "高"),
        "low": side(day_lo, "低"),
        "sigma_remain_bps": round(sigma, 1),
        "drift_bps": round(drift, 1),
        "remaining_min": remain,
        "profile_source": (profile or {}).get("source", "经验先验"),
        "now": session.get("now"),
    }


def predict_next_turn(bars: list[dict], features: dict, sigma_min: float,
                      pred_bps: float, now_dt: datetime | None = None) -> dict | None:
    """下一个局部拐点（时点 + 价位 + 方向）。"""
    if len(bars) < 8:
        return None
    closes = [float(b["close"]) for b in bars]
    p = float(features.get("price") or closes[-1])
    rsi = intraday_rsi(bars)
    up_streak = down_streak = 0
    for a, b in zip(reversed(closes[:-1]), reversed(closes)):
        if b > a and up_streak == 0:
            up_streak += 1
        elif b > a and down_streak == 0:
            up_streak += 1
        elif b < a and down_streak == 0:
            down_streak += 1
        else:
            break
    vols = [float(b.get("volume") or 0) for b in bars]
    v_recent = sum(vols[-3:]) / 3
    v_base = sum(vols[-15:]) / 15 if len(vols) >= 15 else v_recent
    vol_decay = v_recent / v_base if v_base > 0 else 1.0
    sig = max(0.8, float(sigma_min or 3.0))

    kind, in_min, conf, note = "", 0, 0.0, ""
    if rsi >= 70 and up_streak >= 3:
        kind, in_min = "顶", max(2, min(12, 9 - up_streak))
        note = f"RSI {rsi:.0f} 超买 + 连涨 {up_streak} 分钟" + ("，量能衰减" if vol_decay < 0.85 else "")
        conf = 0.35 + 0.25 * min(1.0, (rsi - 70) / 20) + (0.1 if vol_decay < 0.85 else 0)
    elif rsi <= 30 and down_streak >= 3:
        kind, in_min = "底", max(2, min(12, 9 - down_streak))
        note = f"RSI {rsi:.0f} 超卖 + 连跌 {down_streak} 分钟" + ("，量能衰减" if vol_decay < 0.85 else "")
        conf = 0.35 + 0.25 * min(1.0, (30 - rsi) / 20) + (0.1 if vol_decay < 0.85 else 0)
    elif abs(pred_bps) > 2 * sig:
        # 单边趋势中谈"下一个拐点"意义不大，只提示反向修复位
        kind = "顶" if pred_bps > 0 else "底"
        in_min = 12
        note = "趋势较强，拐点提示可靠性低（仅作反向修复参考）"
        conf = 0.15
    else:
        return None

    move = sig * math.sqrt(max(1, in_min)) * 0.35
    price = p * (1 + (move if kind == "顶" else -move) / 1e4)
    base = now_dt or datetime.now()
    return {"kind": kind, "in_minutes": int(in_min), "price": round(price, 2),
            "conf": round(max(0.05, min(0.8, conf)), 2), "note": note,
            "eta": (base + timedelta(minutes=in_min)).strftime("%H:%M")}


def rebuild(symbol: str, bars: list[dict], features: dict, session: dict,
            pred_bps: float, sigma_min: float, persist: bool = True) -> dict:
    """一次性产出：已确认拐点 + 今日极值预测 + 下一拐点。"""
    prof, src = time_profile(symbol)
    if prof:
        prof["source"] = src
        profile = prof
    else:
        profile = {}
    detected = detect_turns(bars)
    ext = predict_extremes(features, session, pred_bps, sigma_min, profile)
    if ext and not prof:
        ext["profile_source"] = src
    nxt = predict_next_turn(bars, features, sigma_min, pred_bps,
                            now_dt=_parse_ts(session.get("now")))
    trade_date = str((bars[-1].get("trade_date") if bars else "") or
                     datetime.now().strftime("%Y-%m-%d"))
    if persist:
        for t in detected[-4:]:
            db.add_turn_point(symbol, trade_date, str(t["ts"]), t["kind"], t["price"],
                              source="detected", conf=0.9,
                              note=f"局部极值，反向 {t.get('swing_bps', 0)} bps")
        if ext:
            ts = str(session.get("now") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            if ext.get("high"):
                db.add_turn_point(symbol, trade_date, ts, "顶", ext["high"]["price"],
                                  source="predicted", conf=1 - ext["high"]["prob_done"],
                                  note=f"今日高点预测 {ext['high']['window']}，"
                                       f"已出现概率 {ext['high']['prob_done']*100:.0f}%")
            if ext.get("low"):
                db.add_turn_point(symbol, trade_date, ts, "底", ext["low"]["price"],
                                  source="predicted", conf=1 - ext["low"]["prob_done"],
                                  note=f"今日低点预测 {ext['low']['window']}，"
                                       f"已出现概率 {ext['low']['prob_done']*100:.0f}%")
    return {"detected": detected, "extremes": ext, "next": nxt,
            "rsi": round(intraday_rsi(bars), 1),
            "profile_source": (profile or {}).get("source", src)}
