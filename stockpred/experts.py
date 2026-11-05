"""分钟内多方法专家 + 在线加权（Hedge）。

十四路专家（每一路都必须能单独解释、单独结算、单独统计命中率）
------------------------------------------------------------
| 名称    | 依据                                   | 数据         | 对应日线信号 |
|---------|----------------------------------------|--------------|--------------|
| micro   | 五档委买/委卖不平衡 → 冲击              | 快照盘口     | pressure(挂单)|
| flow    | tick 主动买卖差（OFI 近似）             | 秒级快照     | pressure(成交)|
| volume  | 量价配合：上涨分钟量占比 + 量能放大      | 1 分钟 K 线  | volume       |
| energy  | K 线能量：实体/影线/收盘位置/量能合成    | 1 分钟 K 线  | energy       |
| pattern | 形态：连阳连阴过度、长影线、双顶双底      | 1 分钟 K 线  | pattern      |
| range   | 日内区间位置 + 量能（突破/假突破）       | 1 分钟 K 线  | -            |
| mom     | 短周期动量延续（带阻尼）                | 1 分钟 K 线  | -            |
| revert  | 相对 VWAP 的均值回复                   | 1 分钟 K 线  | -            |
| season  | 同一时刻的历史平均漂移                  | 历史分钟线库 | -            |
| ml      | 多因子在线岭回归                       | 历史分钟线库 | model(分钟)  |
| model   | 日线多模型方向概率（RF+GBDT+LSTM 融合）  | daily_signals| model        |
| turn    | 拐点风险（顶部为负、底部为正）           | 拐点模型     | turn         |
| llm     | 大模型独立预测（可选、限流）             | 外部模型     | -            |
| news    | 新闻/政策/舆论面（慢变量、窄幅偏置）     | 新闻+大模型  | news + policy|

为什么把日线那几路搬进分钟级
----------------------------
日线决策层（``SIGNAL_WEIGHTS``）原本用 K 线能量/形态/量能/拐点/模型五类证据，
但实时引擎早期只接了盘口与动量，等于把"能解释的证据"丢了一大半。
现在它们都在分钟频率上重新实现，**不是复用日线的值**：
- energy / pattern / volume 用 1 分钟 K 线重算，频率与预测目标对齐；
- turn 用日内拐点模型（RSI + 连涨连跌 + 量能衰减）；
- model 复用日线多模型的方向概率，但它只有日线分辨率，故硬封顶 ±0.35σ，只做慢变量偏置。
日线证据对 5 分钟价格几乎无解释力，给大权重就是自欺；把幅度压住、把权重交给 Hedge。

为什么用在线 Hedge 而不是固定权重
----------------------------------
5 分钟级别的"哪一路有效"会随行情状态漂移（趋势日动量有效、震荡日反转有效），
固定权重必然在风格切换时失灵。这里按「每路预测的已实现绝对误差」更新权重：

    w_i ← w_i · exp(-η · |pred_i - realized| / scale)

再与先验收缩（``w = 0.82·w + 0.18·prior``），防止某一路因为一段坏运气被永久归零。
所有权重与命中率都会落库，页面上可以直接看到"这一路最近 200 次命中 51%"。

诚实的边界
----------
- 方向命中率长期在 50~55%，MAE 通常大于预测幅度本身；本模块**不承诺盈利**。
- 置信度低或专家分歧大时，输出"震荡"，并把有效区间放大 —— 这比硬给方向有用。
- 大模型那一路权重最低（先验 0.08），且幅度硬封顶，不允许它主导结果。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import config
from . import db
from .energy import candle_energy

EPS = 1e-9


# =====================================================================
# 工具
# =====================================================================
def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def _tanh(x: float) -> float:
    return math.tanh(max(-8.0, min(8.0, x)))


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = sum(xs) / len(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def _norm_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


@dataclass
class ExpertOut:
    name: str
    cn: str
    pred_bps: float
    conf: float
    note: str
    available: bool = True

    def as_dict(self) -> dict:
        return {"name": self.name, "cn": self.cn, "pred_bps": round(self.pred_bps, 2),
                "conf": round(self.conf, 3), "note": self.note,
                "available": self.available}


# =====================================================================
# 在线岭回归（纯 Python，无 numpy 依赖，d≈12 时闭式解足够快）
# =====================================================================
ML_FEATURES = ["ret1", "ret3", "ret5", "ret10", "vwap_dev", "range_pos",
               "vol_ratio", "up_vol_ratio", "rsi14", "min_sin", "min_cos", "gap"]


def _ml_vector(bars: list[dict], i: int) -> list[float] | None:
    """用第 i 根 bar 及之前的信息构造特征向量（严格无未来数据）。"""
    if i < 22:
        return None
    closes = [float(b["close"]) for b in bars]
    vols = [float(b.get("volume") or 0) for b in bars]
    c = closes[i]
    if c <= 0:
        return None

    def ret(k):
        return (c / closes[i - k] - 1) * 1e4 if i - k >= 0 and closes[i - k] > 0 else 0.0

    amt = sum(float(b.get("amount") or 0) for b in bars[:i + 1])
    vol = sum(vols[:i + 1])
    vwap = (amt / vol) if (amt > 0 and vol > 0) else c
    hi = max(float(b.get("high") or b["close"]) for b in bars[max(0, i - 30):i + 1])
    lo = min(float(b.get("low") or b["close"]) for b in bars[max(0, i - 30):i + 1])
    v5 = sum(vols[max(0, i - 4):i + 1]) / max(1, len(vols[max(0, i - 4):i + 1]))
    v30 = sum(vols[max(0, i - 29):i + 1]) / max(1, len(vols[max(0, i - 29):i + 1]))
    up = sum(v for a, b, v in zip(closes[max(0, i - 10):i], closes[max(0, i - 9):i + 1],
                                  vols[max(0, i - 9):i + 1]) if b > a)
    tot = sum(vols[max(0, i - 9):i + 1])
    # 日内 RSI(14)
    gains = losses = 0.0
    for a, b in zip(closes[max(0, i - 14):i], closes[max(0, i - 13):i + 1]):
        d = b - a
        gains += max(0.0, d)
        losses += max(0.0, -d)
    rsi = 100 - 100 / (1 + (gains / 14) / ((losses / 14) + EPS))
    minute = int(bars[i].get("minute") or 0)
    if not minute:
        minute = _minute_from_ts(str(bars[i].get("ts") or ""))
    ang = 2 * math.pi * minute / 240.0
    prev_close = closes[max(0, i - 1)]
    open_i = float(bars[i].get("open") or c)
    return [
        ret(1), ret(3), ret(5), ret(10),
        (c / vwap - 1) * 1e4 if vwap else 0.0,
        (c - lo) / (hi - lo) if hi > lo else 0.5,
        (v5 / v30 - 1) if v30 > 0 else 0.0,
        (up / tot - 0.5) if tot > 0 else 0.0,
        rsi - 50.0,
        math.sin(ang), math.cos(ang),
        (open_i / prev_close - 1) * 1e4 if prev_close else 0.0,
    ]


def _minute_from_ts(ts: str) -> int:
    try:
        hm = ts[11:16]
        h, m = int(hm[:2]), int(hm[3:5])
    except Exception:
        return 0
    mins = h * 60 + m
    if 9 * 60 + 30 <= mins <= 11 * 60 + 30:
        return mins - (9 * 60 + 30)
    if 13 * 60 <= mins <= 15 * 60:
        return 120 + mins - 13 * 60
    return 0


def _solve(a: list[list[float]], b: list[float]) -> list[float] | None:
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= pv
        for r in range(n):
            if r == col:
                continue
            f = m[r][col]
            if f == 0:
                continue
            for j in range(col, n + 1):
                m[r][j] -= f * m[col][j]
    return [m[i][n] for i in range(n)]


@dataclass
class RidgeFit:
    w: list[float]
    mu: list[float]
    sd: list[float]
    n: int
    mae_bps: float
    hit: float
    trained_at: float = 0.0


def fit_ridge(samples: list[tuple[list[float], float]], lam: float = 2.0) -> RidgeFit | None:
    if len(samples) < 20:
        return None
    d = len(samples[0][0])
    mu = [sum(x[j] for x, _ in samples) / len(samples) for j in range(d)]
    sd = []
    for j in range(d):
        v = math.sqrt(sum((x[j] - mu[j]) ** 2 for x, _ in samples) / max(1, len(samples) - 1))
        sd.append(v if v > 1e-9 else 1.0)
    xs = [[(x[j] - mu[j]) / sd[j] for j in range(d)] + [1.0] for x, _ in samples]
    ys = [y for _, y in samples]
    p = d + 1
    ata = [[0.0] * p for _ in range(p)]
    aty = [0.0] * p
    for x, y in zip(xs, ys):
        for i in range(p):
            aty[i] += x[i] * y
            for j in range(p):
                ata[i][j] += x[i] * x[j]
    for i in range(d):
        ata[i][i] += lam
    w = _solve(ata, aty)
    if w is None:
        return None
    preds = [sum(wi * xi for wi, xi in zip(w, x)) for x in xs]
    errs = [abs(p - y) for p, y in zip(preds, ys)]
    mae = sum(errs) / len(errs)
    hit = sum(1 for p, y in zip(preds, ys) if (p > 0) == (y > 0)) / len(ys)
    return RidgeFit(w=w, mu=mu, sd=sd, n=len(samples), mae_bps=mae, hit=hit,
                    trained_at=time.time())


def ridge_predict(fit: RidgeFit, x: list[float]) -> float:
    z = [(x[j] - fit.mu[j]) / fit.sd[j] for j in range(len(x))] + [1.0]
    return sum(wi * zi for wi, zi in zip(fit.w, z))


class OnlineML:
    """按周期分别训练（1/3/5 分钟各一个岭回归），从历史分钟线库取样本。"""

    def __init__(self, symbol: str, horizons: list[int]):
        self.symbol = symbol
        self.horizons = horizons
        self.fits: dict[int, RidgeFit] = {}
        self.last_fit = 0.0
        self.note = "未训练"
        self.refit_sec = 300.0

    def maybe_fit(self, extra_bars: list[dict] | None = None, force: bool = False) -> bool:
        if not force and time.time() - self.last_fit < self.refit_sec and self.fits:
            return True
        days = int((getattr(config, "LIVE", {}) or {}).get("history_days", 10))
        rows = db.intraday_bars(self.symbol, limit=days * 260 + 300)
        by_day: dict[str, list[dict]] = {}
        for r in rows:
            by_day.setdefault(r.get("trade_date") or str(r["ts"])[:10], []).append(r)
        if extra_bars:
            by_day.setdefault(str(extra_bars[0].get("trade_date") or "")[:10],
                              list(extra_bars))
        min_n = int((getattr(config, "LIVE", {}) or {}).get("min_samples_ml", 45))
        total = 0
        for h in self.horizons:
            samples: list[tuple[list[float], float]] = []
            for _, bars in by_day.items():
                bars = sorted(bars, key=lambda b: str(b["ts"]))
                closes = [float(b["close"]) for b in bars]
                for i in range(22, len(bars) - h):
                    v = _ml_vector(bars, i)
                    if v is None or closes[i] <= 0:
                        continue
                    y = (closes[i + h] / closes[i] - 1) * 1e4
                    if abs(y) > 400:        # 数据异常（除权/错误价）直接丢
                        continue
                    samples.append((v, y))
            if len(samples) >= min_n:
                # 用最后 30% 做时间序验证，避免"样本内 MAE"虚低
                cut = int(len(samples) * 0.7)
                fit = fit_ridge(samples[:cut] if cut >= 30 else samples)
                if fit:
                    if cut >= 30:
                        verr, vhit = [], []
                        for x, y in samples[cut:]:
                            p = ridge_predict(fit, x)
                            verr.append(abs(p - y))
                            vhit.append(1 if (p > 0) == (y > 0) else 0)
                        fit.mae_bps = sum(verr) / len(verr)
                        fit.hit = sum(vhit) / len(vhit)
                        fit.n = len(samples)
                    self.fits[h] = fit
                    total += len(samples)
            else:
                self.fits.pop(h, None)
        self.last_fit = time.time()
        self.note = (f"样本 {total} 条 / {len(by_day)} 天" if self.fits
                     else f"样本不足（{total} < {min_n * len(self.horizons)}）")
        return bool(self.fits)

    def predict(self, ctx: dict) -> tuple[ExpertOut | None, dict]:
        """返回主周期预测 + 各周期明细。"""
        feats = ctx.get("ml_x")
        if not feats:
            # 模型可能已经训练好了，但当下构不出特征向量（开盘前 23 根分钟线不足），
            # 这两种情况必须区分开，否则页面显示"样本 651 条"却不出预测，看着像 bug。
            return None, {"ok": False,
                          "note": "分钟线不足 23 根，暂时构不出特征向量（开盘约 23 分钟后可用）"
                                  f"｜{self.note}"}
        per = {}
        for h, fit in self.fits.items():
            p = ridge_predict(fit, feats)
            per[h] = {"pred_bps": round(p, 2), "mae_bps": round(fit.mae_bps, 2),
                      "hit": round(fit.hit, 3), "n": fit.n}
        return None, {"ok": bool(per), "per_h": per, "note": self.note}


# =====================================================================
# 专家实现
# =====================================================================
def _sigma_h(ctx: dict, h: int) -> float:
    return max(1.0, float(ctx.get("sigma_min") or 3.0) * math.sqrt(max(1, h)))


def expert_micro(ctx: dict, h: int) -> ExpertOut:
    imb = ctx.get("depth_imb")
    if imb is None:
        return ExpertOut("micro", "五档不平衡", 0.0, 0.0, "无盘口（非交易时段）", False)
    sig = _sigma_h(ctx, h)
    pred = _clip(28.0 * imb * math.sqrt(h / 5.0), -1.5 * sig, 1.5 * sig)
    conf = 0.35 + 0.4 * min(1.0, abs(imb) * 2.0)
    note = f"委比 {ctx.get('weibi', 0):+.1f}%，不平衡 {imb:+.2f}"
    return ExpertOut("micro", "五档不平衡", pred, conf, note)


def expert_flow(ctx: dict, h: int) -> ExpertOut:
    ofi = ctx.get("ofi")
    n = int(ctx.get("ofi_n") or 0)
    if ofi is None or n < 10:
        return ExpertOut("flow", "主动买卖差", 0.0, 0.0, "tick 样本不足", False)
    sig = _sigma_h(ctx, h)
    pred = _clip(26.0 * ofi * math.sqrt(h / 5.0), -1.4 * sig, 1.4 * sig)
    conf = 0.3 + 0.4 * min(1.0, n / 60.0)
    note = f"主动买卖差 {ofi:+.2f}（{n} 个快照）"
    return ExpertOut("flow", "主动买卖差", pred, conf, note)


def expert_volume(ctx: dict, h: int) -> ExpertOut:
    """量能：不猜方向，只看「涨的那几分钟是不是带着量」。

    单纯"放量"对方向毫无意义（放量下跌也是放量），所以这里用**量价配合**：
    近 20 分钟上涨分钟的量占比 tilt，乘以量能放大系数 amp。缩量时 amp→0，
    预测自然收敛到 0，而不是硬给一个方向。
    """
    vr = ctx.get("vol_ratio")
    up = ctx.get("up_vol_ratio")
    if vr is None or up is None:
        return ExpertOut("volume", "量价配合", 0.0, 0.0, "量能样本不足", False)
    sig = _sigma_h(ctx, h)
    tilt = _clip((float(up) - 0.5) * 2.0, -1.0, 1.0)
    amp = _clip((float(vr) - 0.9) / 0.9, 0.0, 1.0)
    pred = _clip(tilt * (0.22 + 0.5 * amp) * sig * math.sqrt(h / 5.0),
                 -1.1 * sig, 1.1 * sig)
    conf = 0.22 + 0.28 * abs(tilt) + 0.15 * amp
    v = "同向偏多" if tilt > 0.3 else ("同向偏空" if tilt < -0.3 else "方向不清")
    note = (f"量比 {float(vr):.2f}，上涨分钟量占比 {float(up) * 100:.0f}%"
            f"（{'放量' if amp > 0.3 else '量能一般'}，量价{v}）")
    return ExpertOut("volume", "量价配合", pred, conf, note)


def expert_energy(ctx: dict, h: int) -> ExpertOut:
    """K 线能量（原 energy.py 的量化搬到分钟频率）。

    取最近 3 根分钟线，按 0.55/0.30/0.15 加权（越新越重要），每根用
    ``candle_energy`` 算 signed∈[-100,100]：实体方向 × 实体占比 × 收盘位置 × 量能放大，
    再按振幅相对 ATR 缩放。长影线且量能不小会被判为"多空衰竭"并打折。
    """
    bars = ctx.get("bars") or []
    if len(bars) < 2:
        return ExpertOut("energy", "K线能量", 0.0, 0.0, "K 线不足 2 根", False)
    sig = _sigma_h(ctx, h)
    atr_norm = max(0.0008, float(ctx.get("sigma_min") or 3.0) / 1e4)
    vr = float(ctx.get("vol_ratio") or 1.0)
    ws = [0.55, 0.30, 0.15][:max(1, min(3, len(bars)))]
    tot = wsum = 0.0
    last_label = ""
    for k, w in enumerate(ws):
        b = bars[-1 - k]
        c = float(b.get("close") or 0)
        if c <= 0:
            continue
        e = candle_energy(b.get("open") or c, b.get("high") or c,
                          b.get("low") or c, c, vol_ratio=vr, atr_norm=atr_norm)
        tot += float(e.get("signed") or 0.0) * w
        wsum += w
        if k == 0:
            last_label = str(e.get("label") or "")
    if wsum <= 0:
        return ExpertOut("energy", "K线能量", 0.0, 0.0, "K 线数据异常", False)
    s = tot / wsum
    pred = _clip(s / 100.0 * 0.85 * sig * math.sqrt(h / 5.0), -1.15 * sig, 1.15 * sig)
    conf = 0.24 + 0.42 * min(1.0, abs(s) / 55.0)
    note = f"近 {len(ws)} 根能量 {s:+.0f}/100（最新一根：{last_label or '无数据'}）"
    return ExpertOut("energy", "K线能量", pred, conf, note)


def expert_pattern(ctx: dict, h: int) -> ExpertOut:
    """分钟级形态：连阳连阴的过度延伸、长影线、双顶双底。

    与 mom 的分工：mom 顺动量，pattern 找**反转**。两者经常对着干，
    这正是 Hedge 存在的意义 —— 谁在当下这段行情里更准，权重就往谁那边走。
    """
    bars = ctx.get("bars") or []
    if len(bars) < 8:
        return ExpertOut("pattern", "形态", 0.0, 0.0, "K 线不足 8 根", False)
    sig = _sigma_h(ctx, h)
    n = len(bars)
    closes = [float(b["close"]) for b in bars]
    opens = [float(b.get("open") or b["close"]) for b in bars]
    highs = [float(b.get("high") or b["close"]) for b in bars]
    lows = [float(b.get("low") or b["close"]) for b in bars]
    score, cues = 0.0, []

    # 1) 连续同向的过度延伸（≥4 根才谈过热，方向取反向）
    run = 0
    for i in range(n - 1, 0, -1):
        d = 1 if closes[i] > closes[i - 1] else (-1 if closes[i] < closes[i - 1] else 0)
        if d == 0 or (run and d * run < 0):
            break
        run += d
    if abs(run) >= 4:
        score -= 0.07 * (abs(run) - 3) * (1 if run > 0 else -1)
        cues.append(f"连{'涨' if run > 0 else '跌'} {abs(run)} 分钟，短线过热")

    # 2) 最新一根的长影线（上影=抛压，下影=承接）
    rng = max(highs[-1] - lows[-1], 1e-9)
    mid = (highs[-1] + lows[-1]) / 2.0
    upper = (highs[-1] - max(opens[-1], closes[-1])) / rng
    lower = (min(opens[-1], closes[-1]) - lows[-1]) / rng
    if upper > 0.5 and closes[-1] < mid:
        score -= 0.40
        cues.append(f"长上影 {upper * 100:.0f}%，冲高回落")
    elif lower > 0.5 and closes[-1] > mid:
        score += 0.40
        cues.append(f"长下影 {lower * 100:.0f}%，下探回升")

    # 3) 双顶/双底：近 30 根前后半段的极值几乎重合，且当前未突破
    m = min(30, n)
    half = max(2, m // 2)
    h1, h2 = max(highs[-m:-half]), max(highs[-half:])
    l1, l2 = min(lows[-m:-half]), min(lows[-half:])
    if h2 > 0 and abs(h2 - h1) / h2 < 0.0008 and closes[-1] < h2 * 0.9995:
        score -= 0.30
        cues.append("双顶（两次冲高同一位置未破）")
    elif l2 > 0 and abs(l1 - l2) / l2 < 0.0008 and closes[-1] > l2 * 1.0005:
        score += 0.30
        cues.append("双底（两次探底同一位置未破）")

    if not cues:
        return ExpertOut("pattern", "形态", 0.0, 0.0, "无明显形态（横盘或结构不清）", False)
    pred = _clip(score * sig * math.sqrt(h / 5.0), -1.0 * sig, 1.0 * sig)
    conf = 0.22 + 0.26 * min(1.0, abs(score))
    return ExpertOut("pattern", "形态", pred, conf, "；".join(cues[:2]))


def expert_mom(ctx: dict, h: int) -> ExpertOut:
    slope = ctx.get("tick_slope")
    if slope is None:
        r5 = ctx.get("ret5")
        if r5 is None:
            return ExpertOut("mom", "动量延续", 0.0, 0.0, "无足够行情", False)
        slope = r5 / 5.0
    sig = _sigma_h(ctx, h)
    pred = _clip(0.35 * slope * h, -1.2 * sig, 1.2 * sig)
    conf = min(1.0, abs(slope) / (0.6 * float(ctx.get("sigma_min") or 3.0) + EPS))
    conf = 0.25 + 0.5 * conf
    note = f"秒级斜率 {slope:+.2f} bps/分钟 → 线性外推×0.35 阻尼"
    return ExpertOut("mom", "动量延续", pred, conf, note)


def expert_revert(ctx: dict, h: int) -> ExpertOut:
    dev = ctx.get("vwap_dev")
    if dev is None:
        return ExpertOut("revert", "VWAP 回复", 0.0, 0.0, "无 VWAP", False)
    sig = _sigma_h(ctx, h)
    pred = _clip(-dev * (1 - math.exp(-h / 12.0)) * 0.35, -1.0 * sig, 1.0 * sig)
    conf = 0.25 + 0.45 * min(1.0, abs(dev) / (2 * sig + EPS))
    note = f"偏离 VWAP {dev:+.1f} bps，半衰期约 12 分钟"
    return ExpertOut("revert", "VWAP 回复", pred, conf, note)


def expert_range(ctx: dict, h: int) -> ExpertOut:
    pos = ctx.get("range_pos")
    vr = ctx.get("vol_ratio")
    if pos is None or vr is None:
        return ExpertOut("range", "区间突破", 0.0, 0.0, "样本不足", False)
    sig = _sigma_h(ctx, h)
    pred, note = 0.0, ""
    if pos >= 0.85:
        if vr >= 1.4:
            pred = 0.45 * sig
            note = f"贴近日高（位置 {pos:.2f}）且量能放大 {vr:.2f} → 突破延续"
        else:
            pred = -0.5 * sig
            note = f"贴近日高但量能不足（{vr:.2f}）→ 冲高回落概率大"
    elif pos <= 0.15:
        if vr >= 1.4:
            pred = -0.45 * sig
            note = f"贴近日低（位置 {pos:.2f}）且放量 → 破位延续"
        else:
            pred = 0.5 * sig
            note = f"贴近日低但缩量（{vr:.2f}）→ 抛压衰竭，易反抽"
    else:
        pred = (0.5 - pos) * 0.4 * sig
        note = f"区间中部（位置 {pos:.2f}），向中枢回归"
    return ExpertOut("range", "区间突破", pred, 0.32, note)


def expert_season(ctx: dict, h: int) -> ExpertOut:
    sb = ctx.get("season_bps")
    n = int(ctx.get("season_n") or 0)
    if sb is None or n < 3:
        return ExpertOut("season", "日内季节性", 0.0, 0.0, "历史样本不足", False)
    shrink = n / (n + 20.0)
    pred = sb * shrink * math.sqrt(h / 5.0)
    conf = 0.1 + 0.3 * shrink
    note = f"该时刻历史平均 {sb:+.1f} bps（{n} 个样本，收缩 {shrink:.2f}）"
    return ExpertOut("season", "日内季节性", pred, conf, note)


def expert_llm(ctx: dict, h: int) -> ExpertOut:
    r = ctx.get("llm") or {}
    if not r.get("ok"):
        return ExpertOut("llm", "大模型预测", 0.0, 0.0, r.get("note") or "未启用", False)
    age = max(0.0, time.time() - float(r.get("_ts") or 0))
    interval = float((getattr(config, "LIVE", {}) or {}).get("llm_interval", 300))
    decay = math.exp(-age / (2.5 * max(30.0, interval)))
    if decay < 0.15:
        return ExpertOut("llm", "大模型预测", 0.0, 0.0, f"结果已过期（{age/60:.0f} 分钟前）", False)
    base = float(r.get("bps") or 0)
    pred = base * math.sqrt(h / 5.0) * max(0.4, decay)
    conf = max(0.0, min(1.0, float(r.get("confidence") or 0))) * decay
    return ExpertOut("llm", "大模型预测", pred, conf,
                     f"{r.get('model', '')}：{r.get('reason', '')[:40]}（{age:.0f}s 前）")


def expert_news(ctx: dict, h: int) -> ExpertOut:
    """新闻 / 政策 / 舆论面 —— 慢变量，只给窄幅方向偏置。

    刻意做成"幅度小、置信低"：新闻对 5 分钟价格的解释力本来就弱，
    如果让它在分钟级拿到和盘口同等的发言权，那就是在编故事。
    真正的权重交给 Hedge 按结算误差决定（配置里的 0.06 只是起手先验）。
    """
    f = ctx.get("news") or {}
    if not f.get("ok") or not int(f.get("news_count") or 0):
        return ExpertOut("news", "新闻政策面", 0.0, 0.0,
                         f.get("note") or "暂无新闻数据", False)
    decay = float(f.get("decay") or 0.0)
    if decay < 0.2:
        age_min = float(f.get("age_sec") or 0) / 60.0
        return ExpertOut("news", "新闻政策面", 0.0, 0.0,
                         f"新闻已过期（{age_min:.0f} 分钟前）", False)
    score = float(f.get("decayed_score") or 0.0)
    sig = _sigma_h(ctx, h)
    # 慢变量上限：5 分钟窗最多 0.35 sigma 的方向偏置，长周期略放大
    cap = 0.35 * sig * (1.0 + 0.15 * max(0, h - 1))
    pred = _clip(score * cap, -cap, cap)
    conf = 0.5 * float(f.get("conf") or 0.0) * (0.4 + 0.6 * decay)
    src = "大模型" if f.get("has_llm") else "词典情绪"
    note = (f"{f.get('stance', '中性')} {score:+.2f}（{f.get('news_count')} 条 / {src}，"
            f"{float(f.get('age_sec') or 0)/60:.0f} 分钟前，衰减 {decay:.2f}）")
    return ExpertOut("news", "新闻政策面", pred, min(0.5, conf), note)


def expert_ml(ctx: dict, h: int) -> ExpertOut:
    m = (ctx.get("ml") or {}).get("per_h", {}).get(h)
    if not m:
        return ExpertOut("ml", "在线岭回归", 0.0, 0.0,
                         (ctx.get("ml") or {}).get("note") or "样本不足", False)
    hit = float(m.get("hit") or 0.5)
    # 只用样本外超过随机的部分给置信度：50% 命中 → 0.12，60% → 0.32
    conf = max(0.08, min(0.62, 0.12 + 2.0 * max(0.0, hit - 0.5)))
    note = (f"岭回归 {m['pred_bps']:+.1f} bps（样本外 MAE {m['mae_bps']:.1f}，"
            f"命中 {hit*100:.1f}%，n={m['n']}）")
    return ExpertOut("ml", "在线岭回归", float(m["pred_bps"]), conf, note)


def expert_model(ctx: dict, h: int) -> ExpertOut:
    """日线多模型的方向概率（RF + GBDT + LSTM 融合，来自 daily_signals）。

    只有日线分辨率，对 5 分钟价格几乎没有解释力，所以这里**硬封顶 ±0.35σ**、
    置信度也压在 0.14~0.36。它的价值是"别在明显看多的日子里做空 5 分钟"，
    而不是提供方向。权重交给 Hedge 按误差决定。
    """
    d = ctx.get("daily_model") or {}
    if not d.get("ok"):
        return ExpertOut("model", "日线模型", 0.0, 0.0,
                         d.get("note") or "无日线模型结论", False)
    v = _clip(float(d.get("value") or 0.0), -1.0, 1.0)
    if abs(v) < 0.05:
        return ExpertOut("model", "日线模型", 0.0, 0.0,
                         f"{d.get('date', '')} 日线模型无明确方向", False)
    sig = _sigma_h(ctx, h)
    cap = 0.35 * sig * (1.0 + 0.1 * max(0, h - 1))
    pred = _clip(v * cap, -cap, cap)
    conf = 0.14 + 0.22 * abs(v)
    return ExpertOut("model", "日线模型", pred, conf,
                     f"{d.get('date', '')} 日线多模型方向 {v:+.2f}（慢变量，封顶 ±0.35σ）")


def expert_turn(ctx: dict, h: int) -> ExpertOut:
    """拐点风险：顶部临近为负、底部临近为正。

    两段证据：
    1. ``next``（日内拐点模型）给出 kind / in_minutes / conf，只有拐点落在本预测
       窗口内才计分 —— 30 分钟后才见顶，对 5 分钟预测没有意义。
    2. ``extremes`` 给今日高/低点的"已出现概率"，贴近日高且已现概率 ≥85% 时上方空间很小。
    """
    t = ctx.get("turning") or {}
    if not t:
        return ExpertOut("turn", "拐点风险", 0.0, 0.0, "拐点模型未就绪", False)
    sig = _sigma_h(ctx, h)
    score, cues = 0.0, []
    nxt = t.get("next") or {}
    kind = str(nxt.get("kind") or "")
    if kind in ("顶", "底"):
        eta = max(1, int(nxt.get("in_minutes") or 1))
        c0 = _clip(float(nxt.get("conf") or 0.0), 0.0, 0.8)
        overlap = _clip((h - eta + 1.0) / max(1.0, float(h)), 0.0, 1.0)
        if overlap > 0:
            mag = (0.22 + 0.55 * c0) * overlap
            score += -mag if kind == "顶" else mag
            cues.append(f"约 {eta} 分钟后可能见{kind}（置信 {c0:.2f}）")
    ext = t.get("extremes") or {}
    pos = float(ctx.get("range_pos") if ctx.get("range_pos") is not None else 0.5)
    hi, lo = ext.get("high") or {}, ext.get("low") or {}
    if hi and float(hi.get("prob_done") or 0) >= 0.85 and pos >= 0.9:
        score -= 0.18
        cues.append("日高已现概率≥85% 且贴近日高")
    if lo and float(lo.get("prob_done") or 0) >= 0.85 and pos <= 0.1:
        score += 0.18
        cues.append("日低已现概率≥85% 且贴近日低")
    if not cues:
        return ExpertOut("turn", "拐点风险", 0.0, 0.0, "无临近拐点", False)
    pred = _clip(score * sig * math.sqrt(h / 5.0), -1.2 * sig, 1.2 * sig)
    conf = 0.18 + 0.3 * min(1.0, abs(score))
    return ExpertOut("turn", "拐点风险", pred, conf, "；".join(cues))


EXPERTS = {
    "micro": expert_micro, "flow": expert_flow,
    "volume": expert_volume, "energy": expert_energy, "pattern": expert_pattern,
    "range": expert_range, "mom": expert_mom, "revert": expert_revert,
    "season": expert_season, "ml": expert_ml, "model": expert_model,
    "turn": expert_turn, "llm": expert_llm, "news": expert_news,
}
CN = {"micro": "五档不平衡", "flow": "主动买卖差",
      "volume": "量价配合", "energy": "K线能量", "pattern": "K线形态",
      "range": "区间突破", "mom": "动量延续", "revert": "VWAP 回复",
      "season": "日内季节性", "ml": "在线岭回归", "model": "日线模型",
      "turn": "拐点风险", "llm": "大模型预测", "news": "新闻政策面"}


# =====================================================================
# 集成
# =====================================================================
class Ensemble:
    def __init__(self, symbol: str, horizons: list[int] | None = None):
        self.symbol = symbol
        self.priors = dict(getattr(config, "LIVE_EXPERT_PRIORS", {})) or {k: 1.0 for k in EXPERTS}
        self.horizons = horizons or list((getattr(config, "LIVE", {}) or {}).get("horizons_min", [1, 3, 5]))
        self.states: dict[str, dict] = {}
        self._load()

    def _load(self):
        saved = db.load_expert_state(self.symbol)
        for name, prior in self.priors.items():
            s = saved.get(name) or {}
            self.states[name] = {
                "name": name, "cn": CN.get(name, name),
                "prior": float(prior),
                "weight": float(s.get("weight") or prior),
                "n": int(s.get("n") or 0),
                "mae_bps": float(s.get("mae_bps") or 0.0),
                "hit": float(s.get("hit") or 0.5),
                "pred_bps": float(s.get("pred_bps") or 0.0),
            }
        self._normalize()

    def _normalize(self):
        tot = sum(s["weight"] for s in self.states.values()) or 1.0
        for s in self.states.values():
            s["weight"] = s["weight"] / tot

    def reliability(self, name: str) -> float:
        st = self.states[name]
        if st["n"] < 8:
            return 1.0
        # MAE 越小越可信；以 8bps 为参考尺度，命中率低于 50% 再打折
        rel = 1.0 / (1.0 + st["mae_bps"] / 8.0)
        rel *= 0.7 + 0.6 * max(0.0, min(1.0, st["hit"]))
        return max(0.15, min(1.5, rel))

    def predict(self, ctx: dict) -> dict:
        """对每个周期做加权，返回主周期与明细。"""
        per_h: dict[int, dict] = {}
        for h in self.horizons:
            outs = []
            for name, fn in EXPERTS.items():
                try:
                    o = fn(ctx, h)
                except Exception as e:
                    o = ExpertOut(name, CN.get(name, name), 0.0, 0.0, f"计算异常：{e}", False)
                outs.append(o)
            active = [o for o in outs if o.available]
            # 有效权重 = 先验×可靠性×专家自评置信（置信低就不该拿满权重）
            ws = {o.name: self.states[o.name]["weight"] * self.reliability(o.name)
                  * (0.35 + 0.65 * o.conf) for o in active}
            wsum = sum(ws.values())
            if wsum <= 0:
                pred = 0.0
                disp = 0.0
            else:
                pred = sum(o.pred_bps * ws[o.name] for o in active) / wsum
                disp = math.sqrt(sum(ws[o.name] * (o.pred_bps - pred) ** 2 for o in active) / wsum)
            sig = _sigma_h(ctx, h)
            # 分歧越大越向 0 收缩：分歧 > 1 个 sigma 时几乎不敢表态
            shrink = 1.0 / (1.0 + (disp / (sig + EPS)))
            pred *= 0.55 + 0.45 * shrink
            sigma = max(sig * 0.85, disp *
                        0.9, float(ctx.get("sigma_min") or 3.0) * math.sqrt(h))
            z = float((getattr(config, "LIVE", {}) or {}).get("band_z", 1.28))
            price = float(ctx.get("price") or 0)
            lo = price * (1 + (pred - z * sigma) / 1e4) if price else 0.0
            hi = price * (1 + (pred + z * sigma) / 1e4) if price else 0.0
            no_skill = abs(pred) < max(1.5, 0.25 * sigma)
            if no_skill:
                direction = "震荡"
            elif pred > 0:
                direction = "看涨"
            else:
                direction = "看跌"
            conf = min(1.0, abs(pred) / (sigma + EPS) * 1.4) * (0.6 + 0.4 * shrink)
            per_h[h] = {
                "h": h, "pred_bps": round(pred, 2),
                "pred_price": round(price * (1 + pred / 1e4), 2) if price else 0.0,
                "lo": round(lo, 2), "hi": round(hi, 2),
                "sigma_bps": round(sigma, 2), "dispersion_bps": round(disp, 2),
                "direction": direction, "conf": round(max(0.0, conf), 3),
                "no_skill": bool(no_skill),
                "experts": [{
                    **o.as_dict(),
                    "weight": round(self.states[o.name]["weight"], 4),
                    "eff_weight": round(ws.get(o.name, 0.0) / wsum, 4) if wsum else 0.0,
                    "contrib": round((o.pred_bps * ws.get(o.name, 0.0) / wsum) if wsum else 0.0, 2),
                    "hit": round(self.states[o.name]["hit"], 3),
                    "mae_bps": round(self.states[o.name]["mae_bps"], 2),
                    "n": self.states[o.name]["n"],
                } for o in outs],
            }
        primary = per_h.get(int((getattr(config, "LIVE", {}) or {}).get("primary_horizon", 5)))
        if primary is None and per_h:
            primary = per_h[sorted(per_h)[len(per_h) // 2]]
        return {"horizons": per_h, "primary": primary or {}, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}

    def update(self, realized_bps: float, experts: list[dict], sigma_bps: float):
        """用一次已实现结果更新各专家权重（Hedge）与统计量。"""
        scale = max(8.0, float(sigma_bps or 8.0))
        for e in experts or []:
            name = e.get("name")
            st = self.states.get(name)
            if not st or not e.get("available"):
                continue
            err = abs(float(e.get("pred_bps") or 0) - realized_bps)
            loss = min(3.0, err / scale)
            st["weight"] = st["weight"] * math.exp(-0.35 * loss)
            # 指数移动平均：既要跟得上风格切换，又不能被单次异常带跑
            st["n"] += 1
            if st["n"] <= 1:
                st["mae_bps"] = err
            else:
                st["mae_bps"] = 0.92 * st["mae_bps"] + 0.08 * err
            h = 1.0 if (float(e.get("pred_bps") or 0) > 0) == (realized_bps > 0) else 0.0
            st["hit"] = h if st["n"] <= 1 else 0.93 * st["hit"] + 0.07 * h
        # 与先验收缩，防止某一路被永久归零
        total_prior = sum(self.states[n]["prior"] for n in self.states) or 1.0
        for name, st in self.states.items():
            pri = st["prior"] / total_prior
            st["weight"] = max(0.02, 0.82 * st["weight"] + 0.18 * pri)
        self._normalize()

    def save(self):
        db.save_expert_state(self.symbol, list(self.states.values()))

    def snapshot(self) -> list[dict]:
        rows = sorted(self.states.values(), key=lambda s: -s["weight"])
        return [{"name": s["name"], "cn": s["cn"], "weight": round(s["weight"], 4),
                 "prior": s["prior"], "n": s["n"],
                 "mae_bps": round(s["mae_bps"], 2), "hit": round(s["hit"], 3)}
                for s in rows]


def build_ctx(features: dict, bars: list[dict], season: dict | None,
              llm_res: dict | None, ml: OnlineML | None,
              news: dict | None = None, turning: dict | None = None,
              daily_model: dict | None = None) -> dict:
    """组装专家上下文（含岭回归特征向量、形态用的分钟线、拐点与日线模型结论）。"""
    ctx = dict(features or {})
    if bars:
        i = len(bars) - 1
        ctx["ml_x"] = _ml_vector(bars, i)
        # energy / pattern 需要原始 K 线；只留尾部 40 根（pattern 最长看 30 根）
        ctx["bars"] = bars[-40:]
    if season:
        ctx["season"] = season
    if llm_res:
        ctx["llm"] = llm_res
    if news:
        ctx["news"] = news
    if turning:
        ctx["turning"] = turning
    if daily_model:
        ctx["daily_model"] = daily_model
    if ml is not None:
        ctx["ml"] = ml.predict(ctx)[1]
    return ctx
