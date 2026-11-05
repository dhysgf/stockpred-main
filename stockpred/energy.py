"""K 线能量与买卖压力：把「一根 K 线有多强」「当下是买盘还是卖盘在推」量化。

为什么单独建一层
----------------
原来的量能分（volume.volume_score）混了 OBV/MFI/量比/资金流，解释得清但**不可分解**：
用户问"这根阳线到底有多强"时，回答不了。这里把两件事拆开：

- **K 线能量**（``candle_energy``）：只看这一根 K 线的内部结构 ——
  实体占比、收盘在区间的位置、影线、相对量能。输出 0~100 的强度与方向标签。
  它回答"这根 K 线本身有多强"，跨股票可比（全部做了归一化）。
- **买卖压力**（``pressure_score``）：只看当下谁在主动买/卖 ——
  五档委比、tick 主动买卖差、主力大单净额、量价配合。输出 0~1（0.5 中性）。
  它回答"现在盘口是谁在推"。

两者都只作为**决策层证据**，不做模型特征：五档与资金流没有历史序列，
进了模型就会变成常量列（详见 volume.py 顶部说明）。
"""
from __future__ import annotations

import math


def _f(x, default=0.0):
    try:
        if x is None:
            return default
        v = float(x)
        return default if v != v else v
    except Exception:
        return default


def candle_energy(o: float, h: float, l: float, c: float,
                  vol_ratio: float = 1.0, atr_norm: float = 0.02) -> dict:
    """单根 K 线能量（0~100）与多空标签。

    强度 = 实体方向 × 实体占比 × (0.5 + 0.5×收盘位置) × 量能放大系数
    再按振幅（相对 ATR）做一次缩放：同样一根阳线，在低波动股上是强信号，
    在天天 5% 振幅的票上就不算什么。

    返回 dict：
      energy    0~100，数值越大越"有能量"（不等于看多，方向看 label）
      signed    -100~100，正=多头能量，负=空头能量
      body_ratio 实体/振幅
      close_pos  收盘在当日区间的位置（0=最低，1=最高）
      label     强多 / 偏多 / 中性 / 偏空 / 强空 / 多空衰竭
    """
    o, h, l, c = _f(o), _f(h), _f(l), _f(c)
    if c <= 0 or h <= l:
        return {"energy": 0.0, "signed": 0.0, "body_ratio": 0.0, "close_pos": 0.5,
                "label": "无数据", "note": "K 线数据异常"}
    rng = max(h - l, 1e-9)
    body = abs(c - o)
    body_ratio = body / rng
    close_pos = (c - l) / rng
    upper = (h - max(o, c)) / rng
    lower = (min(o, c) - l) / rng
    direction = 1.0 if c > o else (-1.0 if c < o else 0.0)

    amp_bps = rng / c * 1e4
    atr_bps = max(20.0, _f(atr_norm, 0.02) * 1e4)
    amp_scale = max(0.35, min(1.6, amp_bps / atr_bps))

    vol_scale = 0.55 + 0.45 * math.tanh((_f(vol_ratio, 1.0) - 0.8) * 1.2)
    raw = direction * (0.35 + 0.65 * body_ratio) * (0.45 + 0.55 * abs(2 * close_pos - 1))
    signed = raw * vol_scale * amp_scale * 100
    energy = abs(signed)

    # 衰竭：长上影 + 收在低位（顶部）或长下影 + 收在高位（底部），且量能不小
    note = ""
    if upper > 0.55 and close_pos < 0.45 and _f(vol_ratio, 1) > 1.1:
        label, note = "多空衰竭", f"长上影 {upper*100:.0f}%，冲高回落，多头能量释放受阻"
        signed *= 0.45
    elif lower > 0.55 and close_pos > 0.55 and _f(vol_ratio, 1) > 1.1:
        label, note = "多空衰竭", f"长下影 {lower*100:.0f}%，下探回升，空头能量释放受阻"
        signed *= 0.45
    elif signed >= 45:
        label = "强多"
        note = f"实体 {body_ratio*100:.0f}%，收在区间 {close_pos*100:.0f}% 分位"
    elif signed >= 12:
        label = "偏多"
        note = f"阳线实体 {body_ratio*100:.0f}%，量能系数 {vol_scale:.2f}"
    elif signed <= -45:
        label = "强空"
        note = f"实体 {body_ratio*100:.0f}%，收在区间 {close_pos*100:.0f}% 分位"
    elif signed <= -12:
        label = "偏空"
        note = f"阴线实体 {body_ratio*100:.0f}%，量能系数 {vol_scale:.2f}"
    else:
        label = "中性"
        note = "小实体/十字星，方向能量不足"

    return {
        "energy": round(min(100.0, energy), 1),
        "signed": round(max(-100.0, min(100.0, signed)), 1),
        "body_ratio": round(body_ratio, 3),
        "close_pos": round(close_pos, 3),
        "upper_shadow": round(upper, 3),
        "lower_shadow": round(lower, 3),
        "amp_bps": round(amp_bps, 1),
        "vol_scale": round(vol_scale, 3),
        "label": label,
        "note": note,
    }


def energy_signal(energy: dict | None) -> tuple[float, str]:
    """把能量转成决策信号 ∈[-1,1]（带饱和，避免一根大阳线就顶满）。"""
    if not energy:
        return 0.0, "无能量数据"
    s = _f(energy.get("signed", 0.0)) / 100.0
    s = math.tanh(s * 1.6)
    return max(-1.0, min(1.0, s)), f"{energy.get('label', '')}(能量 {_f(energy.get('energy')):.0f}/100)"


def pressure_score(quote: dict | None = None, flow: dict | None = None,
                   ofi: float | None = None, ticks_n: int = 0,
                   ret: float | None = None, vol_ratio: float | None = None) -> dict:
    """买卖压力量化：0~1（0.5 中性），并给出每一项的贡献。

    权重刻意保持"每一项都不足以单独定性"：盘口可以挂假单、资金流是快照口径，
    任何单一指标都能骗人，只有多项同向时才有信息量。
    """
    items = []

    # 1) 五档委比（仅盘中有效）
    if quote and (quote.get("bid_vol") or quote.get("ask_vol")):
        wb = _f(quote.get("weibi"))
        s = math.tanh(wb / 40.0) * 0.32
        items.append(("五档委比", f"{wb:+.1f}%", s,
                      "买盘挂单占优" if wb > 8 else ("卖盘挂单占优" if wb < -8 else "挂单均衡")))
    else:
        items.append(("五档委比", "不可用", 0.0, "非交易时段或未返回盘口"))

    # 2) tick 主动买卖差（tick rule 近似）
    if ofi is not None and ticks_n >= 10:
        s = max(-1.0, min(1.0, ofi)) * 0.28
        items.append(("主动买卖差", f"{ofi:+.2f}", s,
                      "主动买盘主导" if ofi > 0.12 else ("主动卖盘主导" if ofi < -0.12 else "主动买卖均衡")))
    else:
        items.append(("主动买卖差", "样本不足", 0.0, "秒级快照不足，无法区分主动方向"))

    # 3) 主力资金净额（超大+大单，快照口径）
    if flow and flow.get("ok"):
        ratio = _f(flow.get("net_ratio"))
        s = math.tanh(ratio / 8.0) * 0.25
        items.append(("主力资金净额", f"{_f(flow.get('net_amount_wan')):+.0f} 万（{ratio:+.2f}%）", s,
                      "主力净流入" if ratio > 0 else "主力净流出"))
    else:
        items.append(("主力资金净额", "不可用", 0.0, "资金流接口未返回，该项不参与"))

    # 4) 量价配合
    if ret is not None and vol_ratio is not None:
        s = math.tanh(ret * 80) * math.tanh((vol_ratio - 1) * 1.5) * 0.15
        items.append(("量价配合", f"涨跌 {ret*100:+.2f}% / 量比 {vol_ratio:.2f}", s,
                      "放量上涨" if s > 0.02 else ("放量下跌" if s < -0.02 else "量价中性")))
    else:
        items.append(("量价配合", "不可用", 0.0, "缺少量比"))

    total = sum(x[2] for x in items)
    score = 0.5 + 0.5 * math.tanh(total)
    if score >= 0.62:
        label = "买盘主导"
    elif score <= 0.38:
        label = "卖盘主导"
    else:
        label = "买卖均衡"
    return {
        "score": round(score, 4),
        "signed": round((score - 0.5) * 2, 4),
        "label": label,
        "items": [{"name": n, "value": v, "contrib": round(c, 4), "note": note}
                  for n, v, c, note in items],
    }


def pressure_signal(pressure: dict | None) -> tuple[float, str]:
    if not pressure:
        return 0.0, "无买卖压力数据"
    s = _f(pressure.get("signed"))
    return max(-1.0, min(1.0, s)), f"{pressure.get('label', '')}({s:+.2f})"


def kline_meta(row: dict, vol_ratio: float = 1.0, atr_norm: float = 0.02) -> dict:
    """给日线入库/报告用的一行 K 线结构快照（真实开高低收 + 能量）。"""
    e = candle_energy(row.get("open"), row.get("high"), row.get("low"), row.get("close"),
                      vol_ratio=vol_ratio, atr_norm=atr_norm)
    return {
        "open": _f(row.get("open")), "high": _f(row.get("high")),
        "low": _f(row.get("low")), "close": _f(row.get("close")),
        "volume": _f(row.get("volume")), "vol_ratio": round(_f(vol_ratio, 1), 3),
        **e,
    }
