"""大模型实时预测：用户自选供应商 / 自填 Key / 限流调用。

三条红线
--------
1. **不喂自己的结论**：提示词里只有原始行情与指标，**不含**本系统的综合分、目标价、
   各专家预测。否则大模型会顺着我们的结论说，加权就变成"自己给自己投票"。
2. **幅度封顶 + 置信收缩**：LLM 经常给出"必涨 3%"这类离谱数字。这里把 5 分钟预测
   硬限制在 ±``LIVE.llm_max_bps``（默认 40bps = 0.4%），并按置信度线性收缩。
3. **限流 + 后台调用**：一次调用 3~20 秒，绝不能放在 3 秒一次的 tick 线程里。
   引擎用独立线程按 ``LIVE.llm_interval`` 调用，结果带时间戳参与加权（越旧权重越低）。

Key 存储：DB settings 表（本机明文，页面会明确警告）> 环境变量 > secrets.json。
接口只回显掩码，不回显明文。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time

import config
from . import db
from .llm import chat, _extract_json, _key_for, available_providers

_LOCK = threading.RLock()
_CACHE: dict[str, tuple[float, dict]] = {}

_KEYS = {
    "enabled": "live_llm_enabled",
    "provider": "live_llm_provider",
    "base_url": "live_llm_base",
    "model": "live_llm_model",
    "key": "live_llm_key",
    "interval": "live_llm_interval",
}

_ENV = {
    "qwen": "DASHSCOPE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "glm": "ZHIPU_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def presets() -> list[dict]:
    return list(getattr(config, "LLM_PRESETS", []))


def _preset(pid: str) -> dict:
    for p in presets():
        if p["id"] == pid:
            return p
    return {}


def mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}{'*' * 8}{key[-4:]}"


_SRC_CN = {"db": "本机数据库（上次保存的）", "env": "环境变量",
           "secrets": "secrets.json"}


def _src_cn(s: str) -> str:
    k = (s or "").split(":")[0]
    return _SRC_CN.get(k, s or "无")


def resolve() -> dict:
    """解析当前生效的 LLM 配置。返回 {ready, enabled, provider, base_url, model, key, source}。"""
    s = {k: db.get_setting(v) for k, v in _KEYS.items()}
    provider = (s.get("provider") or "qwen").strip()
    pre = _preset(provider)
    base = (s.get("base_url") or pre.get("base_url") or "").rstrip("/")
    model = (s.get("model") or pre.get("model") or "").strip()
    key = (s.get("key") or "").strip()
    source = "db" if key else ""
    if not key:
        env = _ENV.get(provider)
        if env and os.environ.get(env):
            key, source = os.environ[env].strip(), f"env:{env}"
        elif provider in ("qwen", "deepseek", "ollama"):
            k = _key_for(provider)
            if k:
                key, source = k, "secrets"
    enabled = str(s.get("enabled") if s.get("enabled") is not None else "1").lower() not in ("0", "false", "off", "")
    ready = bool(base and model and (key or provider == "ollama"))
    try:
        interval = float(s.get("interval") or (getattr(config, "LIVE", {}) or {}).get("llm_interval", 300))
    except Exception:
        interval = 300.0
    return {
        "enabled": enabled, "ready": ready, "provider": provider,
        "base_url": base, "model": model, "key": key, "key_source": source,
        "key_masked": mask_key(key), "interval": max(30.0, interval),
        "preset": pre.get("label", provider),
    }


def public_config() -> dict:
    """给页面用的配置（绝不返回明文 Key）。"""
    r = resolve()
    return {
        "enabled": r["enabled"], "ready": r["ready"], "provider": r["provider"],
        "base_url": r["base_url"], "model": r["model"], "key_masked": r["key_masked"],
        "key_source": _src_cn(r["key_source"]), "interval": r["interval"],
        "preset": r["preset"], "presets": presets(),
        "env_providers": available_providers(),
    }


def save_config(payload: dict) -> dict:
    """保存用户选择的供应商/模型/Key。key 传空串表示"保持不变"，传 "-" 表示清除。"""
    with _LOCK:
        if payload.get("provider"):
            pid = str(payload["provider"]).strip()
            db.set_setting(_KEYS["provider"], pid)
            pre = _preset(pid)
            # 切换供应商时，若用户没同时给地址/模型，则用该预设的默认值覆盖
            if not payload.get("base_url") and pre.get("base_url"):
                db.set_setting(_KEYS["base_url"], pre["base_url"])
            if not payload.get("model") and pre.get("model"):
                db.set_setting(_KEYS["model"], pre["model"])
        if "base_url" in payload and payload["base_url"] is not None:
            db.set_setting(_KEYS["base_url"], str(payload["base_url"]).strip())
        if "model" in payload and payload["model"] is not None:
            db.set_setting(_KEYS["model"], str(payload["model"]).strip())
        if "enabled" in payload:
            db.set_setting(_KEYS["enabled"], "1" if payload["enabled"] else "0")
        if "interval" in payload:
            try:
                db.set_setting(_KEYS["interval"], str(max(30.0, float(payload["interval"]))))
            except Exception:
                pass
        k = payload.get("api_key")
        if isinstance(k, str) and k.strip():
            if k.strip() == "-":
                db.set_setting(_KEYS["key"], "")
            elif "*" not in k:          # 页面回显的掩码值不要存回去
                db.set_setting(_KEYS["key"], k.strip())
        _CACHE.clear()
    return public_config()


def test_connection(payload: dict | None = None) -> dict:
    """连通性测试。

    payload 来自页面输入框 {base_url, model, api_key}，**只用于这次测试、不落库**。

    必须回显「这次到底用了谁的 Key」：输入框是 password 型且不回填，
    用户看到的是空白，但实际可能用的是数据库里上次保存的 Key ——
    不写清楚就成了"我没填 Key 却测通了"，等于骗人。
    """
    p = payload or {}
    r = resolve()
    inp_key = str(p.get("api_key") or "").strip()
    if "*" in inp_key:                 # 页面回显的掩码不是真 Key
        inp_key = ""
    base = str(p.get("base_url") or "").strip().rstrip("/") or r["base_url"]
    model = str(p.get("model") or "").strip() or r["model"]
    if inp_key:
        key, source = inp_key, "输入框（本次新填，未保存）"
    elif r["key"]:
        src = (r["key_source"] or "").split(":")[0]
        key = r["key"]
        source = {"db": "本机数据库里已保存的 Key（不是这次填的）",
                  "env": "环境变量里的 Key",
                  "secrets": "secrets.json 里的 Key"}.get(src, "已保存的 Key")
    else:
        key, source = "", "无"
    info = {"key_source": source, "key_masked": mask_key(key), "base_url": base,
            "model": model, "provider": r["provider"]}
    if not key and r["provider"] != "ollama":
        return {**info, "ok": False,
                "note": "没有可用的 Key：输入框为空，本机也没保存过 Key"}
    if not base or not model:
        return {**info, "ok": False, "note": "缺少 Base URL 或模型名"}
    ov = {"api_key": key, "base_url": base, "model": model}
    t0 = time.time()
    res = chat("只回复两个字符：OK", system="你是连通性测试端点。", max_tokens=8,
               override=ov, timeout=20)
    dt = round(time.time() - t0, 2)
    ok = bool(res.get("ok"))
    db.log_llm(r["provider"], model, "test", ok, time.time() - t0,
               "" if ok else str(res.get("error"))[:200])
    return {**info, "ok": ok, "elapsed": dt,
            "text": (res.get("text") or "")[:60],
            "note": "" if ok else f"调用失败：{res.get('error', '')[:200]}"}


# =====================================================================
# 实时预测提示词
# =====================================================================
_SYS = (
    "你是 A 股日内交易员，只做 1~5 分钟级别的短周期判断。"
    "你只能看到原始行情与指标，看不到任何其他模型的结论。"
    "必须承认短周期预测的不确定性：没有把握就给 0 附近的预测与低置信度。"
    "严格输出 JSON，不要 markdown 代码块，不要解释文字。"
)

_PROMPT = """标的：{name}（{code}）
当前时间：{ts}（交易日内第 {minute} 分钟，剩余 {remain} 分钟）

【最新盘面】
- 现价 {price:.2f}，涨跌 {pct:+.2f}%，当日区间 {low:.2f}~{high:.2f}
- 今日 VWAP {vwap:.2f}，现价相对 VWAP {vwap_dev:+.1f} bps
- 近 1/3/5/10 分钟收益：{ret1} / {ret3} / {ret5} / {ret10} bps
- 1 分钟波动（近 30 根）：{sigma:.2f} bps
- 量能（近 5 分钟 / 近 30 分钟）：{vol_ratio}
- 五档委比：{weibi}%（正=买盘挂单占优）
- 主动买卖差（近 60 秒 tick）：{ofi}
- 日内区间位置：{range_pos}（0=最低，1=最高）

【最近 20 根 1 分钟 K 线】（时间 开/高/低/收 量(手)）
{bars_block}

请判断**未来 5 分钟**（{h} 分钟）的价格变化，并给出今日剩余时段的极值预期。输出 JSON：
{{
  "bps": -40 到 40 的小数,          // 未来 5 分钟相对现价的涨跌，单位 bps（0.01%）
  "confidence": 0 到 1 的小数,      // 把握程度；没把握就给 0.1~0.3
  "direction": "偏多" 或 "偏空" 或 "中性",
  "day_high": 今日剩余时段最高价（元，若认为已出现就给当前最高）,
  "day_low": 今日剩余时段最低价,
  "next_turn": {{"kind": "顶" 或 "底", "in_minutes": 1 到 60 的整数, "price": 元}},
  "reason": "不超过 40 字的核心依据",
  "risk": "不超过 30 字：什么情况会让这个判断失效"
}}
硬性要求：
1. bps 的绝对值通常应小于 2 倍 1 分钟波动 ×√5；极端行情才给到 30 以上。
2. 若盘面是横盘/无量，direction 给"中性"、bps 给 ±5 以内、confidence ≤0.3。
3. 只输出 JSON。"""


def _fmt_bars(bars: list[dict], n: int = 20) -> str:
    lines = []
    for b in bars[-n:]:
        try:
            v = float(b.get("volume") or 0) / 100.0    # 股 → 手
            lines.append(f"  {str(b['ts'])[11:16]} {float(b['open']):.2f}/{float(b['high']):.2f}/"
                         f"{float(b['low']):.2f}/{float(b['close']):.2f} {v:.0f}")
        except Exception:
            continue
    return "\n".join(lines) if lines else "  （无）"


def _cache_key(sym: str, bars: list[dict], price: float) -> str:
    tail = str(bars[-1]["ts"]) if bars else ""
    h = hashlib.sha1(f"{sym}|{tail}|{price:.2f}".encode()).hexdigest()[:16]
    return h


def ask_intraday(symbol: str, name: str, features: dict, bars: list[dict],
                 session: dict | None = None, force: bool = False) -> dict:
    """异步链路里调用：返回结构化结果（失败也要返回可判定的 dict）。"""
    r = resolve()
    if not r["enabled"]:
        return {"ok": False, "note": "未启用大模型", "provider": r["provider"]}
    if not r["ready"]:
        return {"ok": False, "note": "未配置完整的地址/模型/Key", "provider": r["provider"]}
    if not features.get("valid") or len(bars) < 6:
        return {"ok": False, "note": "行情样本不足", "provider": r["provider"]}

    ck = _cache_key(symbol, bars, features.get("price") or 0)
    with _LOCK:
        hit = _CACHE.get(ck)
        if hit and not force and time.time() - hit[0] < 120:
            return {**hit[1], "cached": True}

    series = getattr(config, "LIVE", {}) or {}
    h = int(series.get("primary_horizon", 5))
    minute = int(features.get("minute") or 0)
    remain = max(0, 240 - minute)
    p = features.get("price") or 0
    prompt = _PROMPT.format(
        name=name or symbol, code=symbol, ts=(session or {}).get("now", ""),
        minute=minute, remain=remain, price=p,
        pct=(features.get("day_ret_bps") or 0) / 100.0,
        low=features.get("day_low") or p, high=features.get("day_high") or p,
        vwap=features.get("vwap") or p, vwap_dev=features.get("vwap_dev") or 0,
        ret1=_n(features.get("ret1")), ret3=_n(features.get("ret3")),
        ret5=_n(features.get("ret5")), ret10=_n(features.get("ret10")),
        sigma=float(features.get("sigma_min") or 0.0), vol_ratio=_n(features.get("vol_ratio")),
        weibi=_n(features.get("weibi")), ofi=_n(features.get("ofi")),
        range_pos=_n(features.get("range_pos")),
        bars_block=_fmt_bars(bars), h=h,
    )
    t0 = time.time()
    # 实测：qwen3.7-plus 这类带思考的模型，输出 400 token 的 JSON 要 25~35 秒，
    # 卡 25 秒会大面积超时（超时后还得等 30 秒重试，等于白等）。给到 45 秒留余量；
    # 反正在独立线程里跑，不阻塞 3 秒一次的 tick。
    res = chat(prompt, system=_SYS, max_tokens=420,
               override={"api_key": r["key"], "base_url": r["base_url"], "model": r["model"]},
               timeout=45, temperature=0.1)
    dt = time.time() - t0
    if not res.get("ok"):
        db.log_llm(r["provider"], r["model"], "intraday", False, dt, str(res.get("error"))[:200])
        return {"ok": False, "note": f"调用失败：{str(res.get('error'))[:120]}",
                "provider": r["provider"], "model": r["model"], "elapsed": round(dt, 2)}
    obj = _extract_json(res.get("text") or "")
    if not obj:
        db.log_llm(r["provider"], r["model"], "intraday", False, dt, "JSON 解析失败")
        return {"ok": False, "note": "返回无法解析为 JSON", "provider": r["provider"],
                "model": r["model"], "raw": (res.get("text") or "")[:200]}

    cap = float(series.get("llm_max_bps", 40.0))
    try:
        bps = float(obj.get("bps") or 0)
    except Exception:
        bps = 0.0
    if bps != bps:
        bps = 0.0
    bps = max(-cap, min(cap, bps))
    try:
        conf = float(obj.get("confidence") or 0)
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf if conf == conf else 0.0))

    def _num(key, default=None):
        try:
            v = float(obj.get(key))
            return v if v > 0 else default
        except Exception:
            return default

    day_high = _num("day_high")
    day_low = _num("day_low")
    if day_high and p:
        day_high = min(p * 1.08, max(p * 0.95, day_high))
    if day_low and p:
        day_low = max(p * 0.92, min(p * 1.05, day_low))
    turn = obj.get("next_turn") or {}
    kind = str(turn.get("kind") or "")
    out = {
        "ok": True, "provider": r["provider"], "model": r["model"],
        "bps": round(bps, 2), "confidence": round(conf, 3),
        "direction": "偏多" if bps > 3 else ("偏空" if bps < -3 else "中性"),
        "day_high": round(day_high, 2) if day_high else None,
        "day_low": round(day_low, 2) if day_low else None,
        "next_turn": {
            "kind": kind if kind in ("顶", "底") else "",
            "in_minutes": max(1, min(60, int(float(turn.get("in_minutes") or 0) or 0))) if turn else 0,
            "price": round(float(turn.get("price") or 0), 2) if turn else None,
        } if turn else None,
        "reason": str(obj.get("reason") or "")[:80],
        "risk": str(obj.get("risk") or "")[:60],
        "elapsed": round(dt, 2), "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "",
    }
    with _LOCK:
        _CACHE[ck] = (time.time(), out)
        if len(_CACHE) > 60:
            for k in list(_CACHE)[:-40]:
                _CACHE.pop(k, None)
    db.log_llm(r["provider"], r["model"], "intraday", True, dt, out["reason"])
    return out


def _n(x) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "-"
