"""实时新闻 / 政策 / 舆论量化 —— 给分钟级预测提供第 9 路（慢变量）证据。

为什么单独一层，而不是直接复用日线的 ``analyze_news_llm``
------------------------------------------------------------
日线那套是"每次分析跑一次"，可以等十几秒。实时链路 3 秒一轮，不能这么干。
所以这里加了三件事：

1. **低频刷新**（默认 180 秒）：新闻抓取要 3~10 秒、大模型要花钱，
   跟着 3 秒轮询既跑不动也没必要；
2. **时间衰减**：新闻是慢变量，30 分钟前的利好不该和刚出的利好同权，
   用指数衰减 ``exp(-age/half_life)`` 折掉；
3. **幅度封顶**：这一路只允许给出很窄的方向偏置（见 experts.expert_news），
   绝不假装它能预测 5 分钟的 bps。

诚实的局限（写进返回值，也写进页面）
--------------------------------------
没有历史新闻序列 → 这一路**无法回测**，权重只能给很小先验，再让 Hedge
按实时结算误差慢慢降。默认无大模型 Key 时走本地词典情绪，颗粒度更粗，
``has_llm=False`` 会被明确标出来，不冒充大模型结论。
"""
from __future__ import annotations

import math
import re
import time

import config

from .news_fetcher import collect_news

_CACHE: dict[str, dict] = {}
_INFLIGHT: set[str] = set()
# 大模型端点连续失败时的退避截止时间：避免每 180 秒都去撞一次超时
_LLM_BACKOFF_UNTIL: float = 0.0
_LLM_FAIL_STREAK: int = 0


def _cfg(key: str, default):
    return (getattr(config, "LIVE", {}) or {}).get(key, default)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _llm_override() -> dict | None:
    """取网页里用户自己填的 Key；没配就返回 None（走本地情绪降级）。"""
    try:
        from . import llm_live
        r = llm_live.resolve()
        if r.get("ready"):
            return {"api_key": r["key"], "base_url": r["base_url"], "model": r["model"]}
    except Exception:
        pass
    return None


def _empty(note: str = "") -> dict:
    return {
        "ok": False, "score": 0.0, "senti": 0.0, "policy": 0.0, "conf": 0.0,
        "stance": "中性", "news_count": 0, "headlines": [], "risks": [],
        "provider": "", "model": "", "has_llm": False,
        "ts": 0.0, "age_sec": 0.0, "decay": 0.0, "decayed_score": 0.0,
        "half_life_sec": float(_cfg("news_half_life", 1800)),
        "note": note, "summary": "",
    }


def _compose(res: dict, half_life: float) -> dict:
    """把 ``analyze_news_llm`` 的原始结论压成一路可用信号。"""
    senti = (_clamp(float(res.get("sentiment", 0.5)), 0.0, 1.0) - 0.5) * 2.0
    policy = _clamp(float(res.get("policy_impact", 0.0)), -1.0, 1.0)
    # 个股情绪为主、政策为辅：政策对单只票的传导路径更长更不确定
    score = _clamp(0.65 * senti + 0.35 * policy, -1.0, 1.0)
    provider = str(res.get("provider") or "")
    has_llm = bool(res.get("ok")) and provider not in ("fallback", "off", "")
    conf = _clamp(float(res.get("confidence", 0.2)), 0.0, 1.0)
    if not has_llm:
        conf = min(conf, 0.25)      # 词典情绪：只值这么多，不许它抢戏
    ts = float(res.get("_ts") or time.time())
    age = max(0.0, time.time() - ts)
    decay = math.exp(-age / max(60.0, half_life))
    return {
        "ok": True,
        "score": round(score, 4),
        "senti": round(senti, 4),
        "policy": round(policy, 4),
        "conf": round(conf, 3),
        "stance": str(res.get("stance") or ("偏多" if score > 0.08 else "偏空" if score < -0.08 else "中性")),
        "news_count": int(res.get("news_count") or 0),
        "headlines": list(res.get("key_drivers") or [])[:3],
        "risks": list(res.get("risks") or [])[:2],
        "provider": provider,
        "model": str(res.get("model") or ""),
        "has_llm": has_llm,
        "ts": ts,
        "age_sec": round(age, 1),
        "decay": round(decay, 4),
        "decayed_score": round(score * decay, 4),
        "half_life_sec": float(half_life),
        "note": str(res.get("note") or ""),
        "summary": str(res.get("summary") or "")[:200],
    }


def _local_face(groups: dict, reason: str) -> dict:
    """不走大模型的本地情绪路径（无 Key、或端点连续失败时用）。

    自己组装而不是调 ``analyze_news_llm``，是为了在端点挂掉时**不产生任何网络等待**。
    """
    from .sentiment import analyze_news

    flat = []
    for kind in ("stock", "policy", "market"):
        flat.extend(groups.get(kind) or [])
    s = analyze_news(flat)
    stance = {"偏看涨": "偏多", "偏看跌": "偏空"}.get(s["label"], "中性")
    return {
        "ok": False, "provider": "fallback", "model": "snownlp/词典",
        "sentiment": s["mean"], "policy_impact": 0.0, "confidence": 0.2,
        "stance": stance,
        "key_drivers": [f"本地情绪分 {s['mean']:.3f}（{s['count']} 条材料）"],
        "risks": ["未接入大模型，无法做事件级/政策级归因"],
        "summary": "仅由本地词典情绪得到，颗粒度较粗。",
        "news_count": len(flat), "note": reason, "_ts": time.time(),
    }


def _fetch_face(symbol: str, name: str, price: float, ret5: float, ret20: float,
                vol_expand: float, net_ratio: float) -> dict:
    global _LLM_BACKOFF_UNTIL, _LLM_FAIL_STREAK
    now = time.time()
    # 东财搜索按关键词匹配：名称缺失时别把 "sh600519" 整个丢过去，退化成 6 位代码
    kw = name.strip() if name else re.sub(r"^(sh|sz|bj)", "", str(symbol).lower())
    groups = collect_news(kw, name=kw,
                          stock_num=int(_cfg("news_stock_num", 12)),
                          market_num=int(_cfg("news_market_num", 12)),
                          policy_num=int(_cfg("news_policy_num", 8)))
    total = sum(len(groups.get(k) or []) for k in ("stock", "policy", "market"))
    if total == 0:
        return _empty("没抓到新闻（接口可能限流，或该股过于冷门）")

    if now < _LLM_BACKOFF_UNTIL:
        left = int((_LLM_BACKOFF_UNTIL - now) / 60) + 1
        res = _local_face(groups, f"大模型端点刚失败过，{left} 分钟内先用本地情绪")
    else:
        from .llm import analyze_news_llm
        res = analyze_news_llm(groups, code=symbol, name=name, price=price,
                               ret5=ret5, ret20=ret20, vol_expand=vol_expand,
                               net_ratio=net_ratio, override=_llm_override())
        failed = (str(res.get("provider") or "") == "fallback"
                  and "失败" in str(res.get("note") or ""))
        if failed:
            _LLM_FAIL_STREAK += 1
            # 失败一次就退避 10 分钟：新闻面本来就是低频量，等得起
            if _LLM_FAIL_STREAK >= 1:
                _LLM_BACKOFF_UNTIL = time.time() + 600.0
        elif res.get("ok"):
            _LLM_FAIL_STREAK = 0
            _LLM_BACKOFF_UNTIL = 0.0
    out = _compose(res, float(_cfg("news_half_life", 1800)))
    out["_fetch_ts"] = now
    return out


def face(symbol: str, name: str = "", price: float = 0.0, ret5: float = 0.0,
         ret20: float = 0.0, vol_expand: float = 1.0, net_ratio: float = 0.0,
         force: bool = False, use_cache: bool = True) -> dict:
    """取该股当前的新闻/政策面。带缓存，命中最快秒回；未命中会真抓（几秒）。

    调用方（引擎）应放在后台线程里，别卡住 3 秒的行情轮询。
    """
    key = symbol
    ttl = float(_cfg("news_interval", 180))
    now = time.time()

    if use_cache and not force and key in _CACHE:
        old = _CACHE[key]
        if now - float(old.get("_fetch_ts") or 0) < ttl:
            out = dict(old)
            # 缓存命中也要重算"现在有多旧"和衰减，否则页面上的新鲜度会停住
            age = max(0.0, now - float(out.get("ts") or now))
            hl = float(out.get("half_life_sec") or 1800)
            out["age_sec"] = round(age, 1)
            out["decay"] = round(math.exp(-age / max(60.0, hl)), 4)
            out["decayed_score"] = round(float(out.get("score") or 0) * out["decay"], 4)
            out["cached"] = True
            return out

    try:
        out = _fetch_face(symbol, name, price, ret5, ret20, vol_expand, net_ratio)
        _CACHE[key] = out
        return out
    except Exception as e:
        out = _empty(f"新闻层异常：{type(e).__name__} {e}")
        out["_fetch_ts"] = now
        _CACHE[key] = out
        return out


def refresh_async(symbol: str, name: str = "", price: float = 0.0,
                  ret5: float = 0.0, ret20: float = 0.0,
                  vol_expand: float = 1.0, net_ratio: float = 0.0) -> dict | None:
    """非阻塞刷新。

    - 缓存还新鲜 → 直接返回（不新建线程）；
    - 缓存过期且没有正在跑的刷新 → 起一个线程去抓，本次返回旧结果（或 None）。
    这样引擎每 3 秒调用也永远不会被网络拖住。
    """
    import threading

    key = symbol
    ttl = float(_cfg("news_interval", 180))
    now = time.time()
    cur = _CACHE.get(key)
    fresh = cur and (now - float(cur.get("_fetch_ts") or 0) < ttl)
    if fresh:
        age = max(0.0, now - float(cur.get("ts") or now))
        hl = float(cur.get("half_life_sec") or 1800)
        out = dict(cur)
        out["age_sec"] = round(age, 1)
        out["decay"] = round(math.exp(-age / max(60.0, hl)), 4)
        out["decayed_score"] = round(float(out.get("score") or 0) * out["decay"], 4)
        return out
    if key in _INFLIGHT:
        return cur
    _INFLIGHT.add(key)

    def _work():
        try:
            face(symbol, name=name, price=price, ret5=ret5, ret20=ret20,
                 vol_expand=vol_expand, net_ratio=net_ratio, force=True, use_cache=False)
        finally:
            _INFLIGHT.discard(key)

    threading.Thread(target=_work, name=f"news-{symbol}", daemon=True).start()
    return cur


def status() -> dict:
    return {"cached": len(_CACHE), "inflight": len(_INFLIGHT),
            "interval_sec": float(_cfg("news_interval", 180)),
            "half_life_sec": float(_cfg("news_half_life", 1800)),
            "llm_backoff_sec": round(max(0.0, _LLM_BACKOFF_UNTIL - time.time()), 1),
            "llm_fail_streak": _LLM_FAIL_STREAK}
