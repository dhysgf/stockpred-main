"""东方财富涨幅榜爬虫：A股 / 科创板 实时涨幅前 50。

接口
----
东财公开行情聚合接口（免 token）::

    https://push2.eastmoney.com/api/qt/clist/get
        ?pn=1&pz=50&po=1&np=1&fltt=2&invt=2&fid=f3&fs=<市场过滤>&fields=...

- ``fid=f3`` 按涨跌幅排序，``po=1`` 降序；
- A 股全市场 ``fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23``（深主板/创业板/沪主板/科创板）；
- 科创板 ``fs=m:1+t:23``。

克制原则
--------
1. 30 秒内存缓存：榜单页哪怕开很多个浏览器标签，也不会对东财高频施压；
2. 快照每 5 分钟才落库一次（save_rank_snapshot 内部有同分钟去重），
   用来回看"当时之前的涨幅榜"，而不是把每一次轮询都存下来；
3. 任何字段缺失都显示为 "-"，不编造数据。
"""
from __future__ import annotations

import json
import time
import urllib.request

# f124 = 该股行情最后更新时间（秒级时间戳）。收盘后它就是 15:00:0x，
# 用它而不是本地 now() 才能诚实回答"这份榜单到底有多新"。
_FIELDS = "f2,f3,f5,f6,f8,f10,f12,f14,f15,f16,f17,f18,f20,f21,f22,f62,f124,f184"

_MARKETS = {
    "a": ("A股（沪深全市场）", "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"),
    "kcb": ("科创板", "m:1+t:23"),
}

_CACHE: dict[str, tuple[float, dict]] = {}
_CACHE_TTL = 30.0            # 交易时段：30 秒
_CACHE_TTL_CLOSED = 180.0    # 收盘后数据是冻的，没必要每 30 秒空打一次接口
_LIMIT_UP = {"a": 9.9, "kcb": 19.9}      # 用于标注"接近涨停"（非严格口径，ST 另有 5%）


def _http_json(url: str, timeout: float = 6.0) -> dict:
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://quote.eastmoney.com/",
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "ignore"))


def _num(v, scale: float = 1.0):
    try:
        x = float(v)
        return None if x != x else round(x * scale, 3)
    except Exception:
        return None


def markets() -> list[dict]:
    return [{"id": k, "label": v[0]} for k, v in _MARKETS.items()]


def _is_open() -> bool:
    """复用实时层的交易时段判断；导入失败就保守认为开盘（宁可多抓）。"""
    try:
        from .realtime import is_open
        return bool(is_open())
    except Exception:
        return True


def _fmt_ts(epoch) -> str:
    try:
        x = float(epoch)
        if x <= 0:
            return ""
        return time.strftime("%H:%M:%S", time.localtime(x))
    except Exception:
        return ""


def fetch_rank(market: str = "a", top: int = 50, use_cache: bool = True,
               persist: bool = True) -> dict:
    """抓取涨幅榜前 N。返回 {ok, market, label, ts, total, rows:[...], note}。

    除本地抓取时间 ts 外，还给出行情时间 quote_ts（东财 f124，全榜最新）：
    收盘后榜单是冻结的，只有 quote_ts 能说明数据真实新鲜度。
    """
    key = f"{market}:{top}"
    open_now = _is_open()
    ttl = _CACHE_TTL if open_now else _CACHE_TTL_CLOSED
    if use_cache and key in _CACHE and time.time() - _CACHE[key][0] < ttl:
        out = dict(_CACHE[key][1])
        out["cached"] = True
        return out
    label, fs = _MARKETS.get(market, _MARKETS["a"])
    url = (f"https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz={min(max(top,1),100)}"
           f"&po=1&np=1&fltt=2&invt=2&fid=f3&fs={fs}&fields={_FIELDS}")
    try:
        j = _http_json(url)
    except Exception as e:
        return {"ok": False, "market": market, "label": label, "rows": [],
                "note": f"东财接口请求失败：{type(e).__name__} {e}", "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    d = j.get("data") or {}
    raw = d.get("diff") or []
    rows = []
    latest_epoch = 0.0
    limit_pct = _LIMIT_UP.get(market, 9.9)
    for i, it in enumerate(raw[:top]):
        code = str(it.get("f12") or "")
        pct = _num(it.get("f3"))
        q_epoch = _num(it.get("f124"))
        if q_epoch:
            latest_epoch = max(latest_epoch, float(q_epoch))
        rows.append({
            "rank": i + 1,
            "code": code,
            "name": str(it.get("f14") or ""),
            "price": _num(it.get("f2")),
            "pct": pct,
            "speed": _num(it.get("f22")),                    # 涨速（5 分钟）
            "volume_hand": _num(it.get("f5")),               # 成交量（手）
            "amount_yi": _num(it.get("f6"), 1e-8),           # 成交额（亿元）
            "turnover": _num(it.get("f8")),                  # 换手率 %
            "liangbi": _num(it.get("f10")),                  # 量比
            "high": _num(it.get("f15")),
            "low": _num(it.get("f16")),
            "open": _num(it.get("f17")),
            "prev_close": _num(it.get("f18")),
            "mktcap_yi": _num(it.get("f20"), 1e-8),          # 总市值（亿元）
            "floatcap_yi": _num(it.get("f21"), 1e-8),        # 流通市值（亿元）
            "main_net_wan": _num(it.get("f62"), 1e-4),       # 主力净流入（万元）
            "main_pct": _num(it.get("f184")),                # 主力净占比 %
            "near_limit": bool(pct is not None and pct >= limit_pct),
            "quote_ts": _fmt_ts(q_epoch),                    # 该股行情更新时间
            "quote_epoch": q_epoch,
        })
    rows_out = rows
    # 行情是否已经停更：最后一笔行情距今超过 90 秒，且当前不在交易时段。
    age = (time.time() - latest_epoch) if latest_epoch else None
    stale = bool((not open_now) and (age is None or age > 90))
    if stale:
        note = (f"已收盘：榜单价格已冻结（最新一笔行情 {_fmt_ts(latest_epoch) or '15:00:00'}，"
                f"个别票盘后仍有零星成交），下一交易日 09:30 恢复实时刷新")
    elif rows_out:
        note = ""
    else:
        note = "接口返回为空（可能非交易时段或接口变更）"
    out = {
        "ok": bool(rows_out),
        "market": market, "label": label,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "quote_ts": _fmt_ts(latest_epoch),
        "quote_age_sec": int(age) if age is not None else None,
        "market_open": open_now,
        "stale": stale,
        "total": _num(d.get("total")) or len(rows_out),
        "rows": rows_out,
        "note": note,
    }
    if use_cache and out["ok"]:
        _CACHE[key] = (time.time(), dict(out))
    if persist and out["ok"]:
        from . import db
        db.save_rank_snapshot(market, rows_out, out["total"])
    return out


def history(market: str = "a", limit: int = 48) -> list[dict]:
    from . import db
    return db.rank_history(market, limit)
