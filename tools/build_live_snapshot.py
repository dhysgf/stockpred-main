"""把实时引擎的真实运行数据渲染成静态快照页 docs/live.html。

为什么需要这个：
    GitHub Pages 只能托管静态文件，跑不了 `python live.py`（每 3 秒拉行情、
    Hedge 在线调权、写 SQLite）。所以「把新服务放到线上」唯一可行的形态是
    **真实运行状态的固化快照** —— 数据全部来自 data/stockpred.db，不是编的。

用法：
    python tools/build_live_snapshot.py            # 用默认标的 sh600519
    python tools/build_live_snapshot.py sh600519   # 指定标的

输出：
    docs/live.html（纯静态、零外部依赖、内联 SVG 图表）
"""
from __future__ import annotations

import html
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "stockpred.db"
OUT = ROOT / "docs" / "live.html"

# 脚本位于 tools/ 下，直接运行 sys.path[0] 是 tools/，需补上项目根才能 import stockpred
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 涨红跌绿（A 股习惯）
C_UP = "#c0392b"
C_DOWN = "#1e7f43"
C_NEU = "#6b7280"


def esc(s) -> str:
    return html.escape(str(s or ""))


def cls_of(v: float) -> str:
    return "up" if v > 0 else ("down" if v < 0 else "neu")


def fmt_bps(v) -> str:
    if v is None:
        return "—"
    return f"{v:+.2f}"


# --------------------------------------------------------------------------- 取数


def load(symbol: str) -> dict:
    c = sqlite3.connect(DB, timeout=15)
    c.row_factory = sqlite3.Row
    d: dict = {"symbol": symbol}

    # 最新一条带专家明细的预测
    row = c.execute(
        "select * from live_predictions where symbol=? and experts_json!='[]' "
        "order by id desc limit 1",
        (symbol,),
    ).fetchone()
    d["latest"] = dict(row) if row else None
    d["experts"] = json.loads(row["experts_json"]) if row else []

    # 今日结算统计
    d["stats"] = {}
    for r in c.execute(
        """select horizon_min, count(*) n, sum(realized_bps is not null) settled,
                  avg(case when realized_bps is not null then abs(err_bps) end) mae,
                  avg(case when realized_bps is not null and err_bps>0 then 1.0
                           when realized_bps is not null then 0.0 end) hit,
                  avg(case when realized_bps is not null
                           and abs(err_bps) <= (hi-lo)/2.0/price*10000
                           then 1.0 else 0.0 end) cover
           from live_predictions
           where symbol=? and substr(ts,1,10)=date('now','localtime')
           group by horizon_min order by horizon_min""",
        (symbol,),
    ):
        d["stats"][r["horizon_min"]] = dict(r)

    # 今日 1 分钟预测序列（画 预测 vs 真实）
    d["series"] = [
        dict(r)
        for r in c.execute(
            """select ts, price, pred_price, realized_price, pred_bps, realized_bps, err_bps
               from live_predictions
               where symbol=? and horizon_min=1 and substr(ts,1,10)=date('now','localtime')
               order by ts""",
            (symbol,),
        )
    ]

    # 涨幅榜最新快照
    d["rank"] = {}
    for mk in ("a", "kcb"):
        r = c.execute(
            "select ts,total,rows_json from rank_snapshots where market=? "
            "order by id desc limit 1",
            (mk,),
        ).fetchone()
        if r:
            d["rank"][mk] = {
                "ts": r["ts"],
                "total": r["total"],
                "rows": json.loads(r["rows_json"]),
            }

    # 被跟踪过的标的
    d["tracked"] = [
        dict(r)
        for r in c.execute(
            """select symbol, count(*) n, min(ts) first_ts, max(ts) last_ts,
                      sum(mode='live') live_n, sum(mode='replay') replay_n
               from live_predictions group by symbol order by n desc"""
        )
    ]
    c.close()

    # 可选：联网取一次行情，仅用于「名称 / 涨跌幅 / 当日高低」。
    # 快照价仍以数据库里那一秒的真实成交价为准，不混用。
    d["quote"] = {}
    try:
        from stockpred import realtime

        q = realtime.fetch_quote(symbol)
        d["quote"] = {
            "name": getattr(q, "name", "") or "",
            "pct": getattr(q, "pct", None),
            "high": getattr(q, "high", None),
            "low": getattr(q, "low", None),
            "prev_close": getattr(q, "prev_close", None),
            "ts": getattr(q, "ts", ""),
        }
    except Exception as e:  # 离线也能生成，只是少几项
        d["quote"] = {"_err": f"{type(e).__name__}"}
    return d


# --------------------------------------------------------------------- 组件渲染


def kpi_html(d: dict) -> str:
    lt = d["latest"] or {}
    st1 = d["stats"].get(1) or {}
    st5 = d["stats"].get(5) or {}
    q = d.get("quote") or {}
    price = lt.get("price") or 0
    pct = q.get("pct")
    es = d["experts"]
    on_n = len([x for x in es if x.get("available")])

    pct_txt = f"{pct:+.2f}%" if isinstance(pct, (int, float)) else "—"
    pct_color = C_NEU if not isinstance(pct, (int, float)) else (
        C_UP if pct > 0 else (C_DOWN if pct < 0 else C_NEU)
    )
    pred = lt.get("pred_bps")
    items = [
        ("快照价", f"{price:.2f}", "预测发出时的真实成交价", C_NEU),
        ("当日涨跌", pct_txt,
         f"行情 {q.get('ts', '')}" if q.get("ts") else "行情未取到", pct_color),
        ("5 分钟预测", f"{fmt_bps(pred)} bps", str(lt.get("direction") or ""),
         C_UP if (pred or 0) > 0 else (C_DOWN if (pred or 0) < 0 else C_NEU)),
        ("置信度", f"{(lt.get('conf') or 0) * 100:.0f}%",
         f"波动 σ {(lt.get('sigma_bps') or 0):.1f} bps", C_NEU),
        ("已结算 1/3/5",
         "/".join(str((d["stats"].get(h) or {}).get("settled") or 0) for h in (1, 3, 5)),
         "今日累计", C_NEU),
        ("MAE（1 分钟）", f"{(st1.get('mae') or 0):.2f} bps", "平均绝对误差", C_NEU),
        ("区间覆盖（1 分钟）", f"{(st1.get('cover') or 0) * 100:.0f}%",
         "目标 80%", C_NEU),
        ("参与加权", f"{on_n}/{len(es)} 路", "其余弃权不计权重", C_NEU),
    ]
    out = []
    for k, v, sub, color in items:
        sub_html = f'<span class="sub">{esc(sub)}</span>' if sub else ""
        out.append(
            f'<div class="kpi"><span class="k">{esc(k)}</span>'
            f'<b style="color:{color}">{esc(v)}</b>{sub_html}</div>'
        )
    return "\n".join(out)


def experts_html(d: dict) -> str:
    es = d["experts"]
    if not es:
        return '<p class="muted">该标的还没有专家明细记录。</p>'
    wmax = max((x.get("eff_weight") or 0) for x in es) or 1
    rows = []
    for x in es:
        avail = bool(x.get("available"))
        ew = x.get("eff_weight") or 0
        bar_w = int(ew / wmax * 100)
        contrib = x.get("contrib") or 0
        if avail:
            state = '<span class="pill ok">参与</span>'
        else:
            state = '<span class="pill off">弃权</span>'
        pred = f"{x.get('pred_bps', 0):+.2f}" if avail else "—"
        rows.append(
            "<tr class='{dim}'>"
            "<td><b>{cn}</b><span class='code'>{name}</span></td>"
            "<td>{state}</td>"
            "<td class='num {pc}'>{pred}</td>"
            "<td class='num'><div class='wbar'><i style='width:{bw}%'></i></div>"
            "<span class='tiny'>{ew:.1%}</span></td>"
            "<td class='num {cc}'>{contrib:+.2f}</td>"
            "<td class='num'>{n}</td>"
            "<td class='num'>{mae}</td>"
            "<td class='num'>{hit}</td>"
            "<td class='note'>{note}</td>"
            "</tr>".format(
                dim="" if avail else "offrow",
                cn=esc(x.get("cn")),
                name=esc(x.get("name")),
                state=state,
                pc=cls_of(x.get("pred_bps") or 0) if avail else "neu",
                pred=pred,
                bw=bar_w,
                ew=ew,
                cc=cls_of(contrib),
                contrib=contrib,
                n=x.get("n") or 0,
                mae=f"{x.get('mae_bps', 0):.1f}" if (x.get("n") or 0) else "—",
                hit=f"{(x.get('hit') or 0) * 100:.0f}%" if (x.get("n") or 0) else "—",
                note=esc(x.get("note")),
            )
        )
    return (
        "<table class='ex'><thead><tr><th>专家</th><th>状态</th><th>当前预测</th>"
        "<th>有效权重</th><th>贡献</th><th>样本</th><th>MAE</th><th>命中</th>"
        "<th>依据 / 说明</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def svg_chart(d: dict, w: int = 880, h: int = 260) -> str:
    """预测价 vs 真实结算价（1 分钟周期）。纯内联 SVG，无外部依赖。"""
    s = [r for r in d["series"] if r.get("price") and r.get("pred_price")]
    if len(s) < 5:
        return '<p class="muted">今日样本不足，暂不绘图。</p>'
    # 降采样到 ~160 点
    step = max(1, len(s) // 160)
    s = s[::step]
    real = [r for r in s if r.get("realized_price")]
    ys = [r["price"] for r in s] + [r["pred_price"] for r in s]
    if real:
        ys += [r["realized_price"] for r in real]
    lo, hi = min(ys), max(ys)
    pad = (hi - lo) * 0.12 or 1.0
    lo, hi = lo - pad, hi + pad
    n = len(s)
    ml, mr, mt, mb = 52, 12, 12, 26
    iw, ih = w - ml - mr, h - mt - mb

    def X(i):
        return ml + iw * (i / max(1, n - 1))

    def Y(v):
        return mt + ih * (1 - (v - lo) / (hi - lo))

    pred_pts = " ".join(f"{X(i):.1f},{Y(r['pred_price']):.1f}" for i, r in enumerate(s))
    real_pts = " ".join(
        f"{X(s.index(r)):.1f},{Y(r['realized_price']):.1f}" for r in real
    ) if real else ""
    # 网格
    grid = []
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        y = Y(v)
        grid.append(
            f"<line x1='{ml}' y1='{y:.1f}' x2='{w-mr}' y2='{y:.1f}' "
            f"stroke='#eef1f4' stroke-width='1'/>"
            f"<text x='{ml-6}' y='{y+4:.1f}' text-anchor='end' "
            f"font-size='10' fill='#8a94a2'>{v:.1f}</text>"
        )
    # x 轴时间标签
    xt = []
    for i in (0, n // 3, 2 * n // 3, n - 1):
        t = str(s[i]["ts"])[11:16]
        xt.append(
            f"<text x='{X(i):.1f}' y='{h-8}' text-anchor='middle' "
            f"font-size='10' fill='#8a94a2'>{t}</text>"
        )
    last = s[-1]
    return f"""<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px">
{''.join(grid)}
<polyline points="{real_pts}" fill="none" stroke="{C_NEU}" stroke-width="1.6"
  stroke-linejoin="round"/>
<polyline points="{pred_pts}" fill="none" stroke="#3b82f6" stroke-width="1.4"
  stroke-dasharray="4 3" stroke-linejoin="round"/>
{''.join(xt)}
<circle cx="{X(n-1):.1f}" cy="{Y(last['price']):.1f}" r="3" fill="{C_NEU}"/>
</svg>
<div class="legend">
  <span><i style="background:#3b82f6"></i>预测价（提前 1 分钟给出）</span>
  <span><i style="background:{C_NEU}"></i>到点真实价</span>
  <span class="tiny">两线贴合程度即 1 分钟预测的实际误差；最新价 {last['price']:.2f}</span>
</div>"""


def rank_html(d: dict, market: str, label: str, top: int = 15) -> str:
    r = d["rank"].get(market)
    if not r:
        return f'<p class="muted">{label}暂无快照。</p>'
    rows = []
    for x in r["rows"][:top]:
        pct = x.get("pct") or 0
        main = x.get("main_net_wan") or 0
        near = " 🔥" if x.get("near_limit") else ""
        rows.append(
            "<tr><td class='num'>{rk}</td><td><b>{nm}</b>{near}"
            "<span class='code'>{cd}</span></td>"
            "<td class='num'>{px:.2f}</td>"
            "<td class='num {pc}'>{pct:+.2f}%</td>"
            "<td class='num'>{to:.1f}%</td>"
            "<td class='num {mc}'>{main:.0f}</td></tr>".format(
                rk=x.get("rank"),
                nm=esc(x.get("name")),
                near=near,
                cd=esc(x.get("code")),
                px=x.get("price") or 0,
                pc=cls_of(pct),
                pct=pct,
                to=x.get("turnover") or 0,
                mc=cls_of(main),
                main=main,
            )
        )
    return (
        f"<div class='tiny'>快照 {esc(r['ts'])} · 全市场 {r['total']} 只，展示前 {top}</div>"
        "<table class='rk'><thead><tr><th>#</th><th>名称</th><th>价</th>"
        "<th>涨幅</th><th>换手</th><th>主力(万)</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def tracked_html(d: dict) -> str:
    rows = []
    for t in d["tracked"]:
        sym = t["symbol"]
        disp = sym[2:] if sym[:2] in ("sh", "sz", "bj") else sym
        rows.append(
            "<tr><td><b>{d}</b><span class='code'>{s}</span></td>"
            "<td class='num'>{n}</td><td class='num'>{lv}</td>"
            "<td class='num'>{rp}</td><td>{a}</td><td>{b}</td></tr>".format(
                d=esc(disp), s=esc(sym), n=t["n"], lv=t["live_n"], rp=t["replay_n"],
                a=esc(t["first_ts"]), b=esc(t["last_ts"]),
            )
        )
    return (
        "<table class='rk'><thead><tr><th>标的</th><th>预测条数</th><th>实盘</th>"
        "<th>回放</th><th>首次</th><th>最近</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


# ------------------------------------------------------------------------ 模板

CSS = """
*{box-sizing:border-box}
body{margin:0;background:#f5f6f8;color:#1f2328;
  font:15px/1.7 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:32px 18px 64px}
.nav{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:18px}
.nav a{text-decoration:none;font-size:13.5px;color:#5b6470;background:#fff;
  border:1px solid #dfe3e8;border-radius:999px;padding:5px 14px}
.nav a:hover{border-color:#93c5fd;color:#1d4ed8}
.nav a.on{background:#2ca02c;color:#fff;border-color:#2ca02c}
h1{font-size:25px;margin:0 0 6px}
h2{font-size:17px;margin:0 0 4px;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.lead{color:#5b6470;font-size:14.5px;margin:0 0 10px}
.card{background:#fff;border:1px solid #dfe3e8;border-radius:12px;
  padding:18px 20px;margin:14px 0}
.muted{color:#6b7280;font-size:13px;line-height:1.65}
.tiny{color:#8a94a2;font-size:12px}
.code{display:block;font-size:11px;color:#9aa4b2;font-family:ui-monospace,monospace}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}
th,td{padding:7px 8px;border-bottom:1px solid #edf0f3;text-align:left;vertical-align:middle}
thead th{background:#f6f8fa;color:#5b6470;font-weight:600;font-size:12.5px;
  position:sticky;top:0}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.up{color:#c0392b;font-weight:600}.down{color:#1e7f43;font-weight:600}
.neu{color:#6b7280}
.pill{display:inline-block;font-size:11.5px;padding:2px 8px;border-radius:999px}
.pill.ok{background:#e8f5ec;color:#1e7f43}
.pill.off{background:#f1f3f5;color:#9aa4b2}
.kv{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:10px 0}
.kpi{background:#fafbfc;border:1px solid #eef1f4;border-radius:10px;padding:10px 12px}
.kpi .k{display:block;font-size:12px;color:#6b7280}
.kpi b{font-size:18px;font-variant-numeric:tabular-nums}
.kpi .sub{display:block;font-size:11.5px;color:#9aa4b2}
.wbar{height:5px;background:#eef1f4;border-radius:3px;overflow:hidden;
  display:inline-block;width:56px;vertical-align:middle;margin-right:6px}
.wbar i{display:block;height:100%;background:#3b82f6;border-radius:3px}
.offrow{opacity:.55}
.note{font-size:12px;color:#6b7280;max-width:330px}
.warn{background:#fffbeb;border-left:4px solid #f59e0b;padding:12px 16px;
  border-radius:6px;font-size:13.5px;margin:12px 0}
.tip{background:#eff6ff;border-left:4px solid #3b82f6;padding:12px 16px;
  border-radius:6px;font-size:13.5px;margin:12px 0}
.legend{display:flex;gap:16px;flex-wrap:wrap;align-items:center;
  font-size:12.5px;color:#6b7280;margin-top:6px}
.legend i{display:inline-block;width:14px;height:2.5px;border-radius:2px;
  vertical-align:middle;margin-right:5px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.tag{display:inline-block;font-size:12px;color:#5b6470;background:#eef2f7;
  border-radius:999px;padding:2px 10px;margin:0 6px 6px 0}
.btn{display:inline-block;text-decoration:none;background:#2ca02c;color:#fff;
  padding:10px 20px;border-radius:8px;font-size:14.5px;margin-top:4px}
.btn.alt{background:#3b82f6}
.btn:hover{filter:brightness(1.06)}
.foot{color:#6b7280;font-size:12.5px;margin-top:24px;line-height:1.75}
@media(max-width:820px){
  .kv{grid-template-columns:repeat(2,1fr)}
  .two{grid-template-columns:1fr}
  .note{display:none}
}
"""

TMPL = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>实时预测看板 · 真实运行快照</title>
<style>__CSS__</style>
</head>
<body>
<div class="wrap">

<div class="nav">
  <a href="index.html">首页</a>
  <a href="realtime.html">实时报价</a>
  <a href="live.html" class="on">预测看板</a>
  <a href="rank.html">涨幅榜</a>
</div>

<h1>实时预测看板 · 真实运行快照</h1>
<p class="lead">14 路分钟级专家 + Hedge 在线加权，每 3 秒随行情刷新一次，
预测未来 <b>1 / 3 / 5 分钟</b>，到期用真实价结算并回写误差、自动调权。</p>
<div>
  <span class="tag">__STOCK_NAME__（__STOCK_CODE__）</span>
  <span class="tag">快照时间 __SNAPSHOT_TS__</span>
  <span class="tag">数据来自 data/stockpred.db</span>
</div>

<div class="warn">
  <b>这一页是「真实运行结果的固化快照」，不是实时接口。</b><br>
  GitHub Pages 只能托管静态文件，跑不了行情轮询与在线调权。所以上面的时间点是固定的，
  数字是那一刻引擎的真实输出——不是示意值，也不会自己变化。想要每 3 秒刷新，需在本地运行
  <code>python live.py</code>（见页尾）。
</div>

<div class="card">
  <h2>即时状态</h2>
  <div class="kv">__KPI__</div>
  <div class="muted">__PRIMARY_LINE__</div>
</div>

<div class="card">
  <h2>14 路专家 <span class="tiny" style="font-weight:400">权重由 Hedge 按已实现误差在线自适应</span></h2>
  <div class="muted">「有效权重」= 先验权重 × 命中表现归一化后的实际占比；弃权的专家权重为 0，不参与合成。</div>
  __EXPERTS__
</div>

<div class="card">
  <h2>预测价 vs 到点真实价（1 分钟周期）</h2>
  __CHART__
</div>

<div class="card">
  <h2>东财涨幅榜 <span class="tiny" style="font-weight:400">引擎的选股入口</span></h2>
  <div class="muted">从榜单点进任意标的即可开始实时跟踪。下面是最近一次真实抓取的快照。</div>
  <div class="two" style="margin-top:12px">
    <div><h3>A 股</h3>__RANK_A__</div>
    <div><h3>科创板</h3>__RANK_KCB__</div>
  </div>
</div>

<div class="card">
  <h2>已跟踪标的</h2>
  <div class="muted">回放模式用于非交易时段：用最近交易日分钟线逐条推进，整条链路（结算 / 调权 / 统计）照样真实运转。</div>
  __TRACKED__
</div>

<div class="card">
  <h2>怎么让这个看板真的动起来</h2>
  <div class="tip">
    实时服务必须跑在<b>自己的机器</b>上（本机 / VPS / 内网穿透），Pages 托管不了常驻进程。
  </div>
  <pre class="muted" style="background:#fafbfc;border:1px solid #eef1f4;border-radius:8px;padding:12px;overflow:auto">python live.py --port 8020          # 启动实时服务
python live.py --rank               # 顺便起涨幅榜
python run.py --sync-db 600519      # 同步日线与指标入库</pre>
  <div class="muted">然后打开 <code>http://127.0.0.1:8020/live?code=600519</code>：
  每 3 秒刷新行情与 14 路预测，收盘后自动切换回放模式继续跑。</div>
</div>

<div class="foot">
  页面由 <code>tools/build_live_snapshot.py</code> 从 <code>data/stockpred.db</code> 直接生成，
  未经手工修改。数据库本身不入库（含 API Key），所以线上只发布这份派生快照。
</div>

</div>
</body>
</html>
"""


def main() -> None:
    symbol = sys.argv[1] if len(sys.argv) > 1 else "sh600519"
    d = load(symbol)
    lt = d["latest"] or {}
    es = d["experts"]
    on = [x for x in es if x.get("available")]
    snap_ts = str(lt.get("ts") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    q = d.get("quote") or {}
    name = q.get("name") or (lt.get("symbol") or symbol)
    code = symbol[2:] if symbol[:2] in ("sh", "sz", "bj") else symbol

    primary = (
        f"合成结果：{fmt_bps(lt.get('pred_bps'))} bps · 方向「{esc(lt.get('direction'))}」"
        f" · 置信 {(lt.get('conf') or 0) * 100:.0f}% · "
        f"80% 区间 {lt.get('lo', 0):.2f} ~ {lt.get('hi', 0):.2f}"
        f" · {len(on)}/{len(es)} 路参与加权"
    )
    if q.get("prev_close"):
        primary += (
            f"<br><span class='tiny'>快照价取 {esc(snap_ts)} 那一刻的真实成交价；"
            f"涨跌幅取 {esc(q.get('ts', ''))} 的行情，两者时刻不同，"
            f"所以与快照价×昨收（{q['prev_close']:.2f}）能对上但不完全等价。</span>"
        )

    html_out = (
        TMPL.replace("__CSS__", CSS)
        .replace("__STOCK_NAME__", esc(name))
        .replace("__STOCK_CODE__", esc(code))
        .replace("__SNAPSHOT_TS__", esc(snap_ts))
        .replace("__KPI__", kpi_html(d))
        .replace("__PRIMARY_LINE__", primary)
        .replace("__EXPERTS__", experts_html(d))
        .replace("__CHART__", svg_chart(d))
        .replace("__RANK_A__", rank_html(d, "a", "A 股"))
        .replace("__RANK_KCB__", rank_html(d, "kcb", "科创板"))
        .replace("__TRACKED__", tracked_html(d))
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # 固定 LF：仓库 .gitattributes 是 eol=lf，避免 Windows 上写出 CRLF 造成 diff 噪声
    OUT.write_text(html_out, encoding="utf-8", newline="\n")
    print(f"已生成 {OUT.relative_to(ROOT)}  ({len(html_out):,} 字节)")
    print(f"  标的 {symbol} | 快照 {snap_ts} | 专家 {len(on)}/{len(es)} 可用")
    print(f"  今日结算 " + "/".join(
        str((d['stats'].get(h) or {}).get('settled') or 0) for h in (1, 3, 5)
    ) + " (1/3/5 分钟)")


if __name__ == "__main__":
    main()
