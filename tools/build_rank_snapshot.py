"""把东财涨幅榜的真实抓取结果渲染成静态页 docs/rank.html。

背景：
    动态版在 `stockpred/live_web.py` 的 `/rank` 路由里，需要常驻进程。
    GitHub Pages 托管不了进程，所以线上要有一份**真实抓取结果的固化快照**，
    否则访客看到的站点里根本没有「爬虫」这个东西。

用法：
    python tools/build_rank_snapshot.py            # 用库里最近的快照（离线可用）
    python tools/build_rank_snapshot.py --fetch    # 先实时抓一次再生成（数据最新）

输出：
    docs/rank.html（A 股 / 科创板 各 Top50，前端切 tab，零外部依赖）
"""
from __future__ import annotations

import html
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "stockpred.db"
OUT = ROOT / "docs" / "rank.html"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

C_UP = "#c0392b"
C_DOWN = "#1e7f43"

NAV = """<div class="nav">
  <a href="index.html">首页</a>
  <a href="realtime.html">实时报价</a>
  <a href="live.html">预测看板</a>
  <a href="rank.html" class="on">涨幅榜</a>
</div>"""


def esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def maybe_fetch() -> str:
    """实时抓一次写库，返回提示文案。"""
    try:
        from stockpred import db, rank_fetcher

        msgs = []
        for mk, label in (("a", "A 股"), ("kcb", "科创板")):
            r = rank_fetcher.fetch_rank(mk, 50)
            if r.get("ok") and r.get("rows"):
                db.save_rank_snapshot(mk, r["rows"], int(r.get("total") or 0))
                msgs.append(f"{label} {len(r['rows'])} 条")
            else:
                msgs.append(f"{label} 失败：{r.get('error')}")
        return "已实时抓取 · " + " / ".join(msgs)
    except Exception as e:
        return f"实时抓取失败（{type(e).__name__}），改用库内快照"


def load() -> dict:
    c = sqlite3.connect(DB, timeout=15)
    out = {}
    for mk in ("a", "kcb"):
        r = c.execute(
            "select ts, total, rows_json from rank_snapshots where market=? "
            "order by id desc limit 1",
            (mk,),
        ).fetchone()
        if r:
            out[mk] = {"ts": r[0], "total": r[1], "rows": json.loads(r[2])}
    # 历史快照列表（时间 + 条数），证明爬虫是持续在跑的
    out["_history"] = [
        {"ts": t, "market": m, "n": n}
        for t, m, n in c.execute(
            "select ts, market, count(*) from rank_snapshots group by ts, market "
            "order by ts desc limit 12"
        )
    ]
    out["_snap_count"] = c.execute("select count(*) from rank_snapshots").fetchone()[0]
    c.close()
    return out


CSS = """
*{box-sizing:border-box}
body{margin:0;background:#f5f6f8;color:#1f2328;
  font:15px/1.7 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
.wrap{max-width:1060px;margin:0 auto;padding:26px 18px 64px}
.nav{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:18px}
.nav a{text-decoration:none;font-size:13.5px;color:#5b6470;background:#fff;
  border:1px solid #dfe3e8;border-radius:999px;padding:5px 14px}
.nav a:hover{border-color:#93c5fd;color:#1d4ed8}
.nav a.on{background:#2ca02c;color:#fff;border-color:#2ca02c}
h1{font-size:24px;margin:0 0 6px}
h2{font-size:16.5px;margin:0 0 8px}
.lead{color:#5b6470;font-size:14px;margin:0 0 12px}
.card{background:#fff;border:1px solid #dfe3e8;border-radius:12px;
  padding:18px 20px;margin:14px 0}
.muted{color:#6b7280;font-size:13px;line-height:1.65}
.tiny{color:#8a94a2;font-size:12px}
.warn{background:#fffbeb;border-left:4px solid #f59e0b;padding:12px 16px;
  border-radius:6px;font-size:13.5px;margin:12px 0}
.tip{background:#eff6ff;border-left:4px solid #3b82f6;padding:12px 16px;
  border-radius:6px;font-size:13.5px;margin:12px 0}
.badge{display:inline-block;font-size:11.5px;background:#eef2f7;color:#5b6470;
  border-radius:999px;padding:2px 9px;vertical-align:middle;margin-left:6px}
.tabs{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
.tab{cursor:pointer;font-size:13.5px;padding:6px 16px;border-radius:999px;
  background:#f3f4f6;color:#5b6470;border:1px solid transparent;user-select:none}
.tab:hover{border-color:#c7d2fe}
.tab.on{background:#eef2ff;color:#1d4ed8;border-color:#c7d2fe;font-weight:600}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:7px 8px;border-bottom:1px solid #edf0f3;text-align:right;
  white-space:nowrap}
th:nth-child(2),td:nth-child(2),th:nth-child(1),td:nth-child(1){text-align:left}
thead th{background:#f6f8fa;color:#5b6470;font-weight:600;font-size:12.5px;
  position:sticky;top:0;z-index:2}
tbody tr:hover{background:#f8fafc}
.up{color:#c0392b;font-weight:600}.down{color:#1e7f43;font-weight:600}
.neu{color:#6b7280}
.code{display:block;font-size:11px;color:#9aa4b2;font-family:ui-monospace,monospace}
.fire{font-size:11px;color:#c0392b;background:#fdeaea;border-radius:4px;
  padding:1px 5px;margin-left:5px;vertical-align:middle}
.scroll{max-height:66vh;overflow:auto;border:1px solid #eef1f4;border-radius:8px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:12px 0}
.st{background:#fafbfc;border:1px solid #eef1f4;border-radius:10px;padding:10px 12px}
.st span{display:block;font-size:12px;color:#6b7280}
.st b{font-size:18px;font-variant-numeric:tabular-nums}
.foot{color:#6b7280;font-size:12.5px;margin-top:22px;line-height:1.75}
pre{background:#fafbfc;border:1px solid #eef1f4;border-radius:8px;padding:12px;
  overflow:auto;font-size:12.5px;color:#374151}
@media(max-width:760px){.stats{grid-template-columns:repeat(2,1fr)}}
"""

TMPL = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>涨幅榜 · 东方财富（真实抓取快照）</title>
<style>__CSS__</style>
</head>
<body>
<div class="wrap">
__NAV__
<h1>涨幅榜 <span class="badge">数据源：东方财富 push2</span></h1>
<p class="lead">爬虫按<b>涨跌幅降序</b>抓取沪深全市场与科创板，各留前 50 名。
每行可点，跳转到实时看板查看该标的的 14 路专家预测。</p>

<div class="warn">
  <b>这是爬虫最近一次真实抓取的结果快照，不会自己刷新。</b><br>
  动态版在 <code>python live.py --rank</code> 起的服务里（<code>/rank</code> 路由，每 30 秒重抓）。
  Pages 托管不了常驻进程，所以线上这份是固化的。抓取时刻 __FETCH_TS__，
  行情时刻 __QUOTE_TS__ —— <b>数字是真的，只是停在那一刻</b>。
</div>

<div class="card">
  <div class="tabs">
    <div class="tab on" id="tabA" onclick="pick('a')">A股（沪深全市场）</div>
    <div class="tab" id="tabK" onclick="pick('kcb')">科创板</div>
    <span class="muted" id="meta"></span>
  </div>
  <div class="stats" id="stats"></div>
  <div class="scroll"><table id="tbl"></table></div>
  <div class="muted" style="margin-top:8px" id="note"></div>
</div>

<div class="card">
  <h2>爬虫在做什么</h2>
  <table style="font-size:13px">
    <thead><tr><th style="text-align:left">项</th><th style="text-align:left">说明</th></tr></thead>
    <tbody>
      <tr><td style="text-align:left"><b>接口</b></td>
        <td style="text-align:left"><code>push2.eastmoney.com/api/qt/clist/get</code>，
        免 token、免登录，按 <code>f3</code>（涨跌幅）降序取前 50</td></tr>
      <tr><td style="text-align:left"><b>范围</b></td>
        <td style="text-align:left">A 股 = 沪深京全市场（剔除停牌）；科创板 = 独立板块 622 只</td></tr>
      <tr><td style="text-align:left"><b>去重</b></td>
        <td style="text-align:left">同一分钟内多次抓取只更新时间戳，内容相同不重复入库</td></tr>
      <tr><td style="text-align:left"><b>收盘后</b></td>
        <td style="text-align:left">东财行情会冻结（<code>f124</code> 停在 15:00），
        页面分开展示「抓取时间」与「行情时间」，不会假装在刷新</td></tr>
      <tr><td style="text-align:left"><b>下游</b></td>
        <td style="text-align:left">点任意一行 → 实时看板；本机版可直接开跟踪、输入自己的大模型 Key</td></tr>
    </tbody>
  </table>
</div>

<div class="card">
  <h2>抓取历史 <span class="tiny" style="font-weight:400">库内共 __SNAP_COUNT__ 份快照</span></h2>
  <div class="muted">下面是最近 12 次记录，证明这是持续运行的爬虫，不是一次性截图。</div>
  __HISTORY__
</div>

<div class="card">
  <h2>本地跑真·实时榜单</h2>
  <pre>python live.py --rank            # 起服务并挂载 /rank
python run.py --rank             # 或只抓一次落库</pre>
  <div class="muted">然后访问 <code>http://127.0.0.1:8020/rank</code>：
  每 30 秒重抓、A股/科创板切换、点击行直接开始跟踪。收盘后自动降频并提示冻结。</div>
</div>

<div class="foot">
  本页由 <code>tools/build_rank_snapshot.py</code> 从 <code>data/stockpred.db</code>
  的 <code>rank_snapshots</code> 表直接生成，未经手工修改。
</div>
</div>

<script>
const DATA = __DATA__;
let market = 'a';

function fmt(v, d){ return (v==null||isNaN(v)) ? '—' : Number(v).toFixed(d==null?2:d); }
function signed(v, d){
  if (v==null||isNaN(v)) return '<span class="neu">—</span>';
  const c = v>0?'up':(v<0?'down':'neu');
  const s = (v>0?'+':'') + Number(v).toFixed(d==null?2:d);
  return '<span class="'+c+'">'+s+'</span>';
}

function pick(m){
  market = m;
  document.getElementById('tabA').classList.toggle('on', m==='a');
  document.getElementById('tabK').classList.toggle('on', m==='kcb');
  render();
}

function render(){
  const d = DATA[market];
  const tbl = document.getElementById('tbl');
  if(!d){ tbl.innerHTML = '<tbody><tr><td class="muted">无数据</td></tr></tbody>'; return; }
  const rows = d.rows || [];

  const upN = rows.filter(r=>r.pct>0).length;
  const limitN = rows.filter(r=>r.near_limit).length;
  const avgPct = rows.length ? rows.reduce((a,r)=>a+(r.pct||0),0)/rows.length : 0;
  const mainSum = rows.reduce((a,r)=>a+(r.main_net_wan||0),0);
  document.getElementById('stats').innerHTML =
    '<div class="st"><span>全市场标的</span><b>'+d.total+'</b></div>' +
    '<div class="st"><span>榜内上涨</span><b>'+upN+'/50</b></div>' +
    '<div class="st"><span>榜内涨停/近涨停</span><b style="color:#c0392b">'+limitN+'</b></div>' +
    '<div class="st"><span>榜内主力净额合计</span><b style="color:'+(mainSum>0?'#c0392b':'#1e7f43')+'">'+
      (mainSum>0?'+':'') + (mainSum/10000).toFixed(2) + ' 亿</b></div>';

  document.getElementById('meta').innerHTML =
    '抓取 <b>'+d.ts+'</b> · 行情 <b>'+(rows[0]&&rows[0].quote_ts||'—')+'</b>';

  let h = '<thead><tr><th>#</th><th>名称</th><th>价</th><th>涨幅</th>' +
          '<th>涨速</th><th>换手</th><th>量比</th><th>成交额(亿)</th>' +
          '<th>主力净额(万)</th><th>主力占比</th><th>总市值(亿)</th></tr></thead><tbody>';
  rows.forEach(function(r){
    h += '<tr data-code="'+r.code+'" style="cursor:pointer">' +
      '<td>'+r.rank+'</td>' +
      '<td><b>'+r.name+'</b>'+(r.near_limit?'<span class="fire">近涨停</span>':'')+
        '<span class="code">'+r.code+'</span></td>' +
      '<td>'+fmt(r.price)+'</td>' +
      '<td>'+signed(r.pct)+'%</td>' +
      '<td>'+signed(r.speed,2)+'</td>' +
      '<td>'+fmt(r.turnover,1)+'%</td>' +
      '<td>'+fmt(r.liangbi,2)+'</td>' +
      '<td>'+fmt(r.amount_yi,2)+'</td>' +
      '<td>'+signed(r.main_net_wan,0)+'</td>' +
      '<td>'+signed(r.main_pct,1)+'%</td>' +
      '<td>'+fmt(r.mktcap_yi,1)+'</td></tr>';
  });
  h += '</tbody>';
  tbl.innerHTML = h;
  document.getElementById('note').textContent =
    '点击任意一行 → 打开该股的实时报价（浏览器直连行情，3 秒刷新）。' +
    '预测快照目前只固化了贵州茅台（600519）一份；其他标的的预测需在本机运行 python live.py。';
}

function go(code){
  location.href = 'realtime.html?code=' + encodeURIComponent(code);
}
document.getElementById('tbl').addEventListener('click', function(e){
  const tr = e.target.closest('tr[data-code]');
  if (tr) go(tr.dataset.code);
});
render();
</script>
</body>
</html>
"""


def render_history(d: dict) -> str:
    hist = d.get("_history") or []
    if not hist:
        return '<p class="muted">暂无历史记录。</p>'
    rows = []
    for h in hist:
        label = "A 股" if h["market"] == "a" else "科创板"
        rows.append(
            f"<tr><td style='text-align:left'>{esc(h['ts'])}</td>"
            f"<td style='text-align:left'>{label}</td>"
            f"<td style='text-align:right'>{h['n']}</td></tr>"
        )
    return (
        "<table><thead><tr><th style='text-align:left'>抓取时刻</th>"
        "<th style='text-align:left'>市场</th>"
        "<th style='text-align:right'>条数</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def main() -> None:
    fetch_note = ""
    if "--fetch" in sys.argv:
        fetch_note = maybe_fetch()
        print(fetch_note)

    d = load()
    if "a" not in d or "kcb" not in d:
        print("库里没有榜单快照，先跑 python run.py --rank", file=sys.stderr)
        sys.exit(1)

    # 内嵌数据（只保留渲染需要的字段，减小体积）
    keep = ("rank", "code", "name", "price", "pct", "speed", "turnover",
            "liangbi", "amount_yi", "main_net_wan", "main_pct",
            "mktcap_yi", "near_limit", "quote_ts")
    payload = {}
    for mk in ("a", "kcb"):
        payload[mk] = {
            "ts": d[mk]["ts"],
            "total": d[mk]["total"],
            "rows": [{k: x.get(k) for k in keep} for x in d[mk]["rows"]],
        }
    data_js = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    data_js = data_js.replace("</", "<\\/")

    quote_ts = d["a"]["rows"][0].get("quote_ts") or "—"
    out = (
        TMPL.replace("__CSS__", CSS)
        .replace("__NAV__", NAV)
        .replace("__DATA__", data_js)
        .replace("__HISTORY__", render_history(d))
        .replace("__SNAP_COUNT__", str(d.get("_snap_count") or 0))
        .replace("__FETCH_TS__", esc(d["a"]["ts"]))
        .replace("__QUOTE_TS__", esc(quote_ts))
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(out, encoding="utf-8", newline="\n")
    print(f"已生成 {OUT.relative_to(ROOT)}  ({len(out):,} 字节)")
    for mk in ("a", "kcb"):
        print(f"  {mk}: {len(d[mk]['rows'])} 条 / 全市场 {d[mk]['total']} · 快照 {d[mk]['ts']}")


if __name__ == "__main__":
    main()
