"""实时跟踪与涨幅榜 Web：页面 + JSON API（配合 web.py 或独立 live.py 启动）。

路由
----
GET  /live?code=600519            实时跟踪页（分时 + 预测带 + 专家表 + 拐点 + 日对比）
GET  /rank                        东方财富涨幅榜（A股 / 科创板，各前 50，点击直达跟踪页）
GET  /api/live?code=&start=       引擎快照；start=1 时确保引擎已启动
GET  /api/rank?market=a|kcb       涨幅榜（东财，30 秒缓存）
GET  /api/rank/history?market=    榜单历史快照（每 5 分钟一份）
GET  /api/daily?code=             日线 + 日级预测对比（预测 vs 真实收盘）
GET  /api/settings                大模型配置（Key 只回掩码）
POST /api/settings                保存供应商/模型/Key（Key 明文只进本机 DB）
POST /api/llm/test                连通性测试
POST /api/llm/run?code=           立即让大模型预测一次
POST /api/engine/stop?code=       停止某只票的引擎

Key 输入 → 跟踪 的产品流（按用户要求）
--------------------------------------
涨幅榜点股票 → /live 页；若大模型未配置，顶部是 Key 设置卡，
点「开始实时跟踪」后引擎启动，价格每 3 秒刷新、每 10 秒落一条预测、
到期自动结算并更新各专家权重。Key 保存后大模型专家按间隔参与加权。
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse

from . import db, rank_fetcher
from . import llm_live

# 延迟导入：live_engine / realtime 只有真正用到才加载

# 对外分享模式（web.py 带口令启动时置位）。这一模式下**全局配置写入一律拒绝**：
# /api/settings 写的是 data/stockpred.db 里的唯一一份配置，公网任何访客都能覆盖
# 机器主人的 Key；/api/llm/run 则会直接用那份配置去烧主人的额度。
# 访客要跑大模型请走「股票深度分析」页（/classic），那里按会话隔离、不落库。
PUBLIC_MODE = os.environ.get("STOCKPRED_PUBLIC", "").strip() in ("1", "true", "yes", "on")

# 对外模式同时跟踪的股票上限：每个引擎一个 3 秒轮询线程，不限的话
# 访客多点几十只就把本机线程和行情请求堆满
MAX_ENGINES_PUBLIC = 6

# 对外模式允许访客把 Key 发往的服务商（防止 test 接口被当成任意地址的探针）
_ALLOWED_LLM_HOSTS = ("dashscope.aliyuncs.com", "api.deepseek.com", "open.bigmodel.cn")


def _llm_host_allowed(base_url: str) -> bool:
    if not base_url:
        return True                     # 留空 = 用预设服务商
    try:
        host = (urllib.parse.urlparse(base_url).hostname or "").lower()
    except ValueError:
        return False
    return any(host == h or host.endswith("." + h) for h in _ALLOWED_LLM_HOSTS)


def _json_bytes(obj, status: int = 200):
    body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
    return status, "application/json; charset=utf-8", body


def handle_get(path: str, qs: dict):
    """返回 (status, ctype, body) 或 None（不是本模块的路由）。"""
    q = {k: v[0] if isinstance(v, list) else v for k, v in (qs or {}).items()}
    if path in ("/", "/index.html"):
        return 200, "text/html; charset=utf-8", _page_home(q).encode("utf-8")
    if path == "/live":
        return 200, "text/html; charset=utf-8", _page_live(q).encode("utf-8")
    if path == "/rank":
        return 200, "text/html; charset=utf-8", _page_rank(q).encode("utf-8")
    if path == "/api/live":
        return _api_live(q)
    if path == "/api/rank":
        return _json_bytes(rank_fetcher.fetch_rank(q.get("market", "a"),
                                                   top=int(q.get("top", 50) or 50)))
    if path == "/api/rank/history":
        return _json_bytes({"rows": rank_fetcher.history(q.get("market", "a"),
                                                         limit=int(q.get("limit", 48) or 48))})
    if path == "/api/daily":
        return _api_daily(q)
    if path == "/api/compare":
        return _api_compare(q)
    if path == "/api/settings":
        return _json_bytes(llm_live.public_config())
    if path == "/api/db":
        return _json_bytes(db.db_info())
    if path == "/api/engines":
        from .live_engine import engines_status
        return _json_bytes({"engines": engines_status()})
    return None


def handle_post(path: str, qs: dict, body: bytes):
    q = {k: v[0] if isinstance(v, list) else v for k, v in (qs or {}).items()}
    try:
        payload = json.loads(body.decode("utf-8")) if body else {}
    except Exception:
        payload = {}
    if path == "/api/settings":
        if PUBLIC_MODE:
            return _json_bytes({
                "ok": False,
                "note": "对外分享模式：不写入全局大模型配置（那会覆盖机器主人的 Key）。"
                        "要跑大模型请打开「股票深度分析」页，在那里填你自己的 Key，"
                        "按会话隔离、不落库。",
            }, status=403)
        return _json_bytes(llm_live.save_config(payload))
    if path == "/api/llm/test":
        # 把页面上正在输入的地址/模型/Key 一起带过去：用户改了输入框却测到旧 Key
        # 是最容易"测着成功、跑起来失败"的坑。
        if PUBLIC_MODE and not _llm_host_allowed(str(payload.get("base_url") or "")):
            return _json_bytes({
                "ok": False,
                "note": "对外分享模式：只允许请求预设的模型服务商（通义千问 / DeepSeek / 智谱）。",
            }, status=403)
        return _json_bytes(llm_live.test_connection(payload))
    if path == "/api/llm/run":
        if PUBLIC_MODE:
            return _json_bytes({
                "ok": False,
                "note": "对外分享模式：已关闭「立即跑一次大模型」——它会用机器主人配置的 Key。"
                        "请用「股票深度分析」页，Key 按会话隔离。",
            }, status=403)
        from .live_engine import get_engine
        e = get_engine(q.get("code", ""), autostart=False)
        return _json_bytes(e.run_llm_now() if getattr(e, "state", None) or e.thread.is_alive()
                           else {"ok": False, "note": "引擎未启动，请先开始跟踪"})
    if path == "/api/engine/stop":
        from .live_engine import get_engine
        e = get_engine(q.get("code", ""), autostart=False)
        e.stop()
        return _json_bytes({"ok": True, "note": "已请求停止"})
    return None


# =====================================================================
# API 实现
# =====================================================================
def _api_live(q: dict):
    code = (q.get("code") or "").strip()
    if not code:
        return _json_bytes({"error": "缺少 code 参数"}, 400)
    from .realtime import fetch_quote, is_open
    from .live_engine import get_engine
    autostart = str(q.get("start", "0")) in ("1", "true", "yes")
    e = get_engine(code, autostart=False)
    started = e.thread.is_alive() or bool(e.state)
    if autostart and not started and PUBLIC_MODE:
        from .live_engine import engines_status
        n = len(engines_status())
        if n >= MAX_ENGINES_PUBLIC:
            return _json_bytes({
                "ok": False,
                "note": f"对外分享模式最多同时跟踪 {MAX_ENGINES_PUBLIC} 只（当前 {n} 只）。"
                        "先停掉不看的再开新的。",
            }, 429)
    if autostart and not started:
        e = get_engine(code, autostart=True)
        started = True
        # 给后台线程一点时间产出第一份快照（回放模式拉分钟线约 1~2 秒）
        for _ in range(16):
            if e.state:
                break
            time.sleep(0.25)
    if not started:
        quote = fetch_quote(code)
        return _json_bytes({
            "started": False,
            "symbol": e.symbol,
            "name": quote.name or code,
            "market_open": is_open(),
            "quote": quote.to_dict() if quote.price > 0 else None,
            "llm": llm_live.public_config(),
        })
    snap = e.snapshot()
    snap["started"] = True
    snap["market_open"] = is_open()
    snap["llm"] = snap.get("llm") or {"config": llm_live.public_config()}
    return _json_bytes(snap)


def _api_daily(q: dict):
    code = (q.get("code") or "").strip()
    if not code:
        return _json_bytes({"error": "缺少 code 参数"}, 400)
    bars = db.daily_history(code, limit=int(q.get("days", 60) or 60))
    forecasts = db.daily_forecast_history(code, horizon=1, limit=90)
    signals = db.daily_signal_history(code, limit=20)
    close_by_date = {str(b["date"])[:10]: b["close"] for b in bars}
    compare = []
    for f in forecasts:
        tgt = str(f.get("target_date") or "")[:10]
        actual = close_by_date.get(tgt)
        compare.append({
            "base_date": str(f.get("base_date"))[:10],
            "target_date": tgt,
            "pred_close": f.get("pred_close"),
            "actual_close": actual,
            "err_pct": f.get("err_pct"),
            "lo": f.get("lo"), "hi": f.get("hi"),
            "direction": f.get("direction"),
            "hit": (None if (actual is None or not f.get("pred_close")) else
                    (f["pred_close"] > f.get("base_close") or 0) == (actual > (f.get("base_close") or 0))),
        })
    settled = [c for c in compare if c.get("actual_close")]
    mae = None
    if settled:
        errs = [abs(float(c["err_pct"])) for c in settled if c.get("err_pct") is not None]
        mae = round(sum(errs) / len(errs), 3) if errs else None
    hits = [1 for c in settled if c.get("hit")]
    return _json_bytes({
        "bars": bars, "compare": compare, "signals": signals,
        "stats": {"n": len(settled), "mae_pct": mae,
                  "hit": round(len(hits) / len(settled), 3) if settled else None},
    })


def _api_compare(q: dict):
    code = (q.get("code") or "").strip()
    date = (q.get("date") or "").strip()
    if not code:
        return _json_bytes({"error": "缺少 code 参数"}, 400)
    from . import llm_live as _  # noqa: F401  (保持依赖显式)
    bars = db.intraday_bars(code, trade_date=date or None, limit=600)
    preds = db.live_predictions_on(code, date) if date else db.recent_live_predictions(code, 400)
    return _json_bytes({
        "date": date, "bars": bars,
        "predictions": preds,
        "turns": db.turn_points(code, trade_date=date or None, limit=80),
    })


# =====================================================================
# 页面
# =====================================================================
_CSS = """
:root{--bg:#f6f7f9;--card:#fff;--line:#e5e7eb;--txt:#1f2937;--muted:#6b7280;
--red:#dc2626;--green:#16a34a;--blue:#2563eb;--amber:#d97706;--violet:#7c3aed}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--txt);
font:14px/1.55 "Segoe UI","Microsoft YaHei",sans-serif}
a{color:var(--blue);text-decoration:none}a:hover{text-decoration:underline}
.wrap{max-width:1180px;margin:0 auto;padding:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:14px 16px;margin-bottom:14px}
h1{font-size:19px;margin:2px 0 10px}h2{font-size:15px;margin:0 0 8px}
.muted{color:var(--muted);font-size:12px}
.grid{display:grid;gap:12px}.g2{grid-template-columns:1fr 1fr}.g3{grid-template-columns:1fr 1fr 1fr}
.g4{grid-template-columns:repeat(4,1fr)}
@media(max-width:860px){.g2,.g3,.g4{grid-template-columns:1fr}}
.price{font-size:30px;font-weight:700}.up{color:var(--red)}.dn{color:var(--green)}
.badge{display:inline-block;padding:2px 9px;border-radius:99px;font-size:12px;
background:#eef2ff;color:#4338ca;margin-left:8px}
.badge.gray{background:#f3f4f6;color:#4b5563}
.badge.warn{background:#fef3c7;color:#92400e}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:500;background:#fafafa}
tr.click:hover{cursor:pointer;background:#f0f7ff}
input,select,button{font:inherit;padding:6px 10px;border:1px solid var(--line);
border-radius:8px;background:#fff}
button{cursor:pointer;background:var(--blue);color:#fff;border-color:var(--blue)}
button.ghost{background:#fff;color:var(--blue)}
button:disabled{opacity:.5;cursor:default}
.tag{font-size:12px;padding:1px 7px;border-radius:6px;background:#f3f4f6;color:#374151;margin-right:6px}
canvas{width:100%;display:block}
.kv{display:flex;justify-content:space-between;border-bottom:1px dashed var(--line);
padding:3px 0;font-size:13px}
.kv b{font-weight:600}
.nav{display:flex;gap:14px;align-items:center;padding:10px 0}
.nav a{font-weight:600}
.limit{color:var(--red);font-weight:700}
"""

_BASE_JS = """
function fmt(x, d=2){ if(x===null||x===undefined||isNaN(x)) return '-'; return Number(x).toFixed(d); }
function cls(v){ return v>0?'up':(v<0?'dn':''); }
async function jget(u){ const r=await fetch(u,{cache:'no-store'}); if(!r.ok) throw new Error('HTTP '+r.status); return r.json(); }
async function jpost(u, data){ const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify(data||{})}); return r.json(); }
"""


def _page_live(q: dict) -> str:
    code = q.get("code", "600519")
    return f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>实时跟踪 {code}</title><style>{_CSS}</style></head><body>
<div class="wrap">
<div class="nav"><a href="/rank">← 涨幅榜</a><a href="/">首页</a>
<span class="muted" id="modeBadge"></span></div>
<h1><span id="sname">…</span> <span class="muted" id="scode">{code}</span>
<span class="price" id="price">--</span>
<span class="muted" id="pct"></span>
<span class="badge gray" id="updAt"></span></h1>

<div class="card" id="setupCard">
  <h2>① 大模型接入（可选，用于大模型价格预测一路）</h2>
  <div class="muted" style="margin-bottom:8px">
    Key 只保存在本机数据库（<code>data/stockpred.db</code>），不会上传；不填 Key
    也可以开始跟踪：其余 13 路量化专家全部可用，其中新闻政策面在无 Key 时走本地词典情绪；
    大模型那一路会自动剔除并重新归一化。
    调用按间隔限流（默认 5 分钟一次），失败自动跳过。</div>
  <div class="grid g4" style="align-items:end">
    <div><div class="muted">供应商</div><select id="llmProvider"></select></div>
    <div><div class="muted">Base URL</div><input id="llmBase" style="width:100%" placeholder="OpenAI 兼容端点"></div>
    <div><div class="muted">模型</div><input id="llmModel" style="width:100%" placeholder="如 qwen3.7-plus"></div>
    <div><div class="muted">API Key（掩码回显，传 - 清除）</div>
      <input id="llmKey" type="password" style="width:100%" placeholder="sk-..."></div>
  </div>
  <div id="llmTip" class="muted" style="margin-top:6px;font-size:12.5px"></div>
  <div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap;align-items:center">
    <button onclick="saveLlm()">保存配置</button>
    <button class="ghost" onclick="testLlm()">测试连接</button>
    <span id="llmStatus" class="muted"></span>
  </div>
  <hr style="border:none;border-top:1px solid var(--line);margin:12px 0">
  <h2 style="display:inline">② 开始跟踪</h2>
  <div style="margin-top:8px;display:flex;gap:8px;flex-wrap:wrap">
    <button id="btnStart" onclick="startTrack()">开始实时跟踪（价格每 3 秒刷新）</button>
    <button class="ghost" onclick="runLlmOnce()">让大模型立即预测一次</button>
    <span class="muted" id="llmLiveStatus"></span>
    <span class="muted" id="trackNote"></span>
  </div>
</div>

<div id="dash" style="display:none">
<div class="grid g4">
  <div class="card"><h2>未来 5 分钟预测（主周期）</h2>
    <div id="predMain" class="muted">等待数据…</div></div>
  <div class="card"><h2>联合决策</h2><div id="predDir" class="muted">-</div></div>
  <div class="card"><h2>盘中状态</h2><div id="micro" class="muted">-</div></div>
  <div class="card"><h2>结算与成绩（样本外）</h2><div id="statsBox" class="muted">-</div></div>
</div>
<div class="card"><h2>分时 · 预测对比 <span class="muted">蓝线=真实价，橙虚线=5分钟预测线，灰带=80%区间，▼▲=已确认顶/底拐点</span></h2>
  <canvas id="chart" height="330"></canvas></div>
<div class="grid g2">
  <div class="card"><h2>专家池 <span class="muted" id="expCount"></span>（在线权重随真实误差自适应）</h2>
    <div style="max-height:260px;overflow:auto"><table id="expTbl"></table></div></div>
  <div class="card"><h2>日内拐点</h2><div id="turnBox" class="muted">-</div></div>
</div>
<div class="card"><h2>新闻 / 政策 / 舆论面
  <span class="muted">慢变量：低频抓取 + 时间衰减 + 幅度封顶，只给窄幅方向偏置</span></h2>
  <div id="newsBox" class="muted">尚未抓取…</div>
  <div class="muted" style="margin-top:8px;font-size:12px">
    ⚠️ 这一路<b>没有历史新闻序列，无法回测</b>，权重只能给很小的先验，再由在线学习按误差慢慢调整。
    不配大模型 Key 时走本地词典情绪（颗粒度更粗，会被标成「词典情绪」而非大模型结论）。
  </div></div>
<div class="card"><h2>每日预测 vs 真实收盘（次日自动回填）</h2>
  <div class="grid g2"><div><canvas id="dayChart" height="220"></canvas></div>
  <div style="max-height:220px;overflow:auto"><table id="dayTbl"></table></div></div>
  <div class="muted" id="dayStats"></div></div>
</div>
<p class="muted">仅供技术研究与学习，不构成任何投资建议。5 分钟级别方向命中率的合理预期是
50~55%，任何单笔预测的误差都可能大于预测幅度本身；请把区间和"震荡"当回事。</p>
</div>
<script>{_BASE_JS}
const CODE = new URLSearchParams(location.search).get('code') || '{code}';
let providerSel = document.getElementById('llmProvider');
let PRESETS = {{}};
let started=false, timer=null;

function fillPresets(cfg){{
  PRESETS = {{}};
  (cfg.presets||[]).forEach(p=>{{ PRESETS[p.id]=p; }});
  providerSel.innerHTML = (cfg.presets||[]).map(p=>`<option value="${{p.id}}" ${{cfg.provider===p.id?'selected':''}}>${{p.label}}</option>`).join('');
  document.getElementById('llmBase').value = cfg.base_url||'';
  document.getElementById('llmModel').value = cfg.model||'';
  const tip = document.getElementById('llmTip');
  const cur = PRESETS[cfg.provider];
  if (tip) tip.textContent = (cur && cur.note) || '';
  const st=document.getElementById('llmStatus');
  st.textContent = cfg.ready ? `已配置：${{cfg.preset}} / ${{cfg.model}}（Key ${{cfg.key_masked||'无'}}，来源 ${{cfg.key_source||'-'}}）`
                             : '未配置完整（不填 Key 也可开始跟踪）';
}}

async function loadCfg(){{
  const cfg = await jget('/api/settings'); fillPresets(cfg); return cfg;
}}
providerSel.onchange = ()=>{{
  // 换供应商时把该预设的地址与模型填进输入框，否则会带着上一家的模型名去请求（必然 404）
  const p = PRESETS[providerSel.value];
  if (!p) return;
  if (p.base_url) document.getElementById('llmBase').value = p.base_url;
  if (p.model) document.getElementById('llmModel').value = p.model;
  const tip = document.getElementById('llmTip');
  if (tip) tip.textContent = p.note || '';
}};
async function saveLlm(){{
  const payload={{provider:providerSel.value, base_url:document.getElementById('llmBase').value,
    model:document.getElementById('llmModel').value, api_key:document.getElementById('llmKey').value}};
  const cfg=await jpost('/api/settings', payload); fillPresets(cfg);
  document.getElementById('llmKey').value='';
  document.getElementById('llmStatus').textContent = cfg.ready?`已保存：${{cfg.model}}`:'已保存（配置不完整，大模型一路将不参与）';
}}
async function testLlm(){{
  const st=document.getElementById('llmStatus');
  st.textContent='测试中…（最多 20 秒）';
  const payload={{base_url:document.getElementById('llmBase').value,
    model:document.getElementById('llmModel').value,
    api_key:document.getElementById('llmKey').value}};
  const r=await jpost('/api/llm/test', payload);
  if(r.ok){{
    st.textContent=`连接成功（${{r.elapsed}}s）· 模型回「${{r.text||'OK'}}」· 用的是${{r.key_source}} ${{r.key_masked||''}}`;
  }} else {{
    st.textContent=`失败：${{r.note}} ｜ 地址 ${{r.base_url||'-'}} · 模型 ${{r.model||'-'}} · Key ${{r.key_masked||'无'}}（${{r.key_source||'-'}}）`;
  }}
}}
async function runLlmOnce(){{
  document.getElementById('trackNote').textContent='大模型预测中（5~20 秒）…';
  const r=await jpost('/api/llm/run?code='+CODE);
  document.getElementById('trackNote').textContent = r.ok?`完成：${{r.direction}} ${{fmt(r.bps,1)}}bps（${{r.model}}，${{r.elapsed}}s）`:'失败：'+r.note;
  poll();
}}
function startTrack(){{
  document.getElementById('btnStart').disabled=true;
  document.getElementById('trackNote').textContent='启动中…';
  jget('/api/live?code='+CODE+'&start=1').then(s=>{{
    started=true; document.getElementById('dash').style.display='block';
    render(s); timer=setInterval(poll, 2500);
    document.getElementById('trackNote').textContent='跟踪中（非交易时段自动进入回放模式）';
  }}).catch(e=>{{document.getElementById('trackNote').textContent='启动失败：'+e;}});
}}
async function poll(){{
  if(!started) return;
  try{{ const s=await jget('/api/live?code='+CODE); if(s.started) render(s); }}catch(e){{}}
}}

/* ---------- 渲染 ---------- */
function minIdx(t){{ const [h,m]=t.split(':').map(Number); const v=h*60+m;
  if(v>=570&&v<=690) return v-570; if(v>=780&&v<=900) return 120+v-780; return null; }}
function render(s){{
  // 大模型这一路的真实状态：未配置 / 等行情 / 调用中 / 失败原因，都直接摆出来，
  // 不再用一句"未启用"糊过去（失败时用户根本不知道是缺 Key、超时还是返回解析不了）
  (function(){{
    const lm=s.llm||{{}}; const st=lm.status||{{}};
    const el=document.getElementById('llmLiveStatus');
    if(!el) return;
    const map={{idle:'待命',waiting:'等行情',running:'调用中',ok:'已启用',error:'调用失败',
               disabled:'未启用',not_ready:'配置不完整'}};
    const ok=lm.result&&lm.result.ok;
    const last=ok?` · 最近一次 ${{fmt(lm.result.bps,1)}}bps（${{Math.round(lm.age_sec||0)}}s 前）`:'';
    el.textContent = st.note ? `${{map[st.state]||st.state||''}}：${{st.note}}${{last}}`
                             : (last ? '已启用'+last : '');
    el.style.color = (st.state==='error'||st.state==='not_ready') ? '#b45309' : '';
  }})();
  document.getElementById('sname').textContent = s.name||CODE;
  document.getElementById('modeBadge').textContent =
    s.mode==='live'?'● 实时模式':(s.mode==='replay'?'▶ 回放模式（最近交易日）':(s.mode==='replayed'?'✔ 回放完成':'○ 已收盘'));
  if(s.quote&&s.quote.price){{
    const q=s.quote; const el=document.getElementById('price');
    el.textContent=fmt(q.price); el.className='price '+cls(q.pct);
    document.getElementById('pct').textContent=`${{q.pct>=0?'+':''}}${{fmt(q.pct)}}%`;
    document.getElementById('pct').className='muted '+cls(q.pct);
  }}
  document.getElementById('updAt').textContent=(s.updated_at||'')+(s.error?(' ⚠ '+s.error.slice(0,80)):'');
  const p=(s.ensemble&&s.ensemble.primary)||{{}};
  if(p.pred_bps!==undefined){{
    document.getElementById('predMain').innerHTML =
      `<div class="price ${{cls(p.pred_bps)}}">${{fmt(p.pred_price)}}</div>
       <div class="muted">预测 ${{p.pred_bps>=0?'+':''}}${{fmt(p.pred_bps,1)}} bps ·
       区间 [${{fmt(p.lo)}} , ${{fmt(p.hi)}}] · σ=${{fmt(p.sigma_bps,1)}}bps</div>
       <div class="muted">目标时刻 ${{p.target_ts||'-'}} · 置信 ${{fmt((p.conf||0)*100,0)}}%
       ${{p.no_skill?' · <b style="color:#92400e">幅度不足，判震荡</b>':''}}</div>`;
    document.getElementById('predDir').innerHTML =
      `<div style="font-size:22px;font-weight:700" class="${{cls(p.pred_bps)}}">${{p.direction||'-'}}</div>
       <div class="muted">专家分歧 σ=${{fmt(p.dispersion_bps,1)}}bps ·
       收缩后有效幅度 ${{fmt(p.pred_bps,1)}}bps</div>
       <div class="muted">在线岭回归：${{s.ml_note||'-'}}</div>`;
  }}
  const f=s.features||{{}};
  document.getElementById('micro').innerHTML =
    `<div class="kv"><span>VWAP 偏离</span><b>${{fmt(f.vwap_dev,1)}} bps</b></div>
     <div class="kv"><span>1分钟波动</span><b>${{fmt(f.sigma_min,2)}} bps</b></div>
     <div class="kv"><span>五档委比</span><b>${{fmt(f.weibi,1)}}%</b></div>
     <div class="kv"><span>主动买卖差</span><b>${{fmt(f.ofi,2)}}</b></div>
     <div class="kv"><span>日内位置</span><b>${{fmt((f.range_pos||0)*100,0)}}%</b></div>
     <div class="kv"><span>量能(5/30)</span><b>${{fmt(f.vol_ratio,2)}}</b></div>`;
  const st=s.stats||{{}}; let stHtml='';
  if(st.by_h) for(const [h,v] of Object.entries(st.by_h)){{
    stHtml+=`<div class="kv"><span>${{h}} 分钟（n=${{v.n}}）</span>
      <b>MAE ${{fmt(v.mae_bps,1)}}bps · 命中 ${{v.hit===null?'-':fmt(v.hit*100,1)+'%'}} ·
      区间覆盖 ${{fmt((v.cover||0)*100,0)}}%</b></div>`;
  }}
  stHtml+=`<div class="muted">已结算 ${{s.settle_count||0}} 笔；bias（系统性偏差）
    ${{fmt(st.bias_bps,1)}}bps${{Math.abs(st.bias_bps||0)>8?' · 偏差偏大，谨慎':''}}</div>`;
  document.getElementById('statsBox').innerHTML = stHtml||'尚无结算样本';

  // 专家表
  let th='<tr><th>专家</th><th>预测bps</th><th>权重</th><th>有效占比</th><th>命中</th><th>MAE</th><th>说明</th></tr>';
  const exps=(s.ensemble&&s.ensemble.primary&&s.ensemble.primary.experts)||[];
  const ecEl=document.getElementById('expCount');
  if(ecEl) ecEl.textContent = exps.length?`共 ${{exps.length}} 路` : '';
  document.getElementById('expTbl').innerHTML = th + exps.map(e=>
    `<tr style="${{e.available?'':'opacity:.45'}}">
     <td>${{e.cn}}</td><td class="${{cls(e.pred_bps)}}">${{e.available?fmt(e.pred_bps,1):'-'}}</td>
     <td>${{fmt(e.weight*100,1)}}%</td><td>${{e.available?fmt((e.eff_weight||0)*100,1)+'%':'-'}}</td>
     <td>${{e.n?fmt(e.hit*100,1)+'% ('+e.n+')':'-'}}</td>
     <td>${{e.n?fmt(e.mae_bps,1):'-'}}</td><td class="muted" style="text-align:left;white-space:normal">${{e.note||''}}</td></tr>`).join('');

  // 新闻 / 政策 / 舆论面
  const nw=s.news||{{}};
  let nhtml='';
  if(s.mode==='replay'){{
    nhtml += '<div class="muted" style="margin-bottom:6px">⚠️ 回放模式：新闻是当前时点的，'+
             '与历史分钟线时间错配，<b>仅展示、不参与加权</b>（避免污染误差统计）。</div>';
  }}
  if(!nw.ok){{
    nhtml+=`<div class="muted">${{nw.note||'尚未抓取（盘中每 3 分钟在后台刷新，不影响价格预测）'}}</div>`;
  }} else {{
    const ncls = nw.decayed_score>0.05?'up':(nw.decayed_score<-0.05?'dn':'');
    nhtml+= `<div class="kv"><span>方向</span><b class="${{ncls}}">${{nw.stance}} ·
      衰减后分数 ${{fmt(nw.decayed_score,2)}}（原始 ${{fmt(nw.score,2)}}）</b></div>
      <div class="kv"><span>情绪 / 政策</span><b>${{fmt(nw.senti,2)}} / ${{fmt(nw.policy,2)}} ·
      置信 ${{fmt((nw.conf||0)*100,0)}}%</b></div>
      <div class="kv"><span>新鲜度</span><b>${{fmt((nw.age_sec||0)/60,0)}} 分钟前 ·
      衰减系数 ${{fmt(nw.decay,2)}} · ${{nw.news_count||0}} 条素材</b></div>
      <div class="kv"><span>来源</span><b>${{nw.has_llm?('大模型 '+(nw.model||'')):'词典情绪（未配 Key，颗粒度较粗）'}}</b></div>`;
    const hs=nw.headlines||[];
    if(hs.length) nhtml += `<div class="muted" style="margin-top:6px">关键驱动：`+
      hs.map(h=>`<div>· ${{h}}</div>`).join('')+`</div>`;
    const rk=nw.risks||[];
    if(rk.length) nhtml += `<div class="muted">风险：${{rk.join('；')}}</div>`;
    if(nw.note) nhtml += `<div class="muted">备注：${{nw.note}}</div>`;
  }}
  document.getElementById('newsBox').innerHTML = nhtml;

  // 拐点
  const t=s.turning; if(t){{
    const ex=t.extremes||{{}};
    let html='';
    if(ex.high) html+=`<div class="kv"><span>今日高点预测</span><b>${{fmt(ex.high.price)}}（区间 ${{fmt(ex.high.lo)}}~${{fmt(ex.high.hi)}}）
      · 时间 ${{ex.high.window}} · 已出现概率 ${{fmt(ex.high.prob_done*100,0)}}%</b></div>`;
    if(ex.low) html+=`<div class="kv"><span>今日低点预测</span><b>${{fmt(ex.low.price)}}（区间 ${{fmt(ex.low.lo)}}~${{fmt(ex.low.hi)}}）
      · 时间 ${{ex.low.window}} · 已出现概率 ${{fmt(ex.low.prob_done*100,0)}}%</b></div>`;
    if(t.next) html+=`<div class="kv"><span>下一个拐点</span><b class="${{t.next.kind==='顶'?'dn':'up'}}">${{t.next.kind}} ·
      约 ${{t.next.in_minutes}} 分钟后（≈${{t.next.eta}}）· ${{fmt(t.next.price)}} · 置信 ${{fmt(t.next.conf*100,0)}}%</b></div>`;
    html+=`<div class="muted">日内 RSI ${{t.rsi}} · 时间分布来源：${{t.profile_source||'-'}}</div>`;
    const det=(t.detected||[]).slice(-5).reverse();
    if(det.length) html+=`<div class="muted" style="margin-top:6px">已确认拐点：`+
      det.map(d=>`${{d.kind}} ${{d.ts.slice(11,16)}} @${{fmt(d.price)}}`).join(' · ')+`</div>`;
    document.getElementById('turnBox').innerHTML=html;
  }}

  drawChart(s);
  loadDaily();
}}

/* ---------- 分时图 ---------- */
function drawChart(s){{
  const cv=document.getElementById('chart'), dpr=devicePixelRatio||1;
  const W=cv.clientWidth, H=330; cv.width=W*dpr; cv.height=H*dpr;
  const ctx=cv.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,W,H);
  const bars=s.bars||[], ov=(s.overlay||[]), f=s.features||{{}};
  if(!bars.length){{ ctx.fillStyle='#9ca3af'; ctx.fillText('等待行情…',20,40); return; }}
  const xs=t=>{{ const i=minIdx(t); return i===null?null:18+(W-26)*i/240; }};
  let lo=Infinity, hi=-Infinity;
  bars.forEach(b=>{{lo=Math.min(lo,b.l);hi=Math.max(hi,b.h)}});
  (ov||[]).forEach(o=>{{ if(o.lo)lo=Math.min(lo,o.lo); if(o.hi)hi=Math.max(hi,o.hi);
    if(o.realized)lo=Math.min(lo,o.realized),hi=Math.max(hi,o.realized);}});
  if(f.day_high)hi=Math.max(hi,f.day_high); if(f.day_low)lo=Math.min(lo,f.day_low);
  const pad=(hi-lo)*0.06||1; lo-=pad; hi+=pad;
  const ys=v=>H-24-(H-44)*(v-lo)/(hi-lo);
  // 网格
  ctx.strokeStyle='#f1f5f9'; ctx.fillStyle='#9ca3af'; ctx.font='11px sans-serif';
  for(let i=0;i<=4;i++){{ const v=lo+(hi-lo)*i/4, y=ys(v);
    ctx.beginPath();ctx.moveTo(18,y);ctx.lineTo(W-8,y);ctx.stroke();
    ctx.fillText(v.toFixed(2), W-46, y-3); }}
  ['09:30','10:30','11:30/13:00','14:00','15:00'].forEach((t,i)=>{{
    const x=xs(t==='11:30/13:00'?'11:30':t); if(x)ctx.fillText(t, x-12, H-8);}});
  // 预测区间带
  const ovP=ov.filter(o=>o.pred);
  if(ovP.length>1){{
    ctx.beginPath();
    ovP.forEach((o,i)=>{{const x=xs(o.t);if(x==null)return; const y=ys(o.hi); i?ctx.lineTo(x,y):ctx.moveTo(x,y);}});
    for(let i=ovP.length-1;i>=0;i--){{const o=ovP[i];const x=xs(o.t);if(x==null)continue;ctx.lineTo(x,ys(o.lo));}}
    ctx.closePath(); ctx.fillStyle='rgba(148,163,184,.18)'; ctx.fill();
  }}
  // 真实价
  ctx.beginPath(); ctx.strokeStyle='#2563eb'; ctx.lineWidth=1.6;
  bars.forEach((b,i)=>{{const x=xs(b.t),y=ys(b.c); if(x==null)return; i?ctx.lineTo(x,y):ctx.moveTo(x,y);}});
  ctx.stroke();
  // VWAP
  if(f.vwap){{
    ctx.setLineDash([4,4]); ctx.strokeStyle='#94a3b8'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(18,ys(f.vwap)); ctx.lineTo(W-8,ys(f.vwap)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle='#64748b'; ctx.fillText('VWAP '+f.vwap.toFixed(2), 22, ys(f.vwap)-4);
  }}
  // 预测线（虚线）
  if(ovP.length>1){{
    ctx.beginPath(); ctx.strokeStyle='#ea580c'; ctx.setLineDash([5,4]); ctx.lineWidth=1.4;
    ovP.forEach((o,i)=>{{const x=xs(o.t),y=ys(o.pred); if(x==null)return; i?ctx.lineTo(x,y):ctx.moveTo(x,y);}});
    ctx.stroke(); ctx.setLineDash([]);
  }}
  // 已实现点
  ctx.fillStyle='#0f766e';
  ov.forEach(o=>{{ if(!o.realized)return; const x=xs(o.t); if(x==null)return;
    ctx.beginPath(); ctx.arc(x,ys(o.realized),2.2,0,7); ctx.fill(); }});
  // 拐点标记
  const det=(s.turning&&s.turning.detected)||[];
  det.forEach(d=>{{ const x=xs(d.ts.slice(11,16)); if(x==null)return;
    ctx.fillStyle=d.kind==='顶'?'#dc2626':'#16a34a';
    ctx.beginPath(); const y=ys(d.price)+(d.kind==='顶'?-7:7);
    ctx.moveTo(x,y); ctx.lineTo(x-4,y+8); ctx.lineTo(x+4,y+8); ctx.closePath(); ctx.fill(); }});
  // 现价
  if(s.quote&&s.quote.price){{
    const y=ys(s.quote.price);
    ctx.fillStyle='#111827'; ctx.beginPath(); ctx.arc(W-10,y,3,0,7); ctx.fill();
  }}
}}

/* ---------- 日对比 ---------- */
let dayLoaded=false;
async function loadDaily(){{
  if(dayLoaded) return; dayLoaded=true;
  try{{
    const d=await jget('/api/daily?code='+CODE+'&days=60');
    const closeBy={{}}; (d.bars||[]).forEach(b=>closeBy[b.date.slice(0,10)]=b.close);
    const rows=(d.compare||[]).filter(c=>c.pred_close);
    document.getElementById('dayTbl').innerHTML =
      '<tr><th>基准日</th><th>预测(1日)</th><th>真实</th><th>误差%</th><th>方向</th></tr>'+
      rows.slice(-14).reverse().map(c=>`<tr>
        <td>${{c.base_date}}</td><td>${{fmt(c.pred_close)}}</td>
        <td>${{fmt(c.actual_close)}}</td>
        <td class="${{cls(-(c.err_pct||0))}}">${{c.err_pct===null?'待回填':fmt(c.err_pct)}}</td>
        <td>${{c.direction||'-'}}</td></tr>`).join('');
    document.getElementById('dayStats').textContent =
      `已回填 ${{d.stats.n}} 天 · 1日预测平均误差 ${{d.stats.mae_pct===null?'-':d.stats.mae_pct+'%'}} ·
       方向命中 ${{d.stats.hit===null?'-':fmt(d.stats.hit*100,0)+'%'}}`;
    // 画图：预测 vs 真实
    const cv=document.getElementById('dayChart'), dpr=devicePixelRatio||1;
    const W=cv.clientWidth,H=220; cv.width=W*dpr; cv.height=H*dpr;
    const ctx=cv.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,W,H);
    const pts=rows.filter(c=>c.actual_close).slice(-30);
    if(!pts.length){{ctx.fillStyle='#9ca3af';ctx.fillText('暂无已回填样本（预测到期后自动结算）',20,40);return;}}
    let lo=Infinity,hi=-Infinity;
    pts.forEach(c=>{{lo=Math.min(lo,c.actual_close,c.pred_close);hi=Math.max(hi,c.actual_close,c.pred_close)}});
    const pad=(hi-lo)*0.08||1;lo-=pad;hi+=pad;
    const xs=i=>24+(W-40)*i/Math.max(1,pts.length-1), ys=v=>H-20-(H-36)*(v-lo)/(hi-lo);
    ctx.strokeStyle='#f1f5f9'; for(let i=0;i<=3;i++){{const y=ys(lo+(hi-lo)*i/3);
      ctx.beginPath();ctx.moveTo(24,y);ctx.lineTo(W-12,y);ctx.stroke();}}
    ctx.beginPath();ctx.strokeStyle='#dc2626';ctx.lineWidth=1.6;
    pts.forEach((c,i)=>{{i?ctx.lineTo(xs(i),ys(c.actual_close)):ctx.moveTo(xs(i),ys(c.actual_close))}});ctx.stroke();
    ctx.beginPath();ctx.strokeStyle='#ea580c';ctx.setLineDash([5,4]);
    pts.forEach((c,i)=>{{i?ctx.lineTo(xs(i),ys(c.pred_close)):ctx.moveTo(xs(i),ys(c.pred_close))}});ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle='#6b7280';ctx.font='11px sans-serif';
    ctx.fillText('红=真实收盘',30,14);ctx.fillStyle='#ea580c';ctx.fillText('橙虚=预测',110,14);
  }}catch(e){{dayLoaded=false;}}
}}
loadCfg();
if(new URLSearchParams(location.search).get('auto')==='1') startTrack();
</script></body></html>"""


def _page_home(q: dict) -> str:
    """入口页：实时跟踪 / 涨幅榜 / 数据库概览 / 日线同步。"""
    info = {}
    try:
        info = db.db_info()
    except Exception:
        info = {}
    counts = info.get("counts") or {}
    rows = "".join(f"<tr><td>{k}</td><td style='text-align:right'>{v}</td></tr>"
                   for k, v in counts.items() if v)
    from .realtime import is_open
    market_txt = ("<span style='color:#dc2626'>盘中</span>" if is_open()
                  else "<span class='muted'>已收盘（实时页将用最近交易日分钟线回放）</span>")
    return """<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>股票实时预测 · 本机服务</title><style>""" + _CSS + """</style></head><body>
<div class="wrap">
<h1>股票实时预测 <span class="badge">本机已启动</span></h1>
<div class="card">
  <h2>市场状态</h2>
  <div>""" + market_txt + """</div>
  <div class="muted" style="margin-top:6px">数据库：""" + str(info.get("path", "")) + """</div>
</div>
<div class="card">
  <h2>入口</h2>
  <div style="display:flex;gap:10px;flex-wrap:wrap">
    <a class="tag" href="/classic" style="text-decoration:none">📊 股票深度分析（形态/量能/新闻政策/拐点）</a>
    <a class="tag" href="/rank" style="text-decoration:none">📈 涨幅榜（东财 · A股/科创板 前 50）</a>
    <a class="tag" href="/live?code=600519" style="text-decoration:none">⏱ 实时跟踪（含大模型 Key 设置）</a>
    <a class="tag" href="/api/db" style="text-decoration:none">🗄 数据库概览（JSON）</a>
  </div>
  <div class="muted" style="margin-top:8px"><b>想跑分析</b>：点「股票深度分析」→ 输入 6 位代码 →
  在页面下方「大模型设置」里填<b>你自己的 Key</b> → 提交。一次完整分析 3~6 分钟（训练多模型），
  人多时会排队。<b>不填 Key 也能跑</b>：量化预测照常，新闻/政策那段改走本地词典情绪。
  <br><b>想看盘中</b>：涨幅榜点任意一行 → 进入实时跟踪页（3 秒刷新、1/3/5 分钟预测）。</div>
</div>
<div class="card">
  <h2>已入库数据</h2>
  <table>""" + (rows or "<tr><td class='muted'>暂无数据，去榜单页挑一只开始跟踪</td></tr>") + """</table>
</div>
</div></body></html>"""


def _page_rank(q: dict) -> str:
    return """<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>涨幅榜 · 东方财富</title><style>""" + _CSS + """</style></head><body>
<div class="wrap">
<div class="nav"><a href="/">← 首页</a><a href="/live?code=600519">实时跟踪</a></div>
<h1>涨幅榜 <span class="badge">数据源：东方财富</span>
<span class="muted" id="meta"></span></h1>
<div id="staleBar" style="display:none;margin:8px 0;padding:9px 12px;border-radius:8px;
  background:#fff7ed;border:1px solid #fed7aa;color:#9a3412;font-size:13px"></div>
<div class="card">
  <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
    <div class="tag" id="tabA" style="cursor:pointer" onclick="pick('a')">A股（沪深全市场）</div>
    <div class="tag" id="tabK" style="cursor:pointer" onclick="pick('kcb')">科创板</div>
    <span class="muted">点击任意一行 → 进入实时跟踪页（可输入大模型 Key 后开启跟踪）</span>
    <span style="flex:1"></span>
    <span class="muted" id="timerHint">每 30 秒刷新</span>
    <label class="muted"><input type="checkbox" id="autoChk" checked> 自动刷新</label>
  </div>
  <div style="max-height:64vh;overflow:auto;margin-top:8px">
  <table id="tbl"></table></div>
  <div class="muted" id="note"></div>
</div>
<div class="card"><h2>榜单历史快照（每分钟一份，同分钟覆盖）</h2>
  <div style="max-height:200px;overflow:auto"><table id="histTbl"></table></div></div>
</div>
<script>""" + _BASE_JS + """
let market='a', timer=null, intervalSec=30;
function pick(m){
  market=m;
  document.getElementById('tabA').style.background = m==='a'?'#eef2ff':'#f3f4f6';
  document.getElementById('tabK').style.background = m==='kcb'?'#eef2ff':'#f3f4f6';
  load();
}
function go(code){
  location.href='/live?code='+code;
}
function showStale(d){
  const bar=document.getElementById('staleBar');
  if(d.stale){
    bar.style.display='block';
    bar.innerHTML = '<b>已收盘 · 榜单已冻结</b>　'+ (d.note||'')
      + '<br><span style="opacity:.85">最新一笔行情 '
      + (d.quote_ts||'15:00:00') + '（距现在 '
      + (d.quote_age_sec!=null? Math.round(d.quote_age_sec/60)+' 分钟':'—')
      + '）。<b>数据本身没变，不是页面卡住</b>；下一交易日 09:30 开盘后自动恢复每 30 秒刷新。</span>';
  } else {
    bar.style.display='none';
  }
}
async function load(){
  try{
    const d=await jget('/api/rank?market='+market+'&top=50');
    document.getElementById('meta').textContent =
      `${d.label} · 共 ${d.total||'-'} 只 · 抓取 ${(d.ts||'').slice(11)}`
      + (d.quote_ts? ` · 行情 ${d.quote_ts}`:'')
      + (d.cached?'（30 秒内缓存）':'');
    document.getElementById('note').textContent = d.note||'';
    showStale(d);
    // 收盘后没必要 30 秒空刷一次：自动降到 120 秒；开盘后自动恢复
    const want = d.stale ? 120 : 30;
    if(want !== intervalSec){ intervalSec = want; setTimer(); }
    document.getElementById('timerHint').textContent =
      `每 ${intervalSec} 秒刷新`;
    const head='<tr><th>#</th><th>代码</th><th>名称</th><th>现价</th><th>涨幅</th><th>涨速</th><th>换手</th><th>量比</th><th>成交额(亿)</th><th>主力净流入(万)</th><th>总市值(亿)</th></tr>';
    document.getElementById('tbl').innerHTML = head + (d.rows||[]).map(r=>
      `<tr class="click" onclick="go('${r.code}')">
       <td>${r.rank}</td><td>${r.code}</td>
       <td>${r.name}${r.near_limit?' <span class="limit">涨停</span>':''}</td>
       <td>${fmt(r.price)}</td>
       <td class="up">+${fmt(r.pct)}%</td>
       <td class="${cls(r.speed)}">${fmt(r.speed)}</td>
       <td>${fmt(r.turnover)}%</td><td>${fmt(r.liangbi)}</td>
       <td>${fmt(r.amount_yi)}</td>
       <td class="${cls(r.main_net_wan)}">${fmt(r.main_net_wan,0)}</td>
       <td>${fmt(r.mktcap_yi,0)}</td></tr>`).join('');
    const h=await jget('/api/rank/history?market='+market+'&limit=24');
    document.getElementById('histTbl').innerHTML =
      '<tr><th>时间</th><th>榜首</th><th>涨幅</th><th>前5</th></tr>'+
      (h.rows||[]).reverse().map(s=>{
        const top=(s.rows||[]).slice(0,5);
        return `<tr><td>${s.ts}</td>
        <td>${top[0]?top[0].code+' '+top[0].name:'-'}</td>
        <td class="up">${top[0]?'+'+fmt(top[0].pct)+'%':'-'}</td>
        <td class="muted" style="white-space:normal;text-align:left">${top.slice(1).map(r=>r.code+' '+r.name+' +'+fmt(r.pct)+'%').join('，')}</td></tr>`;
      }).join('');
  }catch(e){ document.getElementById('note').textContent='加载失败：'+e; }
}
function setTimer(){ if(timer)clearInterval(timer);
  if(document.getElementById('autoChk').checked) timer=setInterval(load, intervalSec*1000); }
document.getElementById('autoChk').onchange=setTimer;
pick('a'); setTimer();
</script></body></html>"""
