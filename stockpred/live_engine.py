"""实时预测引擎：行情刷新 → 特征 → 十四路专家 → 加权 → 落库 → 到期结算。

一次 tick 的完整链路（约 <30ms，不含网络）
------------------------------------------
    quote(3s) → TickBuffer → features → Ensemble.predict → 展示/落库
    每-bars_refresh 秒拉一次腾讯 1 分钟线 → upsert intraday_bars
    每-turn_interval 秒重算日内拐点（极值概率 + 下一拐点）
    独立线程每-llm_interval 秒问一次大模型（不阻塞 tick 线程）
    到期预测 → 用真实价结算 → Hedge 更新各专家权重

两种运行模式
------------
- live   ：交易时段，跟新浪 hq 快照同频（默认 3 秒）。
- replay ：非交易时段，取最近一个交易日的 1 分钟线逐条推进（默认 0.35s/根），
  页面会明确标注"回放"。这让结算、权重更新、命中率统计在收盘后也能真实运转，
  而不是摆一组不会变的数字。

落库频率控制：预测每 3 秒都算（页面上永远是最新的），但**每 persist_every 秒
才写一行库**（结算/权重更新/统计都基于库里这些样本），避免 SQLite 每天写几十万行。
"""
from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime

import config
from . import db, llm_live, news_live, turning_live
from .experts import Ensemble, OnlineML, build_ctx
from .realtime import (Quote, TickBuffer, add_trading_minutes, compute_features,
                       fetch_minute_bars, fetch_quote, is_open, now, session_state)

_CFG = getattr(config, "LIVE", {}) or {}


def _ts_to_epoch(ts: str) -> float:
    try:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").timestamp()
    except Exception:
        return 0.0


def _cfg(k, d):
    return _CFG.get(k, d)


class LiveEngine:
    def __init__(self, symbol: str, name: str = ""):
        from .realtime import normalize
        self.symbol = normalize(symbol)
        self.name = name
        self.mode = "live" if is_open() else "replay"
        self.bars: list[dict] = []            # 当前会话 1 分钟线
        self._bar_idx: dict[str, dict] = {}   # ts → bar
        self.ticks = TickBuffer(900)
        self.ens = Ensemble(self.symbol)
        self.ml = OnlineML(self.symbol, self.ens.horizons)
        self.season: dict = {}
        self.season_ts = 0.0
        self.llm_res: dict = {}
        self.llm_last: dict = {}          # 最近一次尝试（含失败原因），页面要能看到
        self.llm_status: dict = {"state": "idle", "note": "尚未调用", "ts": 0.0}
        self._daily_cache: dict | None = None
        self._daily_ts = 0.0
        self.last_bars_ts = 0.0
        self.last_store: dict[int, float] = {}
        self.last_turn_ts = 0.0
        self.last_predict_ts = 0.0
        self.persist_every = 10.0
        self.settle_count = 0
        self.error: str = ""
        self.replay_done = False
        self.last_ens: dict = {}
        self.last_ctx: dict = {}
        self.last_feats: dict = {}
        self.turning: dict | None = None
        self._stop = threading.Event()
        self._started = False
        # 回放时用游标分钟作为"现在"，否则预测目标时间会落到真实墙钟的下一分钟，
        # 结算拿错价 → 误差虚高。live 模式保持 None。
        self.clock = None
        self._pred_gate_n = 0
        self._pred_seq = 0
        self._stats_ts = 0.0
        self._stats_emp = {}
        self._stats_n = 0
        self._stats_err = ""
        self._band_calib_n = 0
        self._lock = threading.RLock()
        self.thread = threading.Thread(target=self._run, name=f"live-{self.symbol}", daemon=True)
        self.state: dict = {}
        self._last_quote: Quote = Quote(symbol=self.symbol)

    # ----------------------------- 生命周期 -----------------------------
    def start(self):
        if not self._started:
            self._started = True
            self.thread.start()
        return self

    def stop(self):
        self._stop.set()

    # ----------------------------- 主循环 -----------------------------
    def _run(self):
        self._resolve_name()
        if self.mode == "replay":
            self._run_replay()
        else:
            self._run_live()

    def _resolve_name(self):
        """补全股票名（回放/收盘状态下没有实时快照，只能单独取一次）。"""
        if self.name and self.name != self.symbol:
            return
        try:
            q = fetch_quote(self.symbol)
            if q.name:
                self.name = q.name
                self._last_quote = q
                return
        except Exception:
            pass
        try:
            from . import db as _db
            hist = _db.daily_history(self.symbol, limit=1)
            if hist and hist[0].get("name"):
                self.name = hist[0]["name"]
        except Exception:
            pass

    def _clock_now(self):
        """回放模式返回游标分钟，实盘模式返回真实时间。"""
        return self.clock or now()

    def _run_live(self):
        self.clock = None          # 切回实盘必须清掉回放游标
        poll = float(_cfg("poll_interval", 3.0))
        while not self._stop.is_set():
            t0 = time.time()
            try:
                self._tick_live()
                self.error = ""
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
            self._stop.wait(max(0.4, poll - (time.time() - t0)))
            # 交易时段切换检测：收盘后自动停
            if not is_open() and now().hour >= 15 and now().minute >= 5 and now().weekday() < 5:
                self.mode = "closed"
                break

    def _tick_live(self):
        q = fetch_quote(self.symbol)
        if q.price > 0:
            self._last_quote = q
            self.ticks.add(time.time(), q.price, q.volume)
        self._maybe_refresh_bars()
        self._maybe_refresh_season()
        feats = compute_features(self.bars, self._last_quote, self.ticks, self.season)
        if feats.get("valid"):
            self._maybe_predict(feats)
        self._settle()
        self._maybe_turning(feats)
        self._build_state(feats)

    # ----------------------------- 回放 -----------------------------
    def _run_replay(self):
        speed = float(_cfg("replay_speed", 0.35))
        bars, trade_date = fetch_minute_bars(self.symbol)
        if not bars:
            self.error = "非交易时段且取不到最近交易日的分钟线（回放不可用）"
            self._build_state({})
            return
        db.upsert_intraday_bars(self.symbol, bars)   # 回放的分钟线也入库，供季节性/岭回归积累
        self.mode = "replay"
        with self._lock:
            self.bars = []
            self._bar_idx = {}
            self.ticks = TickBuffer(900)
            self.ml.maybe_fit(force=True)
        self.replay_done = False
        n = len(bars)
        for i, b in enumerate(bars):
            if self._stop.is_set():
                return
            with self._lock:
                self.bars.append(b)
                self._bar_idx[b["ts"]] = b
                self.clock = datetime.strptime(b["ts"], "%Y-%m-%d %H:%M:%S")
                # 回放时把"当前报价"也换成游标那根 K 线，页面顶栏才不会显示收盘后的真实价
                self._last_quote = Quote(
                    symbol=self.symbol, name=self.name or self.symbol, ts=b["ts"],
                    price=float(b["close"]), prev_close=float(self._last_quote.prev_close or 0),
                    open=float(b["open"]), high=float(b["high"]), low=float(b["low"]),
                    volume=float(b.get("volume") or 0))
                # 用开盘/收盘两个合成 tick 维持 OFI/tick 斜率专家可用
                cum = float(b.get("volume") or 0)
                if i > 0:
                    prev_cum = float(bars[i - 1].get("volume") or 0)
                else:
                    prev_cum = 0.0
                self.ticks.add(_ts_to_epoch(b["ts"]) - 15, float(b["open"]), prev_cum + cum * 0.4)
                self.ticks.add(_ts_to_epoch(b["ts"]), float(b["close"]), cum)
            feats = compute_features(self.bars, None, self.ticks, self.season)
            if feats.get("valid"):
                self._maybe_predict(feats)
                if i % 15 == 0:
                    self._maybe_turning(feats)
                self._settle()
                self._build_state(feats)
            self._stop.wait(speed)
        self.replay_done = True
        self.mode = "replayed"
        self._settle()
        self.ens.save()
        self._build_state(compute_features(self.bars, None, self.ticks, self.season))

    # ----------------------------- 数据刷新 -----------------------------
    def _maybe_refresh_bars(self):
        if time.time() - self.last_bars_ts < float(_cfg("bars_refresh", 20.0)):
            return
        self.last_bars_ts = time.time()
        try:
            bars, trade_date = fetch_minute_bars(self.symbol)
            if bars:
                with self._lock:
                    self.bars = bars
                    self._bar_idx = {b["ts"]: b for b in bars}
                db.upsert_intraday_bars(self.symbol, bars)
        except Exception as e:
            self.error = f"分钟线刷新失败：{e}"

    def _maybe_refresh_season(self):
        if time.time() - self.season_ts < 600:
            return
        self.season_ts = time.time()
        try:
            self.season = db.minute_seasonality(self.symbol, days=12)
        except Exception:
            self.season = {}

    # ----------------------------- 预测 -----------------------------
    def _news_face(self, feats: dict) -> dict:
        """新闻/政策面：**永不阻塞**。

        缓存新鲜就直接用；过期就在后台起线程去抓，本次预测继续用手上这份（可能为空）。
        新闻抓取要 3~10 秒，绝不能挡在 3 秒一轮的行情链路前面。
        """
        try:
            closes = [float(b.get("close") or 0) for b in self.bars]
            ret5 = ret20 = 0.0
            if len(closes) >= 6 and closes[-6] > 0:
                ret5 = (closes[-1] / closes[-6] - 1) * 1e4
            if len(closes) >= 21 and closes[-21] > 0:
                ret20 = (closes[-1] / closes[-21] - 1) * 1e4
            return news_live.refresh_async(
                self.symbol, name=self.name,
                price=float(feats.get("price") or 0),
                ret5=round(ret5, 2), ret20=round(ret20, 2),
                vol_expand=float(feats.get("vol_ratio") or 1.0),
            ) or {}
        except Exception:
            return {}

    def _news_view(self) -> dict:
        """页面读的新闻面：只读缓存，绝不触发抓取。"""
        try:
            cur = news_live._CACHE.get(self.symbol) or {}
            if not cur:
                return {"ok": False, "note": "尚未抓取（盘中每分钟后台刷新一次）",
                        "status": news_live.status()}
            out = {k: v for k, v in cur.items() if not k.startswith("_")}
            age = max(0.0, time.time() - float(out.get("ts") or time.time()))
            hl = float(out.get("half_life_sec") or 1800)
            import math as _m
            out["age_sec"] = round(age, 1)
            out["decay"] = round(_m.exp(-age / max(60.0, hl)), 4)
            out["decayed_score"] = round(float(out.get("score") or 0) * out["decay"], 4)
            if self.mode == "replay":
                out["note"] = ((out.get("note") or "") +
                               "（回放模式：新闻是当前时点的，与历史分钟线时间错配，"
                               "仅展示、不参与加权）").strip()
            out["status"] = news_live.status()
            return out
        except Exception as e:
            return {"ok": False, "note": f"新闻面读取失败：{e}"}

    # ----------------------------- 日线模型（慢变量） -----------------------------
    def _daily_model(self) -> dict:
        """读 daily_signals 里已落库的日线模型方向，10 分钟缓存。

        **不在 tick 线程里跑训练**：日线 RF+GBDT+LSTM 要几十秒，
        放进 3 秒循环会直接卡死行情刷新。只读现成结论，没有就诚实地报不可用。
        """
        now = time.time()
        if self._daily_cache is not None and now - self._daily_ts < 600:
            return self._daily_cache
        self._daily_ts = now
        out = {"ok": False, "note": "尚无日线结论（先用日线分析跑一次该股）"}
        try:
            # 日线侧落库用的是裸代码（600519），实时侧是带市场前缀的 sh600519。
            # 不兼容这两种写法就会永远读不到结论（model 专家一直灰着的真实原因）。
            rows = []
            for sym in self._daily_symbols():
                rows = db.daily_signal_history(sym, limit=5)
                if rows:
                    break
            # 只认真正带信号的记录：历史上 --sync-db 会写入 score=NULL 的空壳行，
            # 拿到空壳时要继续往前找，而不是直接判定"没有 model 信号"。
            picked = None
            for r in reversed(rows):
                sigs = json.loads(r.get("signals_json") or "[]") or []
                if sigs:
                    picked = (r, sigs)
                    break
            if picked:
                r, sigs = picked
                hit = None
                for s in sigs:
                    if str(s.get("key")) == "model":
                        hit = s
                        break
                if hit:
                    out = {"ok": True, "value": float(hit.get("value") or 0.0),
                           "date": str(r.get("date") or ""),
                           "note": str(hit.get("note") or "")}
                else:
                    out = {"ok": False,
                           "note": f"{r.get('date')} 的日线结论里没有 model 信号"}
        except Exception as e:
            out = {"ok": False, "note": f"读取日线信号失败：{type(e).__name__}"}
        self._daily_cache = out
        return out

    def _daily_symbols(self) -> list[str]:
        """日线信号表可能用裸代码或带前缀代码，两种都试。"""
        sym = str(self.symbol or "")
        bare = re.sub(r"^(sh|sz|bj)", "", sym, flags=re.I)
        out = [sym]
        if bare and bare != sym:
            out.append(bare)
        return out

    def _maybe_predict(self, feats: dict, force_store: bool = False):
        if self.mode == "replay":
            # 回放按"模拟分钟"节流，而不是墙钟：否则 3 秒墙钟 ≈ 9 根分钟线才预测一次
            gate = max(1, int(_cfg("replay_predict_every_bars", 1)))
            self._pred_gate_n += 1
            if self._pred_gate_n < gate:
                return
            self._pred_gate_n = 0
        elif time.time() - self.last_predict_ts < float(_cfg("predict_every", 3.0)):
            return
        self.last_predict_ts = time.time()
        if time.time() - self.ml.last_fit > 300 or not self.ml.fits:
            threading.Thread(target=self.ml.maybe_fit, daemon=True).start()
        nf = self._news_face(feats)
        # 回放模式：新闻是"现在"的，分钟线是历史某天的，两者时间错配。
        # 仍然抓取并展示（供参考），但不参与加权，否则会污染误差统计。
        nf_ctx = nf if self.mode != "replay" else {
            "ok": False,
            "note": "回放模式不参与加权（新闻面见下方卡片，仅展示）",
        }
        # 大模型那一路要能看到"为什么没有预测"：直接把它自己的状态当说明传下去，
        # 否则专家表永远只显示"未启用"，用户分不清是缺 Key、超时还是解析失败。
        llm_ctx = self.llm_res or {"ok": False, "note": self._llm_note()}
        self._maybe_turning(feats)
        ctx = build_ctx(feats, self.bars, self.season, llm_ctx, self.ml, news=nf_ctx,
                        turning=self.turning, daily_model=self._daily_model())
        res = self.ens.predict(ctx)
        # 用实测误差自校准区间：模型算的 sigma 常年偏乐观，80% 带实测只覆盖 60~70%。
        # 这里只放宽 lo/hi（不改预测值与专家权重），让区间覆盖率回到名义水平。
        self._calibrate_band(res, float(feats.get("price") or 0))
        tnow = self._clock_now()
        self._pred_seq += 1
        if self.mode == "replay":
            # 回放落库按"模拟分钟"节流：每 N 根 K 线写一组，避免一天写几百上千行
            every = max(1, int(_cfg("replay_persist_every_bars", 3)))
            store_now = force_store or (self._pred_seq % every == 1)
        for h, pr in res.get("horizons", {}).items():
            tgt = add_trading_minutes(tnow, int(h)).strftime("%Y-%m-%d %H:%M:%S")
            pr["target_ts"] = tgt
            if self.mode == "replay":
                do_store = store_now
            else:
                do_store = force_store or (time.time() - self.last_store.get(h, 0.0)
                                           >= self.persist_every)
            if do_store:
                self.last_store[h] = time.time()
                db.add_live_prediction(
                    self.symbol, tnow.strftime("%Y-%m-%d %H:%M:%S"), tgt, int(h),
                    float(feats.get("price") or 0), float(pr.get("pred_price") or 0),
                    float(pr.get("pred_bps") or 0), float(pr.get("lo") or 0),
                    float(pr.get("hi") or 0), str(pr.get("direction")), float(pr.get("conf") or 0),
                    float(pr.get("sigma_bps") or 0), pr.get("experts") or [], self.mode)
                pr["stored"] = True
        with self._lock:
            self.last_ens = res
            self.last_ctx = ctx
            self.last_feats = feats

    def _calibrate_band(self, res: dict, price: float):
        """用近期实测 |误差| 的 80 分位当区间半宽，让"80% 区间"名副其实。

        模型自己算的 sigma 常年偏乐观（实测覆盖只有 60%~70%）。这里不改预测值、
        不改专家权重，只把 lo/hi 换成经验分位；样本 <20 时保持模型原区间。
        """
        if price <= 0:
            return
        win = float(_cfg("stats_cache_sec", 15.0))
        if time.time() - self._stats_ts < win:
            emp = self._stats_emp
        else:
            self._stats_ts = time.time()
            emp = {}
            try:
                by_h = (db.live_stats(self.symbol) or {}).get("by_h") or {}
                for k, v in by_h.items():
                    if int(v.get("n") or 0) >= 20 and v.get("q80_bps"):
                        emp[int(k)] = float(v["q80_bps"])
                self._stats_n = sum(int(v.get("n") or 0) for v in by_h.values())
            except Exception as e:
                self._stats_err = f"{type(e).__name__}: {e}"
                emp = {}
            self._stats_emp = emp
        if not emp:
            return
        z = float(_cfg("band_z", 1.28))
        for h, pr in (res.get("horizons") or {}).items():
            q = emp.get(int(h))
            if not q:
                continue
            pred = float(pr.get("pred_bps") or 0)
            pr["sigma_bps"] = round(q / z, 2)      # 折算回 sigma 口径，供可靠性计算用
            pr["lo"] = round(price * (1 + (pred - q) / 1e4), 2)
            pr["hi"] = round(price * (1 + (pred + q) / 1e4), 2)
            pr["band_calibrated"] = True
            pr["band_half_bps"] = round(q, 2)
            self._band_calib_n += 1

    # ----------------------------- 结算 -----------------------------
    def _settle(self):
        tnow = self._clock_now()
        pend = db.pending_predictions(self.symbol, tnow.strftime("%Y-%m-%d %H:%M:%S"), limit=60)
        for r in pend:
            target = str(r.get("target_ts") or "")
            realized = self._price_at(target)
            if realized is None or not r.get("price"):
                continue
            realized_bps = (realized / float(r["price"]) - 1) * 1e4
            err = realized_bps - float(r.get("pred_bps") or 0)
            db.settle_prediction(int(r["id"]), realized, round(realized_bps, 3), round(err, 3))
            try:
                experts = json.loads(r.get("experts_json") or "[]")
            except Exception:
                experts = []
            self.ens.update(realized_bps, experts, float(r.get("sigma_bps") or 0))
            self.settle_count += 1
            if self.settle_count % 10 == 0:
                self.ens.save()

    def _price_at(self, target: str) -> float | None:
        b = self._bar_idx.get(target)
        if b:
            return float(b["close"])
        # 分钟线还没刷到：用 tick 缓冲兜底（±90 秒）
        epoch = _ts_to_epoch(target)
        if epoch:
            return self.ticks.price_near(epoch, tol=90.0)
        return None

    # ----------------------------- 拐点 -----------------------------
    def _maybe_turning(self, feats: dict):
        if self.mode != "replay" and time.time() - self.last_turn_ts < float(_cfg("turn_interval", 15.0)):
            return
        self.last_turn_ts = time.time()
        if not feats.get("valid"):
            return
        try:
            ens_pred = float(((self.last_ens or {}).get("primary") or {}).get("pred_bps") or 0)
            sig = float(feats.get("sigma_min") or 3.0)
            self.turning = turning_live.rebuild(
                self.symbol, self.bars, feats, session_state(self._clock_now()),
                pred_bps=ens_pred, sigma_min=sig, persist=(self.mode == "live"))
        except Exception as e:
            self.error = f"拐点计算失败：{e}"

    # ----------------------------- 大模型 -----------------------------
    def _llm_note(self) -> str:
        st = self.llm_status or {}
        return str(st.get("note") or "尚未调用（启动后约 2~20 秒出第一份结果）")

    def _set_llm_status(self, state: str, note: str):
        with self._lock:
            self.llm_status = {"state": state, "note": note, "ts": time.time()}

    def _llm_loop(self):
        """大模型独立线程：限流调用，绝不阻塞 3 秒一次的 tick 线程。

        这里最容易踩的坑是「第一次结果要等满 5 分钟」：LLM 线程和 tick 线程同时启动，
        此刻 last_feats 还是空的。如果此时直接 wait(llm_interval=300)，就白等一轮，
        用户只看到"未启用"。所以按状态给不同等待：
          缺配置 10s / 等行情 2s / 成功 llm_interval / 失败 30s。
        失败原因也必须写进状态，否则页面只能显示一句没有信息量的"未启用"。
        """
        while not self._stop.is_set():
            cfg = llm_live.resolve()
            interval = float(cfg.get("interval") or _cfg("llm_interval", 300))
            if not cfg.get("enabled"):
                self._set_llm_status("disabled", "未启用（页面「① 大模型接入」里打开）")
                self._stop.wait(10.0)
                continue
            if not cfg.get("ready"):
                miss = []
                if not cfg.get("base_url"):
                    miss.append("Base URL")
                if not cfg.get("model"):
                    miss.append("模型名")
                if not cfg.get("key") and cfg.get("provider") != "ollama":
                    miss.append("API Key")
                self._set_llm_status("not_ready",
                                     "配置不完整：缺 " + ("、".join(miss) or "未知项"))
                self._stop.wait(10.0)
                continue
            feats = getattr(self, "last_feats", None) or {}
            if not feats.get("valid") or len(self.bars) < 6:
                self._set_llm_status("waiting", "等行情样本（启动后约 5~20 秒）")
                self._stop.wait(2.0)
                continue
            self._set_llm_status("running", f"调用中（{cfg.get('model')}）…")
            t0 = time.time()
            try:
                res = llm_live.ask_intraday(self.symbol, self.name, feats, self.bars,
                                            session_state(self._clock_now()))
            except Exception as e:
                res = {"ok": False, "note": f"调用异常：{type(e).__name__}: {e}"}
            dt = time.time() - t0
            res["_try_ts"] = time.time()
            res["elapsed"] = round(dt, 2)
            with self._lock:
                self.llm_last = res
                if res.get("ok"):
                    res["_ts"] = time.time()
                    self.llm_res = res
            if res.get("ok"):
                self._set_llm_status(
                    "ok", f"成功：{res.get('model')} {float(res.get('bps') or 0):+.1f}bps"
                          f"（置信 {float(res.get('confidence') or 0):.2f}，{dt:.1f}s）")
                self._stop.wait(interval)
            else:
                self._set_llm_status("error", str(res.get("note") or "调用失败"))
                self._stop.wait(30.0)

    # ----------------------------- 状态快照 -----------------------------
    def _build_state(self, feats: dict):
        q = self._last_quote
        ens = getattr(self, "last_ens", None) or {}
        ctx = getattr(self, "last_ctx", None) or {}
        bars_view = [{"t": b["ts"][11:16], "o": b["open"], "h": b["high"],
                      "l": b["low"], "c": b["close"], "v": b.get("volume", 0)}
                     for b in self.bars[-240:]]
        preds = db.recent_live_predictions(self.symbol, limit=420)
        primary = int(_cfg("primary_horizon", 5))
        overlay = [{"t": str(r.get("target_ts") or "")[11:16],
                    "pred": r.get("pred_price"), "lo": r.get("lo"), "hi": r.get("hi"),
                    "realized": r.get("realized_price"), "h": r.get("horizon_min")}
                   for r in preds if int(r.get("horizon_min") or 0) == primary]
        stats = db.live_stats(self.symbol)
        lcfg = llm_live.resolve()
        state = {
            "symbol": self.symbol,
            "name": self.name or q.name or self.symbol,
            "mode": self.mode,
            "replay_done": self.replay_done,
            "session": session_state(self._clock_now()),
            "quote": q.to_dict() if q.price > 0 else None,
            "features": {k: (round(v, 3) if isinstance(v, float) else v)
                         for k, v in feats.items() if not isinstance(v, (list, dict))},
            "ensemble": ens,
            "experts": self.ens.snapshot(),
            "ml_note": self.ml.note,
            "band_calib": {"n": self._band_calib_n, "stats_n": self._stats_n,
                           "err": self._stats_err},
            "season_n": len(self.season),
            "turning": getattr(self, "turning", None),
            "llm": {"config": {k: lcfg[k] for k in
                               ("enabled", "ready", "provider", "base_url", "model",
                                "key_masked", "key_source", "interval", "preset")},
                    "status": self.llm_status,
                    "last": {k: v for k, v in (self.llm_last or {}).items()
                             if k not in ("_ts", "_try_ts")},
                    "result": {k: v for k, v in (self.llm_res or {}).items()
                               if k not in ("_ts",)},
                    "age_sec": round(time.time() - float(self.llm_res.get("_ts") or 0), 1)
                    if self.llm_res.get("_ts") else None},
            "news": self._news_view(),
            "bars": bars_view,
            "overlay": overlay[-200:],
            "stats": stats,
            "settle_count": self.settle_count,
            "db": db.db_info(),
            "error": self.error,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with self._lock:
            self.state = state

    def snapshot(self) -> dict:
        with self._lock:
            return self.state

    def run_llm_now(self) -> dict:
        feats = getattr(self, "last_feats", None) or {}
        res = llm_live.ask_intraday(self.symbol, self.name, feats, self.bars,
                                    session_state(self._clock_now()), force=True)
        if res.get("ok"):
            res["_ts"] = time.time()
            with self._lock:
                self.llm_res = res
        return res


# =====================================================================
# 引擎注册表（每只股票一个线程）
# =====================================================================
_ENGINES: dict[str, LiveEngine] = {}
_REG_LOCK = threading.RLock()


def get_engine(symbol: str, name: str = "", autostart: bool = True) -> LiveEngine:
    from .realtime import normalize
    sym = normalize(symbol)
    with _REG_LOCK:
        e = _ENGINES.get(sym)
        if e is None:
            e = LiveEngine(sym, name)
            _ENGINES[sym] = e
        elif name and not e.name:
            e.name = name
        if autostart and not e._started:
            e.start()
            threading.Thread(target=e._llm_loop, name=f"llm-{sym}", daemon=True).start()
        elif autostart and e.replay_done:
            # 回放已结束：重新开一轮（重置回放游标）
            e.replay_done = False
            e.mode = "replay" if not is_open() else "live"
            e.thread = threading.Thread(target=e._run, name=f"live-{sym}", daemon=True)
            e.thread.start()
        return e


def engines_status() -> list[dict]:
    with _REG_LOCK:
        return [{"symbol": e.symbol, "name": e.name, "mode": e.mode,
                 "updated_at": e.state.get("updated_at"),
                 "error": e.error} for e in _ENGINES.values()]


def stop_all():
    with _REG_LOCK:
        for e in _ENGINES.values():
            e.stop()
