"""实时跟踪 + 涨幅榜 独立入口（不依赖 pandas，二进制依赖为零）。

用法：
    python live.py                     # http://127.0.0.1:8020
    python live.py --port 8020
    python live.py --code 600519       # 启动时直接开始跟踪某只票

也可以用主入口 python web.py —— 那边同样挂了 /live /rank 和全部 /api 路由；
本文件适合「只想要实时跟踪 + 涨幅榜」的最小部署。
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from stockpred import live_web


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):        # 安静模式
        pass

    def _send(self, res):
        if res is None:
            res = (404, "text/plain; charset=utf-8", b"not found")
        status, ctype, body = res
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def do_GET(self):
        parsed = urlparse(self.path)
        self._send(live_web.handle_get(parsed.path, parse_qs(parsed.query)))

    def do_POST(self):
        parsed = urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        res = live_web.handle_post(parsed.path, parse_qs(parsed.query), body)
        if res is None:
            res = live_web.handle_get(parsed.path, parse_qs(parsed.query))
        self._send(res)


def main():
    ap = argparse.ArgumentParser(description="stockpred 实时跟踪 + 涨幅榜")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8020)
    ap.add_argument("--code", default=None, help="启动时立即开始跟踪的股票代码")
    args = ap.parse_args()
    if args.code:
        from stockpred.live_engine import get_engine
        get_engine(args.code, autostart=True)
        print(f"[live] 已开始跟踪 {args.code}")
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[live] 实时跟踪页  http://{args.host}:{args.port}/live?code=600519")
    print(f"[live] 涨幅榜      http://{args.host}:{args.port}/rank")
    print("[live] Ctrl+C 退出")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        from stockpred.live_engine import stop_all
        stop_all()


if __name__ == "__main__":
    main()
