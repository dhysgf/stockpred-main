"""把本机分析服务发布成公网临时链接（web.py + cloudflared 快速隧道）。

用法：
    python tools/serve_public.py                       # 随机口令，仅公网
    python tools/serve_public.py --password abc12345    # 指定口令
    python tools/serve_public.py --lan                  # 同时允许同一局域网直连
    python tools/serve_public.py --port 8030 --open-key # 允许访客留空用服务器 Key

它会做三件事：
1. 起 `web.py`（默认只绑 127.0.0.1，公网流量经隧道进来）；
2. 起 `cloudflared tunnel --url`，从输出里抓出 https://xxx.trycloudflare.com；
3. 打印口令与链接，Ctrl+C 一起收工。

为什么绑 127.0.0.1：隧道进程就在本机，公网流量经它转发，不需要把端口暴露到局域网。
想给同办公室的人直连再加 --lan（此时绑 0.0.0.0，局域网免口令、公网仍需口令）。
"""
from __future__ import annotations

import argparse
import os
import re
import secrets
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLOUDFLARED = os.path.join(ROOT, ".bin", "cloudflared.exe" if os.name == "nt" else "cloudflared")

URL_RE = re.compile(r"https://[a-z0-9][a-z0-9-]*\.trycloudflare\.com")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _pick_python() -> str:
    """优先用项目内 .venv 的解释器启动 web.py。

    为什么：依赖（pandas/sklearn/matplotlib）装在哪，就用哪的解释器。
    宿主机的 Python 环境被重置过（实测一夜之间四个包全没了，
    ModuleNotFoundError: No module named 'matplotlib'），而 .venv 在项目目录里，
    不会被外部清理工具动到，是唯一稳定的选择。
    """
    for rel in (os.path.join(".venv", "Scripts", "python.exe"),
                os.path.join(".venv", "bin", "python")):
        p = os.path.join(ROOT, rel)
        if os.path.exists(p):
            return p
    return sys.executable


def _pump(proc: subprocess.Popen, on_url=None, prefix: str = "") -> None:
    """逐行转发子进程输出；顺带把隧道域名挑出来。"""
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = ANSI_RE.sub("", raw.rstrip())
        if not line:
            continue
        low = line.lower()
        # cloudflared 会刷一堆 INF 日志，只留有用的：错误、域名、连接状态
        if on_url and "trycloudflare.com" in line:
            m = URL_RE.search(line)
            if m:
                on_url(m.group(0))
            continue
        if any(k in low for k in ("err", "warn", "failed", "registered", "connection")):
            print(f"{prefix}{line}")


def main() -> None:
    ap = argparse.ArgumentParser(description="本机服务 → 公网临时链接")
    ap.add_argument("--port", type=int, default=8030)
    ap.add_argument("--password", default=os.environ.get("STOCKPRED_PASSWORD", ""))
    ap.add_argument("--lan", action="store_true", help="同时允许局域网直连（绑 0.0.0.0）")
    ap.add_argument("--open-key", action="store_true",
                    help="允许访客留空、使用服务器内置大模型 Key（默认关闭，不烧本机额度）")
    ap.add_argument("--no-tunnel", action="store_true", help="只起服务，不建隧道")
    args = ap.parse_args()

    password = (args.password or "").strip() or secrets.token_urlsafe(6)
    host = "0.0.0.0" if args.lan else "127.0.0.1"

    py = _pick_python()
    cmd = [py, os.path.join(ROOT, "web.py"),
           "--host", host, "--port", str(args.port), "--password", password]
    if not args.open_key:
        cmd.append("--require-key")

    print("=" * 66)
    print("  stockpred 本机对外服务")
    print("=" * 66)
    print(f"  访问口令： {password}")
    print(f"  大模型 Key： {'访客可留空用服务器 Key' if args.open_key else '只认访客自带的 Key'}")
    print(f"  监听地址： {host}:{args.port}")
    print(f"  解释器：   {py}")
    print("=" * 66)

    web = subprocess.Popen(cmd, cwd=ROOT,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, encoding="utf-8", errors="replace", bufsize=1)
    threading.Thread(target=_pump, args=(web,), kwargs={"prefix": "[web] "}, daemon=True).start()

    # 等服务真正开始监听，再建隧道，否则 cloudflared 会先报 502
    time.sleep(6)
    if web.poll() is not None:
        print("web.py 启动失败，先单独跑 python web.py 看报错。", file=sys.stderr)
        raise SystemExit(1)

    if args.no_tunnel:
        print("\n未建隧道：本机 http://127.0.0.1:%d   Ctrl+C 退出\n" % args.port)
        try:
            web.wait()
        except KeyboardInterrupt:
            pass
        finally:
            web.terminate()
        return

    if not os.path.isfile(CLOUDFLARED):
        print(f"缺少隧道程序：{CLOUDFLARED}\n"
              f"下载：https://github.com/cloudflare/cloudflared/releases/latest/download/"
              f"cloudflared-windows-amd64.exe 放到 .bin/ 下即可。", file=sys.stderr)
        web.terminate()
        raise SystemExit(1)

    tunnel = subprocess.Popen([CLOUDFLARED, "tunnel", "--url", f"http://127.0.0.1:{args.port}",
                               "--no-autoupdate"],
                              cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", bufsize=1)

    found: list[str] = []

    def on_url(url: str) -> None:
        if url in found:
            return
        found.append(url)
        line = "=" * 66
        print("\n" + line)
        print(f"  公网链接： {url}")
        print(f"  访问口令： {password}")
        print(line + "\n")
        try:
            with open(os.path.join(ROOT, "public_url.txt"), "w", encoding="utf-8") as f:
                f.write(f"{url}\npassword: {password}\n")
        except OSError:
            pass

    threading.Thread(target=_pump, args=(tunnel,), kwargs={"on_url": on_url},
                     daemon=True).start()

    try:
        while True:
            if tunnel.poll() is not None:
                print("\n隧道进程已退出。", file=sys.stderr)
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在关闭…")
    finally:
        for p in (tunnel, web):
            if p.poll() is None:
                p.terminate()
        print("已停止。")


if __name__ == "__main__":
    main()
