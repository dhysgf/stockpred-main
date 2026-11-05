"""把本地 HEAD 的树同步到远端 main —— 走 GitHub Git Data API，绕开 git 协议。

为什么需要它：`git push` 走 github.com:443，代理常直接拒 CONNECT（502），
而 api.github.com 可用。用法：

    GH_TOKEN=ghp_xxx python tools/deploy_api.py

两个已知约束：
1. token 需要 repo 权限；**没有 workflow 权限时，含 .github/workflows 的树会被
   GitHub 整体拒绝（返回 404，不是 403）**，脚本会跳过该目录并提示。
2. 脚本以远端当前 main 为 parent 建提交，因此远端历史与本地区提交历史相互独立，
   内容一致但 SHA 不同——这是刻意取舍：内容同步优先于历史同步。

安全：token 只从环境变量读，不写文件、不进命令行参数。
"""
import base64
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

TOKEN = os.environ["GH_TOKEN"]
REPO = "dhysgf/stockpred-main"
API = f"https://api.github.com/repos/{REPO}"
BRANCH = "main"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def call(method, url, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"token {TOKEN}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "stockpred-deploy")
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code} {method} {url}\n{e.read().decode()[:800]}", file=sys.stderr)
        raise


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout


def local_tree() -> dict:
    """path -> (mode, sha)，来自本地 HEAD。"""
    out = git("ls-tree", "-r", "HEAD")
    files = {}
    for line in out.splitlines():
        meta, path = line.split("\t", 1)
        mode, _type, sha = meta.split()
        files[path] = (mode, sha)
    return files


def remote_tree(tree_sha: str) -> dict:
    d = call("GET", f"{API}/git/trees/{tree_sha}?recursive=1")
    return {e["path"]: (e["mode"], e["sha"]) for e in d["tree"] if e["type"] == "blob"}


def main() -> None:
    ref = call("GET", f"{API}/git/ref/heads/{BRANCH}")
    head_sha = ref["object"]["sha"]
    commit = call("GET", f"{API}/git/commits/{head_sha}")
    base_tree = commit["tree"]["sha"]
    print(f"远端 {BRANCH}: {head_sha[:7]}  tree {base_tree[:7]}")

    loc = local_tree()
    rem = remote_tree(base_tree)
    print(f"本地 {len(loc)} 个文件，远端 {len(rem)} 个文件")

    entries = []
    # 删除：远端有、本地无
    for path in sorted(set(rem) - set(loc)):
        entries.append({"path": path, "mode": rem[path][0], "type": "blob", "sha": None})
    # 新增 / 修改：内容 sha 不同
    changed = [p for p in sorted(loc) if p not in rem or rem[p][1] != loc[p][1]]
    print(f"删除 {len(entries)} 个，新增/修改 {len(changed)} 个")

    # 分两步建树：先删后增，便于定位失败的那一批
    cur_tree = base_tree
    if entries:
        t = call("POST", f"{API}/git/trees", {"base_tree": cur_tree, "tree": entries})
        cur_tree = t["sha"]
        print(f"  删除批 -> {cur_tree[:7]}")

    adds = []
    for path in changed:
        # token 无 workflow scope，GitHub 会整体拒绝含 .github/workflows 的树（返回 404）
        if path.startswith(".github/workflows/"):
            print(f"  ~ 跳过 {path}（token 无 workflow 权限）")
            continue
        mode, sha = loc[path]
        raw = subprocess.run(["git", "cat-file", "blob", sha],
                             cwd=ROOT,
                             capture_output=True, check=True).stdout
        created = call("POST", f"{API}/git/blobs", {
            "content": base64.b64encode(raw).decode(),
            "encoding": "base64",
        })
        adds.append({"path": path, "mode": mode, "type": "blob", "sha": created["sha"]})
        print(f"  + {path}  ({len(raw):,} B)")

    if adds:
        t = call("POST", f"{API}/git/trees", {"base_tree": cur_tree, "tree": adds})
        cur_tree = t["sha"]
        print(f"  新增批 -> {cur_tree[:7]}")

    new_tree = {"sha": cur_tree}
    msg = git("log", "-1", "--pretty=%B").strip()
    new_commit = call("POST", f"{API}/git/commits", {
        "message": msg,
        "tree": new_tree["sha"],
        "parents": [head_sha],
    })
    print(f"新 tree {new_tree['sha'][:7]}  新 commit {new_commit['sha'][:7]}")

    call("PATCH", f"{API}/git/refs/heads/{BRANCH}", {"sha": new_commit["sha"], "force": True})
    print(f"==> {BRANCH} 已更新到 {new_commit['sha']}")
    print(f"本地 HEAD {git('rev-parse', '--short', 'HEAD').strip()} · "
          f"本地与远端历史独立，内容以本脚本推送的树为准")


if __name__ == "__main__":
    main()
