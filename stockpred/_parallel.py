"""sklearn / joblib 并行后端选择。

**为什么需要这个模块**

`RandomForest*.fit(n_jobs=-1)` 在 Windows 上默认走 joblib 的 loky **多进程**：
每个 worker 都要重新 import pandas/sklearn（实测 5~8 秒），还要把训练矩阵
序列化复制过去。一次完整分析是 `HORIZONS = [1,5,7,30]` × 2 个森林 =
8 次 fit，于是：

- 单周期 fit 就要 100 秒以上（真实测量）；
- 在 `web.py` 的 ThreadingHTTPServer 里，进程内存涨到 4.5GB，
  请求 300 秒不返回，用户看到的就是「点深度分析没反应」。

随机森林的建树在 Cython 里会释放 GIL，所以 **线程后端同样能并行**，
却没有进程启动与内存复制的代价。这个模块统一把森林类 fit 包进线程后端。

保留 `n_jobs=-1`：线程数仍按 CPU 核数走，只是把"进程"换成"线程"。
"""

from __future__ import annotations

from contextlib import contextmanager

try:  # joblib 随 sklearn 一起装，正常都在
    from joblib import parallel_backend as _parallel_backend
except Exception:  # pragma: no cover - 极端环境缺 joblib
    _parallel_backend = None


@contextmanager
def forest_backend():
    """在 `with` 内让 sklearn 森林用线程并行，而不是多进程。

    joblib 未安装时退化为空上下文（sklearn 自己会处理 n_jobs）。
    """
    if _parallel_backend is None:
        yield
        return
    with _parallel_backend("threading"):
        yield
