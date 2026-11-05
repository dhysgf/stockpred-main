"""stockpred 包：股价走势预测工具（价格 / 方向 / 拐点 + 多信号联合决策）。

两层结构：
- **量化分析层**（pandas/sklearn）：日线多周期预测、回测、报告 —— 需要 pandas；
- **实时层**（纯标准库）：分钟级跟踪、涨幅榜、SQLite 落库、大模型接入 —— 无 pandas 依赖，
  ``python live.py`` 只用这一层也能跑。

所以这里的 import 全部带 ImportError 保护：环境缺 pandas 时，实时层照常可用，
只有调用量化分析才会报"缺少 pandas"。
"""
from __future__ import annotations

_IMPORT_ERRORS: list[str] = []

# ------------------------- 实时层（无 pandas 依赖） -------------------------
from . import db
from .energy import candle_energy, energy_signal, pressure_score, pressure_signal
from .realtime import (Quote, TickBuffer, add_trading_minutes, compute_features,
                       fetch_minute_bars, fetch_quote, is_open, minute_index,
                       session_state)
from .rank_fetcher import fetch_rank
from .turning_live import detect_turns, predict_extremes, predict_next_turn, rebuild

try:  # ------------------------- 量化分析层（需要 pandas） -------------------------
    from .data_fetcher import load, fetch_kline, fetch_money_flow, normalize_symbol
    from .news_fetcher import fetch_market_news, fetch_stock_news, fetch_policy_news, collect_news
    from .sentiment import analyze_news
    from .patterns import add_patterns, detect_patterns, pattern_summary
    from .volume import add_volume_indicators, volume_score, depth_stats, obv_divergence
    from .features import add_technicals, build_horizon_samples, enrich_with_index, FEATURES
    from .model import ForecastModel, forecast
    from .models import (build_forecaster, RFForecaster, GBDTForecaster, LSTMForecaster,
                         EnsembleForecaster)
    from .alpha import add_alpha158_lite, alpha_columns
    from .strategy import walk_forward_scores, backtest_topk, score_ic
    from .universe import get_universe, pool_label
    from .turning_point import (turning_labels, recent_turns, build_turn_samples,
                                TurningPointModel, eval_turning)
    from .decision import decide, eval_signals
    from .analysis import analyze
    from .backtest import walk_forward_backtest
    from .visualize import plot_forecast, plot_backtest, plot_candles, plot_equity
    from .daily_store import store_from_analysis, sync_symbol
    _HAS_PANDAS = True
except ImportError as _e:      # pandas/sklearn 缺失：实时层照常工作
    _HAS_PANDAS = False
    _IMPORT_ERRORS.append(str(_e))


def has_pandas() -> bool:
    return _HAS_PANDAS


def import_errors() -> list[str]:
    return list(_IMPORT_ERRORS)


__all__ = [
    # 实时层
    "db", "candle_energy", "energy_signal", "pressure_score", "pressure_signal",
    "Quote", "TickBuffer", "add_trading_minutes", "compute_features",
    "fetch_minute_bars", "fetch_quote", "is_open", "minute_index", "session_state",
    "fetch_rank", "detect_turns", "predict_extremes", "predict_next_turn", "rebuild",
    "has_pandas", "import_errors",
    # 量化层（可用时）
    "load", "fetch_kline", "fetch_money_flow", "normalize_symbol",
    "fetch_market_news", "fetch_stock_news", "fetch_policy_news", "collect_news",
    "analyze_news",
    "add_patterns", "detect_patterns", "pattern_summary",
    "add_volume_indicators", "volume_score", "depth_stats", "obv_divergence",
    "add_technicals", "build_horizon_samples", "enrich_with_index", "FEATURES",
    "ForecastModel", "forecast",
    "build_forecaster", "RFForecaster", "GBDTForecaster", "LSTMForecaster",
    "EnsembleForecaster", "add_alpha158_lite", "alpha_columns",
    "walk_forward_scores", "backtest_topk",
    "turning_labels", "recent_turns", "build_turn_samples", "TurningPointModel", "eval_turning",
    "decide", "eval_signals", "analyze",
    "walk_forward_backtest", "plot_forecast", "plot_backtest", "plot_candles", "plot_equity",
    "store_from_analysis", "sync_symbol",
]
