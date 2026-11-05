"""全局配置：默认标的、预测周期、新闻源等。"""
from __future__ import annotations

# 默认预测标的（A 股代码，无需前缀也行，程序会自动补全 sh/sz）
DEFAULT_SYMBOL = "600519"   # 贵州茅台
DEFAULT_NAME = "贵州茅台"

# 预测周期（交易日）：1日/5日/7日/30日，均作为「预测线」绘出
HORIZONS = [1, 5, 7, 30]

# 训练用的历史 K 线根数（经验：700 根左右在 7/30 日误差上更稳）
KLINE_LEN = 700

# 新闻条数（用于情绪面）
NEWS_NUM = 40

# 大盘指数代码（作为方向预测的外生因子，个股 beta 强相关）
INDEX_SYMBOL = "sh000001"  # 上证指数

# 模型随机种子
RANDOM_STATE = 42

# 单日预测收益率截断（防止递归预测复利漂移失控；±5%/日，符合 A 股涨跌停约束）
PRED_RET_CLIP = 0.05

# 大样本横截面评估（32,130 样本）显示这些周期**没有方向预测力**：
#   30日方向准确率 48.3%（n=7830，SE≈0.55%，即 -3.1σ，显著低于随机）——连反向都不可用
# 故这些周期一律不给方向，只给价格预测，避免误导
NO_SKILL_HORIZONS = (30,)

# 方向置信度门控：只有上涨概率明显偏离 0.5 才给出方向，否则标"方向不明"
# （实测：无门控 ~49%；|prob-0.5|>=0.20 时准确率 ~67%，覆盖率 15%）
CONF_HI = 0.65
CONF_LO = 0.35

# 新闻情绪对每日收益率的微调强度：pred_ret += TILT * (sentiment - 0.5)
# sentiment∈[0,1]，故极端看涨最多 +TILT/日，极端看跌最多 -TILT/日（透明、有界）
SENTIMENT_TILT = 0.015

# 数据源开关：默认使用新浪（行情/历史真实可用）
# 若改用 yfinance（美股/全球），把 DATA_SOURCE 改为 "yfinance"
DATA_SOURCE = "sina"

# 输出目录
OUTPUT_DIR = "output"

# =====================================================================
# 股票池 / 本地缓存
#   POOL_MODE = "default" : 27 只行业龙头（仅调试用，样本量太小：1.9 万行）
#   POOL_MODE = "hs300"   : 沪深300 当前成分股（约 300 只，20.9 万行）← 默认
#   POOL_MODE = "zz500"   : 中证500
# POOL_LIMIT：只取前 N 只（0=不限），调试时很有用
# ⚠️ 用"当前成分股"回测历史存在**幸存者偏差**，策略超额要打折看（README 有说明）
# =====================================================================
POOL_MODE = "hs300"
POOL_LIMIT = 0

# 本地缓存：行情与成分股列表都落盘，避免每次跑都重新抓 300 只
CACHE_DIR = ".cache"
KLINE_CACHE_TTL = 86400        # K 线缓存有效期（秒），0=总是重新抓
POOL_CACHE_TTL = 7 * 86400     # 成分股缓存有效期（秒）

# 单次训练的最大样本行数（0=不限制）。
# 300 只票 × 700 天 ≈ 20.9 万行：策略打分每 20 天重训一次、单股分析还要跑 5 折回测，
# 全量训练会让单次分析从 1 分钟涨到 10 分钟以上。实测 8 万行抽样的 RankIC 与全量几乎一致
# （信号来自横截面排序，不是样本量），所以默认抽 8 万行控耗时；设 0 可关掉抽样。
MAX_TRAIN_ROWS = 60000

# =====================================================================
# 特征集 / 标签 / 模型（参考 Microsoft qlib 的 Alpha158 与模型库）
#   FEATURE_SET = "core"  : 现有 33 个手工特征（动量/均线/摆动/量能/形态/指数）
#   FEATURE_SET = "alpha" : core + qlib Alpha158 精简版（KBar 9 + 19 算子 × 3 窗口）
# LABEL_MODE = "ret"     : 回归目标 = h 日真实收益率（默认，便于还原价格）
# LABEL_MODE = "csrank"  : 回归目标 = 当日截面排名（qlib CSRankNorm，抗极值、跨股可比）
#   注意：方向分类标签永远是「真实收益 > 0」，不受 LABEL_MODE 影响。
# =====================================================================
FEATURE_SET = "core"
LABEL_MODE = "ret"
ALPHA_WINDOWS = (5, 10, 20)

# 模型：rf（随机森林）| gbdt（XGBoost/HistGBDT）| lstm（PyTorch）| ensemble（等权融合）
# 默认用 ensemble（RF+GBDT+LSTM 等权融合）：单模型在切分点很少时噪声太大，
# 融合能把单模型的偶然性摊掉。用户明确要求"决策要综合多个模型，不要只有 RF"。
MODEL = "ensemble"

# LSTM 超参（对齐 qlib examples/benchmarks/LSTM：hidden 64 / 2 层 / lr 1e-3）
LSTM = {
    "seq_len": 20,        # 回看窗口（qlib TSDatasetH 的 step_len）
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.1,
    "lr": 1e-3,
    "batch_size": 512,
    "max_epochs": 30,
    "early_stop": 6,
    "ret_scale": 0.02,    # 回归目标缩放（收益/2%），让 MSE 与 BCE 同量级
    "seed": RANDOM_STATE,
}

# =====================================================================
# 组合策略（参考 qlib TopkDropoutStrategy）：横截面打分选股 + 定期换仓
# 信号在 t 日收盘后生成，t+1 日收盘成交，收益从 t+1 收盘起算（故意保守，无前视）
# =====================================================================
STRATEGY = {
    "topk": 5,            # 持仓数量
    "n_drop": 1,          # 每次换仓替换几只
    "rebalance_days": 5,  # 换仓周期（交易日）
    "open_cost": 0.0005,  # 买入佣金
    "close_cost": 0.0015, # 卖出佣金+印花税
    "limit_pct": 0.095,   # 涨跌停阈值（触及则视为不可交易）
    "min_amount": 0.0,    # 流动性门槛（成交额，元），0=不限制
}

# =====================================================================
# 大模型（新闻/政策文本分析）
# 免费额度可选：
#   - 千问 DashScope（OpenAI 兼容）：https://dashscope.aliyuncs.com/compatible-mode/v1
#   - DeepSeek：https://api.deepseek.com/v1
#   - 本地 Ollama：http://127.0.0.1:11434/v1（无需密钥）
# 密钥读取顺序：环境变量 > 项目根 secrets.json
#   环境变量：DASHSCOPE_API_KEY / QWEN_API_KEY / DEEPSEEK_API_KEY / LLM_API_KEY
# =====================================================================
LLM_ENABLED = True
LLM_PROVIDER = "auto"        # auto | qwen | deepseek | ollama | none
LLM_MODEL = ""               # 留空则用下方默认模型
LLM_DEFAULT_MODELS = {
    # 百炼现在按版本号发模型，老名字 qwen-plus 在新账号里常常没有额度。
    # 省钱：qwen3.7-flash；更强：qwen3.8-max（本用途没必要，贵且慢）
    "qwen": "qwen3.7-plus",
    "deepseek": "deepseek-chat",
    "ollama": "qwen2.5:7b",
}
LLM_BASE_URLS = {
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "ollama": "http://127.0.0.1:11434/v1",
}
LLM_TIMEOUT = 90
LLM_MAX_NEWS = 30            # 送进模型的新闻条数（控制 token 成本）
LLM_CACHE_TTL = 1800         # 同一批新闻的缓存秒数

# =====================================================================
# 联合决策权重（各项均归一到 [-1,1] 后再加权；权重和不必为 1，内部会归一）
# 权重不是拍脑袋：`python run.py --signal-eval` 在本机 27 只股票池 18,900 行上实测，
#   K线形态 / 量能 的 IC ≈ 0.00~0.016、明确时方向命中率 49~50%（≈随机），
#   所以两者只给 0.15 的辅助权重（用于解释与时点判断），不主导方向；
#   模型方向概率是唯一有大样本统计支持（1日53.1%、5日52.1%）的信号 → 0.45；
#   新闻政策无历史新闻序列、无法回测，只能作辅助 → 0.15；
#   拐点模型样本外 AUC 0.60~0.69（见 --signal-eval），主要用于风险提示 → 0.10。
#
# 【大模型新闻/政策研判入决策】原权重里 news=0.15 是把"新闻情绪"和"政策影响"
# 按 6:4 混在一起的，用户要求新闻**与政策**都进决策，故拆成两路独立信号：
#   news   = 大模型对新闻的情绪判断（外生信息，可加权）
#   policy = 大模型对政策/监管的判断（外生信息，可加权）
# 两路合计仍为 0.15（与原口径等量），从 model 借 0.05、turn 不动。
# 注意：`analyze_decision_llm`（大模型对量化决策的二次解读）**仍然不进加权**——
# 那是让 LLM 给自己打分，属循环论证；有据可依的是"新闻/政策→情绪与政策影响"这条外生链路。
# =====================================================================
SIGNAL_WEIGHTS = {
    "model": 0.36,     # 模型方向概率（1/5/7 日，30 日无技能不参与）
    "pattern": 0.10,   # K 线形态
    "volume": 0.12,    # 量能/资金流/盘口
    "energy": 0.12,    # K 线能量（实体/影线/量能合成的多空强度，见 energy.py）
    "pressure": 0.10,  # 买卖压力量化（主动买卖差、五档不平衡、大单净额）
    "news": 0.08,      # 大模型新闻情绪研判
    "policy": 0.06,    # 大模型政策/监管研判
    "turn": 0.06,      # 拐点风险（顶部风险为负、底部机会为正）
}

# 综合分 → 建议的阈值
DECISION_BUY_STRONG = 0.35
DECISION_BUY = 0.15
DECISION_SELL = -0.15
DECISION_SELL_STRONG = -0.35

# =====================================================================
# 目标价位区间 + 买卖点预判
# 区间带宽 = max(多模型分歧, 波动率外推) × band_z，band_z=1.28 约对应 80% 覆盖。
# 止损受风险预算约束：单笔最大亏损不超过 risk_budget_pct。
# =====================================================================
TRADE = {
    "band_z": 1.28,           # 区间带宽系数
    "risk_budget_pct": 0.08,  # 单笔止损风险预算（占总资金 8% 对应价格跌幅）
    "atr_window": 14,         # ATR 窗口
    "vol_window": 20,         # 波动率窗口
    "horizons": [5, 7, 30],   # 要出结果的档位（每 N 个交易日一份结果）
}

# =====================================================================
# 拐点预测
# 拐点定义：收盘价的局部极值（左右各 TURN_CONFIRM_WIN 根内最高/最低），
#          且随后反向幅度 >= TURN_MIN_SWING 才算有效拐点（滤掉横盘噪声）。
# =====================================================================
TURN_CONFIRM_WIN = 3
TURN_MIN_SWING = 0.025        # 2.5%
TURN_HORIZONS = [5, 10, 20]   # 预测"未来 N 日内出现拐点"的概率

# =====================================================================
# 实时（分钟级）预测引擎
#
# 定位与边界（必须先说清楚，否则又会变成"AI 预测股价"的吹牛）：
#   - 目标：行情刷新（约 3 秒一次）后立刻给出「未来 1/3/5 分钟」的价格与方向，
#     精度以 bps（万分之一）衡量，单次误差通常 > 预测幅度本身。
#   - 5 分钟级别的可预测性极低（方向命中率长期在 50~55% 区间），
#     所以引擎必须①给出区间而非单点、②在线统计每一路专家的真实命中率并自适应降权、
#     ③在分歧大时明确输出"震荡/无优势"。任何把 5 分钟预测当稳赚的行为都会亏在手续费上。
#   - 非交易时段自动进入**回放模式**：用最后一根 1 分钟 K 线逐条推进，
#     让页面、结算、统计链路都能被真实验证，而不是给一堆假数字。
# =====================================================================
LIVE = {
    "db_path": "data/stockpred.db",
    "poll_interval": 3.0,        # 行情轮询间隔（秒），与新浪 hq 刷新节奏一致
    "horizons_min": [1, 3, 5],   # 预测未来 N 分钟
    "primary_horizon": 5,        # 主展示周期（对比图/落库结算用）
    "predict_every": 3.0,        # 最小预测间隔（秒）
    "bars_refresh": 20.0,        # 1 分钟 K 线刷新间隔（秒）
    "llm_interval": 300.0,       # 大模型参与间隔（秒）；每次都要花钱，不能跟着 3 秒刷
    # --- 实时新闻/政策/舆论面（实时专家池扩到 14 路后并入）---
    "news_interval": 180.0,      # 新闻抓取+研判间隔（秒）；抓一次要几秒，不能更密
    "news_half_life": 1800.0,    # 新闻影响半衰期（秒，30 分钟）；越旧越不该算数
    "news_stock_num": 12,        # 每次取多少条个股新闻
    "news_policy_num": 8,        # 政策新闻条数
    "news_market_num": 12,       # 市场新闻条数
    "llm_max_bps": 40.0,         # 大模型预测幅度硬上限（bps），防止离谱值污染加权
    "turn_interval": 15.0,       # 日内拐点重算间隔（秒）
    "replay_speed": 0.35,        # 回放模式每条 1 分钟线的间隔（秒）
    "replay_predict_every_bars": 1,   # 回放中每 N 根分钟线做一次预测（按模拟时间，不用墙钟）
    "replay_persist_every_bars": 3,   # 回放中每 N 次预测落一次库（防止一天写上千行）
    "stats_cache_sec": 15.0,     # 区间自校准读取历史误差的缓存时长（秒）
    "history_days": 10,          # 在线岭回归用的历史分钟线天数
    "max_pending": 400,          # 待结算预测上限（防止内存膨胀）
    "band_z": 1.28,              # 区间带宽（约 80% 覆盖）
    "min_samples_ml": 45,        # 岭回归最少训练样本（分钟 bar 数）
}

# 实时专家先验权重（在线 Hedge 会在此基础上按真实误差自适应调整，不必严格求和为 1）
# 14 路，与日线决策层的 SIGNAL_WEIGHTS 一一对应（见模块头说明）：
#   model→model  pattern→pattern  volume→volume  energy→energy
#   pressure→micro+flow（盘口挂单与主动成交两路拆开，比合成一个分更可结算）
#   news/policy→news（合并，内部仍分别算情绪与政策）  turn→turn
LIVE_EXPERT_PRIORS = {
    "micro": 0.10,    # 五档委买委卖不平衡 → 短期冲击（原 pressure 的挂单部分）
    "flow": 0.11,     # tick 主动买卖差 OFI（原 pressure 的成交部分）
    "volume": 0.06,   # 量价配合：上涨分钟的量占比 + 量能放大（原 volume）
    "energy": 0.10,   # K 线能量：实体/影线/收盘位置/量能合成（原 energy）
    "pattern": 0.06,  # 分钟级形态：连阳连阴过度、长影线、双顶双底（原 pattern）
    "range": 0.06,    # 日内区间位置 + 量能（突破/假突破）
    "mom": 0.08,      # 短周期动量延续
    "revert": 0.06,   # 相对 VWAP 的均值回复
    "season": 0.05,   # 同一时刻的历史平均漂移（日内季节性）
    "ml": 0.10,       # 在线岭回归（多因子，分钟级重新训练）
    "model": 0.05,    # 日线多模型方向概率（慢变量，只给窄幅偏置）
    "turn": 0.06,     # 拐点风险（顶部风险为负、底部机会为正，原 turn）
    "llm": 0.06,      # 大模型独立预测（可选，按时限流）
    "news": 0.05,     # 新闻/政策/舆论面（慢变量；无 Key 时走词典情绪）
}

# =====================================================================
# 大模型预设（实时预测页的下拉；用户自己填 Key，Key 只存本机 DB / 环境变量）
# 说明：全部走 OpenAI 兼容 /chat/completions，换供应商只改 base_url + model。
# =====================================================================
LLM_PRESETS = [
    {"id": "qwen", "label": "通义千问（阿里云百炼）",
     "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
     "model": "qwen3.7-plus",
     "note": "模型填 qwen3.7-plus；更省用 qwen3.7-flash，更强用 qwen3.8-max"},
    {"id": "deepseek", "label": "DeepSeek",
     "base_url": "https://api.deepseek.com/v1",
     "model": "deepseek-chat", "note": "按量计费，价格低；无免费额度"},
    {"id": "glm", "label": "智谱 GLM",
     "base_url": "https://open.bigmodel.cn/api/paas/v4",
     "model": "glm-4-flash", "note": "glm-4-flash 有免费额度"},
    {"id": "moonshot", "label": "月之暗面 Kimi",
     "base_url": "https://api.moonshot.cn/v1",
     "model": "moonshot-v1-8k", "note": "需充值，长文本能力强"},
    {"id": "openai", "label": "OpenAI",
     "base_url": "https://api.openai.com/v1",
     "model": "gpt-4o-mini", "note": "需海外网络与付费 Key"},
    {"id": "ollama", "label": "本地 Ollama（无需 Key）",
     "base_url": "http://127.0.0.1:11434/v1",
     "model": "qwen2.5:7b", "note": "完全免费，本机需先 ollama pull"},
    {"id": "custom", "label": "自定义（OpenAI 兼容端点）",
     "base_url": "", "model": "", "note": "任何兼容 /chat/completions 的服务"},
]
