"""
US Regime Monitor - 完整实验脚本（双轨版 + 控制图）
"""

import os
import json
import logging
import urllib3
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# 关闭 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ===================== 配置 =====================
load_dotenv()
FRED_API_KEY = os.environ.get("FRED_API_KEY")

if not FRED_API_KEY:
    raise ValueError("请在 .env 文件中设置 FRED_API_KEY")

LOG_FILE = "experiment.log"
STATE_FILE = "state_history.csv"
T0_FILE = "t0_trigger.json"
AUCTION_CACHE_FILE = "auction_cache.csv"
BASELINE_START = "2025-01-01"
BASELINE_END = "2026-06-21"
EWMA_LAMBDA = 0.94
CUSUM_K = 0.5
CUSUM_H = 5.0
LAYER1_THRESHOLD = 2.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ===================== FRED 数据拉取 =====================
def fetch_fred(series_id, start_date, end_date=None):
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
        "sort_order": "asc",
        "limit": 10000
    }
    try:
        resp = requests.get(url, params=params, timeout=30, verify=False)
        resp.raise_for_status()
        data = resp.json()
        obs = data.get("observations", [])
        if not obs:
            return pd.DataFrame(columns=["date", "value"])
        df = pd.DataFrame(obs)[["date", "value"]]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        return df.dropna(subset=["value"])
    except Exception as e:
        logger.error(f"FRED请求失败 {series_id}: {e}")
        return pd.DataFrame(columns=["date", "value"])

# ===================== 拍卖数据（双轨） =====================
def fetch_auction_api(day=31):
    """从 TreasuryDirect API 获取近期拍卖数据"""
    try:
        url = "http://www.treasurydirect.gov/TA_WS/securities/auctioned"
        params = {"format": "json", "day": day}
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        auctions = []
        for item in data:
            row = {
                "auction_date": item.get("auctionDate"),
                "security_type": item.get("securityType"),
                "security_term": item.get("securityTerm"),
                "bid_to_cover": item.get("bidToCoverRatio"),
                "total_accepted": item.get("totalAccepted"),
                "total_submitted": item.get("totalSubmitted")
            }
            auctions.append(row)
        df = pd.DataFrame(auctions)
        logger.info(f"API 获取 {len(df)} 条拍卖记录")
        return df
    except Exception as e:
        logger.warning(f"API 获取失败: {e}")
        return pd.DataFrame()

def get_latest_pd():
    """获取最近的一级交易商获配比例（从 bid_to_cover 推算）"""
    # 优先从缓存读取
    if os.path.exists(AUCTION_CACHE_FILE):
        cache = pd.read_csv(AUCTION_CACHE_FILE)
        if "primary_dealer_award_pct" in cache.columns:
            val = float(cache["primary_dealer_award_pct"].iloc[-1])
            logger.info(f"从缓存读取 PD: {val:.2f}%")
            return val

    # 拉取新数据
    df = fetch_auction_api()
    if df.empty or "bid_to_cover" not in df.columns:
        logger.warning("无法获取拍卖数据，使用默认值 15%")
        return 15.0

    # 用 bid_to_cover 推算 PD（经验公式：PD ≈ 30 - 2 * bid_to_cover）
    # 当 bid_to_cover 低时，PD 高（被迫接盘多）
    latest_bid = float(df["bid_to_cover"].iloc[-1])
    pd_est = max(5, min(40, 30 - 2 * latest_bid))
    logger.info(f"bid_to_cover={latest_bid:.2f}, 推算 PD={pd_est:.2f}%")

    # 缓存
    cache_df = pd.DataFrame([{
        "primary_dealer_award_pct": pd_est,
        "cached_at": datetime.now().isoformat(),
        "bid_to_cover": latest_bid
    }])
    cache_df.to_csv(AUCTION_CACHE_FILE, index=False)

    return pd_est

# ===================== 核心计算 =====================
def ewma_smoothing(series, lambda_val=EWMA_LAMBDA):
    return series.ewm(alpha=1-lambda_val, adjust=False).mean()

def cusum_update(s_t, value, mu0, sigma0):
    k = CUSUM_K * sigma0
    return max(0, s_t + value - mu0 - k)

def compute_baseline_stats(compressions):
    if len(compressions) < 10:
        return 0.0, 1.0
    return np.mean(compressions), np.std(compressions)

# ===================== 控制图生成 =====================
def generate_control_chart():
    """从 state_history.csv 生成控制图"""
    if not os.path.exists(STATE_FILE):
        logger.warning("state_history.csv 不存在，跳过生成控制图")
        return

    df = pd.read_csv(STATE_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if df.empty:
        logger.warning("state_history.csv 为空，跳过生成控制图")
        return

    # 取最近 90 天
    df = df.tail(90)

    plt.figure(figsize=(12, 5))
    plt.plot(df["timestamp"], df["s_t"], label="S_t", color="blue", linewidth=1.5)
    plt.axhline(y=df["control_limit"].iloc[-1], color="red", linestyle="--", label="控制限 (5σ₀)")
    plt.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    # 标记触发点
    triggered_rows = df[df["triggered"] == True]
    if not triggered_rows.empty:
        plt.scatter(triggered_rows["timestamp"], triggered_rows["s_t"],
                   color="red", s=100, zorder=5, label="🚨 T₀ 触发")

    plt.xlabel("日期")
    plt.ylabel("CUSUM S_t")
    plt.title("US Regime Monitor - 控制图")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("control_chart.png", dpi=150)
    plt.close()
    logger.info("控制图已更新: control_chart.png")

# ===================== 主实验 =====================
def run_experiment():
    logger.info("=" * 60)
    logger.info(f"实验运行 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. FRED 数据
    logger.info("拉取 FRED 数据...")
    dgs10 = fetch_fred("DGS10", BASELINE_START)
    sofr = fetch_fred("SOFR", BASELINE_START)
    if dgs10.empty:
        logger.error("DGS10 为空，中止")
        return

    # 2. 拍卖数据
    logger.info("获取拍卖数据...")
    current_pd = get_latest_pd()
    logger.info(f"一级交易商获配比例: {current_pd:.2f}%")

    # 3. 构建压缩比序列
    basis = dgs10["value"]
    # 用当前 PD 值填充所有日期（简化，后续可优化为逐日匹配）
    compression = pd.Series([current_pd / (b + 0.001) for b in basis])

    # 4. 期限溢价（SOFR 波动代理）
    if not sofr.empty:
        term_premium = sofr["value"].pct_change() * 100
    else:
        term_premium = pd.Series([0] * len(basis))

    # 5. 基线标定
    comp_clean = compression.dropna()
    if len(comp_clean) < 20:
        logger.warning("数据不足，使用默认基线")
        mu0, sigma0 = 5.0, 0.3
    else:
        mu0, sigma0 = compute_baseline_stats(comp_clean.iloc[:int(len(comp_clean)*0.8)])
    logger.info(f"基线: mu0={mu0:.4f}, sigma0={sigma0:.4f}")

    # 6. CUSUM 状态
    if os.path.exists(STATE_FILE):
        hist = pd.read_csv(STATE_FILE)
        last_s = hist["s_t"].iloc[-1] if not hist.empty else 0
    else:
        last_s = 0

    current_comp = comp_clean.iloc[-1] if not comp_clean.empty else 1.0
    current_term = term_premium.iloc[-1] if not term_premium.empty else 0

    s_t = cusum_update(last_s, current_comp, mu0, sigma0)
    control_limit = CUSUM_H * sigma0
    triggered = s_t > control_limit
    layer1_triggered = abs(current_term) > LAYER1_THRESHOLD * sigma0

    # 7. 记录状态
    state = {
        "timestamp": datetime.now().isoformat(),
        "compression": current_comp,
        "term_premium": current_term,
        "s_t": s_t,
        "mu0": mu0,
        "sigma0": sigma0,
        "control_limit": control_limit,
        "layer1": layer1_triggered,
        "triggered": triggered,
        "primary_dealer_pct": current_pd
    }
    df_new = pd.DataFrame([state])
    if os.path.exists(STATE_FILE):
        existing = pd.read_csv(STATE_FILE)
        df_new = pd.concat([existing, df_new], ignore_index=True)
    df_new.to_csv(STATE_FILE, index=False)

    # 8. 触发判断
    if triggered:
        t0_data = {
            "t0_timestamp": datetime.now().isoformat(),
            "compression": current_comp,
            "s_t": s_t,
            "mu0": mu0,
            "sigma0": sigma0,
            "layer1_triggered": layer1_triggered,
            "primary_dealer_pct": current_pd
        }
        with open(T0_FILE, "w") as f:
            json.dump(t0_data, f, indent=2)
        logger.warning(f"🚨 T₀ 触发！时间戳: {t0_data['t0_timestamp']}")
    else:
        logger.info(f"✓ 未触发 | S_t={s_t:.4f} | 控制限={control_limit:.4f}")

    # 9. 生成控制图
    generate_control_chart()

    logger.info("=" * 60)

# ===================== 入口 =====================
if __name__ == "__main__":
    if os.path.exists(T0_FILE):
        logger.info("T₀ 已记录，如需重新实验请删除 t0_trigger.json")
    else:
        run_experiment()
