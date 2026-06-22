"""
US Regime Monitor - 双轨拍卖数据采集
轨1: FiscalData API (高时效，日常扫描)
轨2: Investor Class Allotments (高精度，预警确认)
"""

import os
import json
import logging
import urllib3
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
from io import BytesIO
from dotenv import load_dotenv

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

# ===================== FRED 数据 =====================
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

# ===================== 轨1: FiscalData API (高时效) =====================
def fetch_auction_api(day=7):
    """
    从 TreasuryDirect API 获取近期拍卖数据（用于快速预警）
    API: http://www.treasurydirect.gov/TA_WS/securities/auctioned
    """
    try:
        url = "http://www.treasurydirect.gov/TA_WS/securities/auctioned"
        params = {
            "format": "json",
            "day": day
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        # 解析数据
        auctions = []
        for item in data:
            row = {
                "auction_date": item.get("auctionDate"),
                "security_type": item.get("securityType"),
                "security_term": item.get("securityTerm"),
                "cusip": item.get("cusip"),
                "issue_date": item.get("issueDate"),
                "maturity_date": item.get("maturityDate"),
                "high_yield": item.get("highYield"),
                "high_rate": item.get("highRate"),
                "price": item.get("price"),
                "total_accepted": item.get("totalAccepted"),
                "bid_to_cover": item.get("bidToCoverRatio"),
                "total_submitted": item.get("totalSubmitted")
            }
            auctions.append(row)
        
        df = pd.DataFrame(auctions)
        logger.info(f"API 获取 {len(df)} 条拍卖记录 (时效优先)")
        return df
        
    except Exception as e:
        logger.warning(f"API 获取失败: {e}")
        return pd.DataFrame()

# ===================== 轨2: Investor Class (高精度) =====================
def fetch_investor_class_allotments():
    """
    从财政部 Investor Class Auction Allotments 获取一级交易商获配比例
    数据源: https://home.treasury.gov/data/investor-class-auction-allotments
    """
    # 尝试获取最新一期的 Coupon 和 Bill 数据
    # 文件名格式: June_8_2026_IC_Coupons.xls
    
    # 先获取最新文件的发布日期
    try:
        # 方法1: 尝试从网页获取最新链接（更可靠）
        # 但解析HTML较复杂，先用当前已知的最新文件
        # 建议定期更新这里的日期
        base_url = "https://home.treasury.gov/system/files/276/"
        
        # 尝试多个可能的文件名（往前推几个月）
        dates_to_try = []
        today = datetime.now()
        for i in range(0, 90, 7):
            d = today - timedelta(days=i)
            dates_to_try.append(d.strftime("%B_%d_%Y").replace("_0", "_"))
        
        # 去重并尝试 Coupon 和 Bill
        for date_str in dates_to_try:
            for suffix in ["Coupons", "Bills"]:
                filename = f"{date_str}_IC_{suffix}.xls"
                url = base_url + filename
                try:
                    resp = requests.head(url, timeout=5)
                    if resp.status_code == 200:
                        logger.info(f"找到文件: {filename}")
                        return fetch_xls_allotments(url)
                except:
                    continue
    except Exception as e:
        logger.warning(f"查找 Investor Class 文件失败: {e}")
    
    # 如果找不到最新文件，尝试直接使用已知的最新链接
    known_urls = [
        "https://home.treasury.gov/system/files/276/June_8_2026_IC_Coupons.xls",
        "https://home.treasury.gov/system/files/276/June_8_2026_IC_Bills.xls"
    ]
    for url in known_urls:
        try:
            resp = requests.head(url, timeout=5)
            if resp.status_code == 200:
                logger.info(f"使用已知文件: {url}")
                return fetch_xls_allotments(url)
        except:
            continue
    
    logger.warning("无法获取 Investor Class 数据")
    return pd.DataFrame()

def fetch_xls_allotments(url):
    """下载并解析 Investor Class Allotments XLS 文件"""
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        
        # 读取 Excel 文件
        df = pd.read_excel(BytesIO(resp.content), sheet_name=0)
        logger.info(f"解析 XLS 成功，{len(df)} 行，列: {df.columns.tolist()}")
        
        # 查找一级交易商相关列
        # 常见列名: "Primary Dealer", "Primary Dealers", "Primary Dealer (Pct)"
        pd_col = None
        for col in df.columns:
            if "primary" in col.lower() and "dealer" in col.lower():
                pd_col = col
                break
            if "pd" in col.lower() and "pct" in col.lower():
                pd_col = col
                break
        
        if pd_col is None:
            logger.warning("未找到一级交易商列")
            return df
        
        # 提取日期列
        date_col = None
        for col in df.columns:
            if "date" in col.lower() or "auction" in col.lower():
                date_col = col
                break
        
        # 提取安全类型列
        sec_col = None
        for col in df.columns:
            if "security" in col.lower() or "type" in col.lower() or "term" in col.lower():
                sec_col = col
                break
        
        # 构建标准输出
        result = pd.DataFrame()
        if date_col:
            result["auction_date"] = pd.to_datetime(df[date_col], errors="coerce")
        if sec_col:
            result["security_type"] = df[sec_col]
        if pd_col:
            result["primary_dealer_award_pct"] = pd.to_numeric(df[pd_col], errors="coerce")
        
        result = result.dropna(subset=["primary_dealer_award_pct"])
        logger.info(f"提取 {len(result)} 条一级交易商数据")
        return result
        
    except Exception as e:
        logger.error(f"解析 XLS 失败: {e}")
        return pd.DataFrame()

# ===================== 双轨获取函数 =====================
def get_auction_data_dual():
    """
    双轨获取拍卖数据
    返回: (DataFrame, source) 其中 source 为 'api' 或 'investor'
    """
    # 轨1: 先尝试 API（快速）
    df_api = fetch_auction_api(day=7)
    if not df_api.empty:
        logger.info("✅ 使用 API 数据（时效优先）")
        return df_api, 'api'
    
    # 轨2: API 失败，尝试 Investor Class（高精度）
    df_investor = fetch_investor_class_allotments()
    if not df_investor.empty:
        logger.info("✅ 使用 Investor Class 数据（高精度）")
        return df_investor, 'investor'
    
    # 都失败，返回空
    logger.warning("❌ 所有数据源均失败")
    return pd.DataFrame(), 'none'

def get_latest_pd_dual():
    """
    双轨获取最新一级交易商获配比例
    优先使用 API 数据（快速），若需要精确确认则使用 Investor Class
    """
    # 先检查缓存（有效期24小时）
    if os.path.exists(AUCTION_CACHE_FILE):
        cache = pd.read_csv(AUCTION_CACHE_FILE)
        if "cached_at" in cache.columns:
            cache_time = pd.to_datetime(cache["cached_at"].iloc[0])
            if (datetime.now() - cache_time) < timedelta(hours=24):
                # 缓存有效
                if "primary_dealer_award_pct" in cache.columns:
                    val = float(cache["primary_dealer_award_pct"].iloc[0])
                    logger.info(f"使用缓存 PD: {val:.2f}%")
                    return val
    
    # 获取新数据
    df, source = get_auction_data_dual()
    if df.empty:
        # 如果 Investor Class 也失败，尝试从缓存找历史值
        if os.path.exists(AUCTION_CACHE_FILE):
            cache = pd.read_csv(AUCTION_CACHE_FILE)
            if "primary_dealer_award_pct" in cache.columns:
                val = float(cache["primary_dealer_award_pct"].iloc[0])
                logger.warning(f"数据源均失败，使用缓存历史值: {val:.2f}%")
                return val
        logger.warning("无法获取拍卖数据，使用默认值15%")
        return 15.0
    
    # 尝试提取 primary_dealer_award_pct
    if "primary_dealer_award_pct" in df.columns:
        pd_val = float(df["primary_dealer_award_pct"].iloc[-1])
    else:
        # 如果是 API 数据，用 bid_to_cover 作为代理（预警用）
        if source == 'api' and "bid_to_cover" in df.columns:
            # bid_to_cover 高 = 需求强 = PD 可能被迫接盘少
            # 这里仅作为预警代理，不作为精确值
            bid = float(df["bid_to_cover"].iloc[-1])
            pd_val = max(5, 30 - bid * 2)  # 粗略转换
            logger.info(f"使用 bid_to_cover 推算 PD: {pd_val:.2f}% (代理值)")
        else:
            pd_val = 15.0
    
    # 保存缓存
    cache_df = pd.DataFrame([{
        "primary_dealer_award_pct": pd_val,
        "cached_at": datetime.now().isoformat(),
        "source": source,
        "data_count": len(df)
    }])
    cache_df.to_csv(AUCTION_CACHE_FILE, index=False)
    
    return pd_val

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

# ===================== 主实验 =====================
def run_experiment():
    logger.info("=" * 60)
    logger.info(f"实验运行 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. FRED数据
    logger.info("拉取 FRED 数据...")
    dgs10 = fetch_fred("DGS10", BASELINE_START)
    sofr = fetch_fred("SOFR", BASELINE_START)
    if dgs10.empty:
        logger.error("DGS10为空，中止")
        return

    # 2. 拍卖数据（双轨）
    logger.info("获取拍卖数据（双轨）...")
    current_pd = get_latest_pd_dual()
    logger.info(f"一级交易商获配比例: {current_pd:.2f}%")

    # 3. 基差
    basis = dgs10["value"]

    # 4. 构建压缩比序列
    if os.path.exists(AUCTION_CACHE_FILE):
        cache = pd.read_csv(AUCTION_CACHE_FILE)
        # 用缓存中的PD值构建历史序列
        if "primary_dealer_award_pct" in cache.columns:
            pd_hist = float(cache["primary_dealer_award_pct"].iloc[0])
            compression = pd.Series([pd_hist / (b + 0.001) for b in basis])
        else:
            compression = pd.Series([current_pd / (b + 0.001) for b in basis])
    else:
        compression = pd.Series([current_pd / (b + 0.001) for b in basis])

    # 5. 期限溢价
    if not sofr.empty:
        term_premium = sofr["value"].pct_change() * 100
    else:
        term_premium = pd.Series([0] * len(basis))

    # 6. 基线标定
    comp_clean = compression.dropna()
    if len(comp_clean) < 20:
        logger.warning("数据不足，使用默认基线")
        mu0, sigma0 = 3.5, 0.15
    else:
        mu0, sigma0 = compute_baseline_stats(comp_clean.iloc[:int(len(comp_clean)*0.8)])
    logger.info(f"基线: mu0={mu0:.4f}, sigma0={sigma0:.4f}")

    # 7. CUSUM状态
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

    # 8. 记录状态
    state = {
        "timestamp": datetime.now().isoformat(),
        "compression": current_comp,
        "term_premium": current_term,
        "s_t": s_t,
        "mu0": mu0,
        "sigma0": sigma0,
        "layer1": layer1_triggered,
        "triggered": triggered,
        "primary_dealer_pct": current_pd
    }
    df_new = pd.DataFrame([state])
    if os.path.exists(STATE_FILE):
        existing = pd.read_csv(STATE_FILE)
        df_new = pd.concat([existing, df_new], ignore_index=True)
    df_new.to_csv(STATE_FILE, index=False)

    # 9. 触发判断
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

    logger.info("=" * 60)

if __name__ == "__main__":
    if os.path.exists(T0_FILE):
        logger.info("T₀已记录，如需重新实验请删除 t0_trigger.json")
    else:
        run_experiment()