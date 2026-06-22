import os
import requests
import urllib3
from dotenv import load_dotenv

# 关闭 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

FRED_API_KEY = os.environ.get("FRED_API_KEY")

if not FRED_API_KEY:
    print("❌ 未找到 FRED_API_KEY，请在 .env 文件中设置")
    exit(1)

url = "https://api.stlouisfed.org/fred/series/observations"
params = {
    "series_id": "DGS10",
    "api_key": FRED_API_KEY,
    "file_type": "json",
    "limit": 3,
    "sort_order": "desc"
}

try:
    # 加上 verify=False 跳过 SSL 验证
    resp = requests.get(url, params=params, timeout=10, verify=False)
    resp.raise_for_status()
    data = resp.json()
    obs = data.get("observations", [])
    if obs:
        print("✅ FRED 连接成功！最新 DGS10 数据：")
        for o in obs[:3]:
            print(f"  {o['date']}: {o['value']}")
    else:
        print("⚠️ 返回数据为空")
except Exception as e:
    print(f"❌ 请求失败: {e}")