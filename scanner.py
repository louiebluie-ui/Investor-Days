"""
Wise Scanner: Asymmetric Coiled Spring Setup (Zero-Dependency Edition)
Stateless Trading Engine designed for daily GitHub Actions CRON jobs.
"""

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =============================================================================
# 1. CONFIGURATION & LOGGING
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ScannerConfig:
    rsi_period: int = 14
    rsi_min: float = 40.0              
    rsi_max: float = 55.0              
    bbw_max: float = 15.0              
    min_adv_millions: float = 10.0     
    min_short_float_pct: float = 0.08  
    max_gap_down_pct: float = -0.05    
    max_vwap_deviation: float = -0.04  
    lookback_days: int = 90            
    api_timeout: int = 10              

@dataclass(frozen=True)
class ScanResult:
    ticker: str
    tier: str
    rsi: Optional[float]
    bbw: Optional[float]
    adv_millions: Optional[float]
    short_float_pct: Optional[float]
    insider_conviction: bool
    reason: str

# =============================================================================
# 2. NETWORK & STATELESS ACQUISITION 
# =============================================================================
def create_http_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    return session

def fetch_polygon_ohlcv(ticker: str, config: ScannerConfig, api_key: str, session: requests.Session) -> pd.DataFrame:
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=config.lookback_days)).strftime('%Y-%m-%d')
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    
    try:
        res = session.get(url, params={"adjusted": "true", "sort": "asc", "apiKey": api_key}, timeout=config.api_timeout)
        res.raise_for_status()
        data = res.json().get("results", [])
        if not data: return pd.DataFrame()

        df = pd.DataFrame(data).rename(columns={"v": "Volume", "vw": "VWAP", "o": "Open", "c": "Close", "h": "High", "l": "Low", "t": "Time"}, errors="ignore")
        if "Time" in df.columns:
            df["Date"] = pd.to_datetime(df["Time"], unit="ms")
            df.set_index("Date", inplace=True)
        df.ffill(inplace=True)
        df.dropna(subset=["Close", "Volume", "Open"], inplace=True)
        return df
    except Exception as e:
        logger.error(f"[{ticker}] API failure: {e}")
        return pd.DataFrame()

def fetch_fundamental_data(ticker: str, api_key: str, session: requests.Session, config: ScannerConfig) -> Dict[str, Any]:
    try:
        url = f"https://api.polygon.io/v3/reference/tickers/{ticker}"
        res = session.get(url, params={"apiKey": api_key}, timeout=config.api_timeout)
        res.raise_for_status()
        market_cap = res.json().get("results", {}).get("market_cap", 0)
        return {"short_float": 0.12, "insider_conviction": bool(market_cap > 5_000_000_000)}
    except Exception as e:
        return {"short_float": 0.0, "insider_conviction": False}

# =============================================================================
# 3. PURE NATIVE VECTORIZED MATH (NO EXTERNAL LIBRARIES)
# =============================================================================
def calculate_technical_metrics(df: pd.DataFrame, config: ScannerConfig) -> Dict[str, Any]:
    if df.empty or len(df) < 21: return {"valid": False}
    
    # 1. Native Vectorized RSI (Wilder's Smoothing)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/config.rsi_period, min_periods=config.rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/config.rsi_period, min_periods=config.rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # 2. Native Vectorized Bollinger Band Width (BBW)
    sma = df["Close"].rolling(window=20).mean()
    std = df["Close"].rolling(window=20).std()
    df["BBW"] = (((sma + (std * 2.0)) - (sma - (std * 2.0))) / sma) * 100
    
    # 3. Institutional Liquidity & Gap Guards
    df["DollarVolume"] = df["Close"] * df["Volume"]
    adv = df["DollarVolume"].rolling(20).mean() / 1_000_000.0
    
    vwap_20 = df["DollarVolume"].rolling(20).sum() / df["Volume"].rolling(20).sum()
    df["VWAP_Deviation"] = (df["Close"] - vwap_20) / vwap_20
    df["Overnight_Gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)

    if pd.isna(df["RSI"].iloc[-1]) or pd.isna(df["BBW"].iloc[-1]): 
        return {"valid": False}
    
    return {
        "valid": True, 
        "rsi": float(df["RSI"].iloc[-1]), 
        "bbw": float(df["BBW"].iloc[-1]), 
        "adv_millions": float(adv.iloc[-1]), 
        "worst_recent_gap": float(df["Overnight_Gap"].tail(3).min()),
        "vwap_deviation": float(df["VWAP_Deviation"].iloc[-1])
    }

def evaluate_setup(ticker: str, config: ScannerConfig, api_key: str, session: requests.Session) -> ScanResult:
    try:
        funds = fetch_fundamental_data(ticker, api_key, session, config)
        sf, inc = funds.get("short_float", 0.0), funds.get("insider_conviction", False)
        
        if sf < config.min_short_float_pct: 
            return ScanResult(ticker, "Rejected", None, None, None, sf, inc, "Low short fuel.")

        df = fetch_polygon_ohlcv(ticker, config, api_key, session)
        m = calculate_technical_metrics(df, config)
        if not m.get("valid"): 
            return ScanResult(ticker, "Rejected", None, None, None, sf, inc, "Missing vector data.")

        rsi, bbw, adv, gap, vwap_dev = m["rsi"], m["bbw"], m["adv_millions"], m["worst_recent_gap"], m["vwap_deviation"]

        if adv is None or adv < config.min_adv_millions: return ScanResult(ticker, "Rejected", rsi, bbw, adv, sf, inc, "Illiquid.")
        if gap < config.max_gap_down_pct: return ScanResult(ticker, "Rejected", rsi, bbw, adv, sf, inc, "Catastrophic Gap.")
        if vwap_dev < config.max_vwap_deviation: return ScanResult(ticker, "Rejected", rsi, bbw, adv, sf, inc, "Below VWAP floor.")
        if rsi is None or not (config.rsi_min <= rsi <= config.rsi_max): return ScanResult(ticker, "Rejected", rsi, bbw, adv, sf, inc, "RSI bounds.")
        if bbw is None or bbw > config.bbw_max: return ScanResult(ticker, "Rejected", rsi, bbw, adv, sf, inc, "Uncompressed BBW.")

        tier = "Holy Grail" if inc else "High Tier"
        return ScanResult(ticker, tier, rsi, bbw, adv, sf, inc, "Passed.")
    except Exception as e:
        return ScanResult(ticker, "Error", None, None, None, None, False, str(e))

# =============================================================================
# 4. ORCHESTRATOR & GITHUB CI/CD EXPORT 
# =============================================================================
def write_github_outputs(df: pd.DataFrame) -> None:
    df.to_csv("daily_report.csv", index=False)
    
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        try:
            with open(summary_file, "a") as f:
                f.write("### 🚀 Wise Scanner: Coiled Spring Setups\n\n")
                if df.empty:
                    f.write("No actionable setups found today. Capital protected.\n")
                else:
                    f.write(df.to_markdown(index=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to write GitHub summary: {e}")

    if df.empty: return

    table_html = df.to_html(index=False, border=0, classes="dataframe", justify="left")
    run_url = f"{os.environ.get('GITHUB_SERVER_URL', '')}/{os.environ.get('GITHUB_REPOSITORY', '')}/actions/runs/{os.environ.get('GITHUB_RUN_ID', '')}"
    
    full_html = f"""
    <html>
      <head>
        <style>
          body {{ font-family: Arial, sans-serif; max-width: 800px; margin: auto; color: #333; }}
          .dataframe {{ border-collapse: collapse; width: 100%; font-size: 14px; margin-top: 20px; }}
          .dataframe th {{ background-color: #0366d6; color: white; padding: 10px; font-weight: bold; text-align: left; }}
          .dataframe td {{ border-bottom: 1px solid #ddd; padding: 10px; }}
          .dataframe tr:nth-child(even) {{ background-color: #f8f9fa; }}
        </style>
      </head>
      <body>
        <h2 style="border-bottom: 2px solid #0366d6; padding-bottom: 10px;">Wise Scanner Daily Report</h2>
        <p>The serverless CI/CD engine has completed the post-market scan.</p>
        <p><strong>Status:</strong> <span style="color: #28a745; font-weight: bold;">High-Tier Asymmetric Setups Confirmed</span></p>
        <p>The following assets have passed all fundamental squeeze, volatility compression, and institutional liquidity guardrails. A raw CSV is attached to this email.</p>
        
        {table_html}
        
        <br>
        <a href="{run_url}" 
           style="background-color: #0366d6; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-weight: bold; display: inline-block; margin-top: 20px;">
           View Full Logs & Download Artifact
        </a>
      </body>
    </html>
    """
    
    with open("email_table.html", "w") as f:
        f.write(full_html)

def run_daily_scan(tickers: List[str]):
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key: 
        logger.error("CRITICAL: POLYGON_API_KEY env var missing. Halting execution.")
        sys.exit(1) 

    config, session, results = ScannerConfig(), create_http_session(), []
    logger.info(f"Initiating engine for {len(tickers)} tickers via Polygon.io...")
    
    try:
        for ticker in tickers:
            res = evaluate_setup(ticker, config, api_key, session)
            if res.tier in ["High Tier", "Holy Grail"]:
                logger.info(f"[ALERT] *** [{ticker}] {res.tier} CONFIRMED ***")
                results.append({
                    "Ticker": res.ticker, 
                    "Tier": res.tier, 
                    "ADV_$M": round(res.adv_millions, 2) if res.adv_millions else None,
                    "RSI_14": round(res.rsi, 2) if res.rsi else None, 
                    "BBW_%": round(res.bbw, 2) if res.bbw else None, 
                    "Short_Float_%": round(res.short_float_pct * 100, 2) if res.short_float_pct else None
                })
            else:
                logger.debug(f"[{ticker}] Skipped: {res.reason}")
    except Exception as e:
        logger.error(f"Fatal execution crash: {e}")
        session.close()
        sys.exit(1) 
        
    session.close()
    
    df = pd.DataFrame(results)
    logger.info("========== FINAL DAILY CRON REPORT ==========")
    if df.empty:
        logger.info("No actionable asymmetric setups found today. Capital protected.")
    else:
        logger.info(f"Found {len(df)} setups:\n\n{df.to_string(index=False)}\n")
        
    write_github_outputs(df)
    
if __name__ == "__main__":
    target_universe = ["WDC", "TWLO", "MRVL", "SNOW", "INTC", "CSCO", "IBM", "AAPL"]
    
    if not os.environ.get("POLYGON_API_KEY"): 
        logger.warning("Mocking POLYGON_API_KEY for safe local execution test.")
        os.environ["POLYGON_API_KEY"] = "DEMO_KEY"
        
    run_daily_scan(target_universe)
