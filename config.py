import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Database configuration
DATABASE_URL = f"sqlite:///{DATA_DIR}/trading_data.db"

# Redis configuration (optional, will fallback to in-memory if not available)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
USE_REDIS = os.getenv("USE_REDIS", "false").lower() == "true"

# Binance WebSocket configuration
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws"

# Default symbols for trading pairs
DEFAULT_SYMBOLS = ["btcusdt", "ethusdt", "bnbusdt", "adausdt", "solusdt"]

# Sampling intervals (in seconds)
SAMPLING_INTERVALS = {
    "1s": 1,
    "1m": 60,
    "5m": 300
}

# Analytics configuration
DEFAULT_WINDOW_SIZE = 20
DEFAULT_ZSCORE_THRESHOLD = 2.0
ADF_SIGNIFICANCE_LEVEL = 0.05

# Alert configuration
ALERT_CHECK_INTERVAL = 1  # seconds

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

# Frontend configuration
STREAMLIT_PORT = 8501
UPDATE_INTERVAL = 500  # milliseconds for live updates
