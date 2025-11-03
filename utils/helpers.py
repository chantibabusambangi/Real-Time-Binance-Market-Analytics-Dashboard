"""Helper utility functions"""

import pandas as pd
import numpy as np
from datetime import datetime


def timestamp_to_datetime(timestamp: float) -> datetime:
    """Convert Unix timestamp to datetime"""
    return datetime.fromtimestamp(timestamp)


def format_number(num: float, decimals: int = 2) -> str:
    """Format number with commas"""
    return f"{num:,.{decimals}f}"


def calculate_percentage_change(old: float, new: float) -> float:
    """Calculate percentage change"""
    if old == 0:
        return 0
    return ((new - old) / old) * 100
```

### 7. **config/__init__.py** - Empty file

### 8. **backend/__init__.py** - Empty file

### 9. **frontend/__init__.py** - Empty file

### 10. **.gitignore**
```
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
data/*.db
data/raw_ticks/*
data/resampled/*
logs/*.log
.env
.DS_Store
.vscode/
.idea/
```

Now your project structure should be:
```
trading-analytics/
├── app.py ✅
├── requirements.txt ✅
├── README.md ✅
├── config.py ✅
├── .gitignore ✅
├── config/
│   └── __init__.py ✅
├── backend/
│   ├── __init__.py ✅
│   ├── analytics.py ✅ (NEW - copy code above)
│   ├── data_ingestion.py ✅ (from previous artifact)
│   ├── database.py ✅ (from previous artifact)
│   ├── alerts.py ✅ (NEW - copy code above)
│   └── api.py ✅ (NEW - copy code above)
├── frontend/
│   ├── __init__.py ✅
│   └── dashboard.py ✅ (from previous artifact)
├── utils/
│   ├── __init__.py ✅ (NEW - copy code above)
│   ├── helpers.py ✅ (NEW - copy code above)
│   └── logger.py ✅ (from previous artifact)
└── data/ (auto-created)
