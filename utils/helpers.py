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

