import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import logging
from typing import Dict, Tuple, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalyticsEngine:
    """
    Computes various trading analytics including statistical tests,
    hedge ratios, spreads, and correlations.
    """
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.cache = {}
        self.cache_duration = 60  # seconds
        
    def compute_price_stats(self, prices: pd.Series) -> Dict:
        """Compute basic price statistics"""
        if len(prices) == 0:
            return {}
        
        return {
            'mean': float(prices.mean()),
            'std': float(prices.std()),
            'min': float(prices.min()),
            'max': float(prices.max()),
            'current': float(prices.iloc[-1]),
            'change': float(prices.iloc[-1] - prices.iloc[0]) if len(prices) > 1 else 0,
            'change_pct': float((prices.iloc[-1] / prices.iloc[0] - 1) * 100) if len(prices) > 1 and prices.iloc[0] != 0 else 0
        }
    
    def compute_returns(self, prices: pd.Series, log_returns: bool = True) -> pd.Series:
        """Compute returns"""
        if log_returns:
            return np.log(prices / prices.shift(1))
        else:
            return prices.pct_change()
    
    def compute_volatility(self, returns: pd.Series, window: int = 20, 
                          annualize: bool = True) -> pd.Series:
        """Compute rolling volatility"""
        vol = returns.rolling(window=window).std()
        
        if annualize:
            # Assuming crypto trades 24/7, use appropriate annualization factor
            # For 1-minute data: sqrt(525600), for 1-hour: sqrt(8760)
            vol = vol * np.sqrt(525600)  # Annualized for minute data
        
        return vol
    
    def compute_hedge_ratio_ols(self, y: pd.Series, x: pd.Series, 
                                window: Optional[int] = None) -> Dict:
        """
        Compute hedge ratio using OLS regression.
        y = alpha + beta * x + epsilon
        """
        if len(y) < 2 or len(x) < 2:
            return {'beta': np.nan, 'alpha': np.nan, 'r_squared': np.nan}
        
        # Align series
        df = pd.DataFrame({'y': y, 'x': x}).dropna()
        
        if len(df) < 2:
            return {'beta': np.nan, 'alpha': np.nan, 'r_squared': np.nan}
        
        if window is not None and len(df) > window:
            df = df.iloc[-window:]
        
        try:
            X = add_constant(df['x'])
            model = OLS(df['y'], X).fit()
            
            return {
                'beta': float(model.params['x']),
                'alpha': float(model.params['const']),
                'r_squared': float(model.rsquared),
                'std_err': float(model.bse['x']),
                't_stat': float(model.tvalues['x']),
                'p_value': float(model.pvalues['x'])
            }
        except Exception as e:
            logger.error(f"Error in OLS regression: {e}")
            return {'beta': np.nan, 'alpha': np.nan, 'r_squared': np.nan}
    
    def compute_rolling_hedge_ratio(self, y: pd.Series, x: pd.Series, 
                                   window: int = 20) -> pd.DataFrame:
        """Compute rolling hedge ratio"""
        results = []
        
        for i in range(window, len(y) + 1):
            y_window = y.iloc[i-window:i]
            x_window = x.iloc[i-window:i]
            
            ratio = self.compute_hedge_ratio_ols(y_window, x_window)
            ratio['timestamp'] = y.index[i-1]
            results.append(ratio)
        
        return pd.DataFrame(results)
    
    def compute_spread(self, y: pd.Series, x: pd.Series, 
                      hedge_ratio: float) -> pd.Series:
        """Compute spread: y - beta * x"""
        return y - hedge_ratio * x
    
    def compute_zscore(self, series: pd.Series, window: int = 20) -> pd.Series:
        """Compute rolling z-score"""
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        
        zscore = (series - rolling_mean) / rolling_std
        return zscore
    
    def compute_adf_test(self, series: pd.Series) -> Dict:
        """
        Augmented Dickey-Fuller test for stationarity.
        H0: Series has a unit root (non-stationary)
        H1: Series is stationary
        """
        if len(series.dropna()) < 3:
            return {
                'adf_statistic': np.nan,
                'p_value': np.nan,
                'is_stationary': False,
                'critical_values': {}
            }
        
        try:
            result = adfuller(series.dropna(), autolag='AIC')
            
            return {
                'adf_statistic': float(result[0]),
                'p_value': float(result[1]),
                'used_lag': int(result[2]),
                'n_obs': int(result[3]),
                'critical_values': {k: float(v) for k, v in result[4].items()},
                'is_stationary': result[1] < 0.05  # 5% significance level
            }
        except Exception as e:
            logger.error(f"Error in ADF test: {e}")
            return {
                'adf_statistic': np.nan,
                'p_value': np.nan,
                'is_stationary': False,
                'critical_values': {}
            }
    
    def compute_correlation(self, x: pd.Series, y: pd.Series, 
                           window: Optional[int] = None, 
                           method: str = 'pearson') -> float:
        """Compute correlation between two series"""
        df = pd.DataFrame({'x': x, 'y': y}).dropna()
        
        if len(df) < 2:
            return np.nan
        
        if window is not None and len(df) > window:
            df = df.iloc[-window:]
        
        return float(df['x'].corr(df['y'], method=method))
    
    def compute_rolling_correlation(self, x: pd.Series, y: pd.Series, 
                                   window: int = 20) -> pd.Series:
        """Compute rolling correlation"""
        df = pd.DataFrame({'x': x, 'y': y})
        return df['x'].rolling(window=window).corr(df['y'])
    
    def compute_half_life(self, series: pd.Series) -> float:
        """
        Compute half-life of mean reversion using AR(1) model.
        Useful for pairs trading.
        """
        lag = series.shift(1)
        delta = series - lag
        
        df = pd.DataFrame({'delta': delta, 'lag': lag}).dropna()
        
        if len(df) < 2:
            return np.nan
        
        try:
            X = add_constant(df['lag'])
            model = OLS(df['delta'], X).fit()
            
            lambda_param = model.params['lag']
            
            if lambda_param >= 0:
                return np.inf
            
            half_life = -np.log(2) / lambda_param
