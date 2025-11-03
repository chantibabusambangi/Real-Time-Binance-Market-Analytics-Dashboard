"""
Complete Analytics Engine
Combines all analytics modules into one comprehensive file
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class AnalyticsEngine:
    """
    Comprehensive analytics engine for trading
    Combines all statistical analysis in one place
    """
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.cache = {}
        self.cache_duration = 60
    
    # ==================== BASIC STATISTICS ====================
    
    @staticmethod
    def compute_price_stats(prices: pd.Series) -> Dict:
        """Compute comprehensive price statistics"""
        if len(prices) == 0:
            return {}
        
        return {
            'mean': float(prices.mean()),
            'median': float(prices.median()),
            'std': float(prices.std()),
            'min': float(prices.min()),
            'max': float(prices.max()),
            'range': float(prices.max() - prices.min()),
            'current': float(prices.iloc[-1]),
            'first': float(prices.iloc[0]),
            'change': float(prices.iloc[-1] - prices.iloc[0]),
            'change_pct': float((prices.iloc[-1] / prices.iloc[0] - 1) * 100) if prices.iloc[0] != 0 else 0,
            'skewness': float(prices.skew()),
            'kurtosis': float(prices.kurtosis()),
            'count': len(prices)
        }
    
    @staticmethod
    def compute_returns(prices: pd.Series, log_returns: bool = True) -> pd.Series:
        """Compute returns (log or simple)"""
        if log_returns:
            return np.log(prices / prices.shift(1))
        else:
            return prices.pct_change()
    
    @staticmethod
    def compute_volatility(returns: pd.Series, window: int = 20, 
                          annualize: bool = True) -> pd.Series:
        """Compute rolling volatility"""
        vol = returns.rolling(window=window).std()
        
        if annualize:
            # For 1-minute data: sqrt(525600 minutes/year)
            vol = vol * np.sqrt(525600)
        
        return vol
    
    @staticmethod
    def compute_sharpe_ratio(returns: pd.Series, window: int = 20,
                            risk_free_rate: float = 0.02) -> pd.Series:
        """Compute rolling Sharpe ratio"""
        excess_returns = returns - risk_free_rate / 525600
        sharpe = excess_returns.rolling(window).mean() / returns.rolling(window).std()
        return sharpe * np.sqrt(525600)
    
    # ==================== REGRESSION ANALYSIS ====================
    
    @staticmethod
    def ols_regression(y: pd.Series, x: pd.Series, 
                      window: Optional[int] = None) -> Dict:
        """
        OLS regression: y = alpha + beta * x
        Returns hedge ratio and statistics
        """
        df = pd.DataFrame({'y': y, 'x': x}).dropna()
        
        if len(df) < 2:
            return {
                'beta': np.nan, 'alpha': np.nan, 
                'r_squared': np.nan, 'p_value': np.nan
            }
        
        if window and len(df) > window:
            df = df.iloc[-window:]
        
        try:
            X = add_constant(df['x'])
            model = OLS(df['y'], X).fit()
            
            return {
                'beta': float(model.params['x']),
                'alpha': float(model.params['const']),
                'r_squared': float(model.rsquared),
                'r_squared_adj': float(model.rsquared_adj),
                'std_err': float(model.bse['x']),
                't_stat': float(model.tvalues['x']),
                'p_value': float(model.pvalues['x']),
                'observations': int(model.nobs)
            }
        except Exception as e:
            logger.error(f"OLS regression error: {e}")
            return {
                'beta': np.nan, 'alpha': np.nan, 
                'r_squared': np.nan, 'error': str(e)
            }
    
    @staticmethod
    def rolling_regression(y: pd.Series, x: pd.Series, 
                          window: int = 20) -> pd.DataFrame:
        """Compute rolling OLS regression"""
        results = []
        
        for i in range(window, len(y) + 1):
            y_window = y.iloc[i-window:i]
            x_window = x.iloc[i-window:i]
            
            reg = AnalyticsEngine.ols_regression(y_window, x_window)
            reg['timestamp'] = y.index[i-1] if hasattr(y, 'index') else i-1
            results.append(reg)
        
        return pd.DataFrame(results)
    
    @staticmethod
    def compute_spread(y: pd.Series, x: pd.Series, 
                      hedge_ratio: float) -> pd.Series:
        """Compute spread: y - beta * x"""
        return y - hedge_ratio * x
    
    @staticmethod
    def compute_zscore(series: pd.Series, window: int = 20) -> pd.Series:
        """Compute rolling z-score"""
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        return (series - rolling_mean) / rolling_std
    
    # ==================== STATIONARITY TESTS ====================
    
    @staticmethod
    def adf_test(series: pd.Series, maxlag: Optional[int] = None) -> Dict:
        """
        Augmented Dickey-Fuller test for stationarity
        H0: Series has unit root (non-stationary)
        H1: Series is stationary
        """
        series_clean = series.dropna()
        
        if len(series_clean) < 3:
            return {
                'adf_statistic': np.nan,
                'p_value': np.nan,
                'is_stationary': False
            }
        
        try:
            result = adfuller(series_clean, maxlag=maxlag, autolag='AIC')
            
            return {
                'adf_statistic': float(result[0]),
                'p_value': float(result[1]),
                'used_lag': int(result[2]),
                'n_observations': int(result[3]),
                'critical_values': {
                    '1%': float(result[4]['1%']),
                    '5%': float(result[4]['5%']),
                    '10%': float(result[4]['10%'])
                },
                'is_stationary': result[1] < 0.05,
                'interpretation': 'Stationary' if result[1] < 0.05 else 'Non-stationary'
            }
        except Exception as e:
            logger.error(f"ADF test error: {e}")
            return {
                'adf_statistic': np.nan,
                'p_value': np.nan,
                'is_stationary': False,
                'error': str(e)
            }
    
    @staticmethod
    def compute_half_life(series: pd.Series) -> float:
        """Compute half-life of mean reversion"""
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
            return float(half_life)
        except Exception:
            return np.nan
    
    # ==================== CORRELATION ANALYSIS ====================
    
    @staticmethod
    def compute_correlation(x: pd.Series, y: pd.Series,
                           window: Optional[int] = None,
                           method: str = 'pearson') -> float:
        """Compute correlation coefficient"""
        df = pd.DataFrame({'x': x, 'y': y}).dropna()
        
        if len(df) < 2:
            return np.nan
        
        if window and len(df) > window:
            df = df.iloc[-window:]
        
        return float(df['x'].corr(df['y'], method=method))
    
    @staticmethod
    def rolling_correlation(x: pd.Series, y: pd.Series, 
                           window: int = 20) -> pd.Series:
        """Compute rolling correlation"""
        df = pd.DataFrame({'x': x, 'y': y})
        return df['x'].rolling(window=window).corr(df['y'])
    
    @staticmethod
    def correlation_matrix(data: Dict[str, pd.Series],
                          window: Optional[int] = None) -> pd.DataFrame:
        """Compute correlation matrix for multiple series"""
        df = pd.DataFrame(data).dropna()
        
        if window and len(df) > window:
            df = df.iloc[-window:]
        
        return df.corr()
    
    # ==================== CREATIVE ANALYTICS ====================
    
    @staticmethod
    def momentum_indicators(prices: pd.Series, window: int = 14) -> Dict:
        """Compute momentum-based indicators"""
        returns = prices.pct_change()
        
        # Rate of Change
        roc = (prices / prices.shift(window) - 1) * 100
        
        # RSI-like momentum
        gains = returns.clip(lower=0)
        losses = -returns.clip(upper=0)
        
        avg_gain = gains.rolling(window).mean()
        avg_loss = losses.rolling(window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return {
            'roc': float(roc.iloc[-1]) if len(roc) > 0 else np.nan,
            'momentum_score': float(rsi.iloc[-1]) if len(rsi) > 0 else np.nan,
            'trend_strength': float(abs(roc.iloc[-1])) if len(roc) > 0 else np.nan
        }
    
    @staticmethod
    def mean_reversion_signals(spread: pd.Series, window: int = 20) -> Dict:
        """Generate mean reversion trading signals"""
        zscore = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
        
        current_z = zscore.iloc[-1] if len(zscore) > 0 else 0
        
        if current_z > 2:
            signal = 'SHORT'
        elif current_z < -2:
            signal = 'LONG'
        elif abs(current_z) < 0.5:
            signal = 'EXIT'
        else:
            signal = 'HOLD'
        
        return {
            'zscore': float(current_z),
            'signal': signal,
            'strength': float(abs(current_z)),
            'distance_from_mean': float(spread.iloc[-1] - spread.mean())
        }
    
    @staticmethod
    def volume_profile(data: pd.DataFrame, price_bins: int = 20) -> Dict:
        """Compute volume profile analysis"""
        if 'close' not in data.columns or 'volume' not in data.columns:
            return {}
        
        price_range = data['close'].max() - data['close'].min()
        if price_range == 0:
            return {}
        
        bin_size = price_range / price_bins
        data['price_bin'] = ((data['close'] - data['close'].min()) / bin_size).astype(int)
        
        volume_profile = data.groupby('price_bin')['volume'].sum()
        
        if len(volume_profile) == 0:
            return {}
        
        poc_bin = volume_profile.idxmax()
        poc_price = data['close'].min() + (poc_bin + 0.5) * bin_size
        
        return {
            'poc_price': float(poc_price),
            'total_volume': float(data['volume'].sum()),
            'avg_volume': float(data['volume'].mean()),
            'volume_std': float(data['volume'].std())
        }
