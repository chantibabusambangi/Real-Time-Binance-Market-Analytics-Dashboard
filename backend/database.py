"""
Database manager for tick data, OHLCV, and analytics storage
Designed for easy migration to TimescaleDB or InfluxDB
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from threading import Lock
import json
import numpy as np

from utils.logger import setup_logger

logger = setup_logger(__name__)


class DatabaseManager:
    """
    Manages all database operations with focus on:
    - High-frequency tick data storage
    - OHLCV resampled data
    - Analytics results caching
    - Alert management
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        logger.info(f"Database initialized at {db_path}")
    
    def _get_connection(self):
        """Get database connection with optimizations"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        # Performance optimizations
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA temp_store=MEMORY")
        return conn
    
    def _init_database(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Tick data table - high frequency data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tick_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    is_buyer_maker INTEGER,
                    trade_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_tick_symbol_time 
                ON tick_data(symbol, timestamp DESC)
            """)
            
            # OHLCV resampled data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    trade_count INTEGER DEFAULT 0,
                    vwap REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(timestamp, symbol, interval)
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_interval_time 
                ON ohlcv_data(symbol, interval, timestamp DESC)
            """)
            
            # Analytics results cache
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analytics_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    symbol_pair TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    metric_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_analytics_pair_metric 
                ON analytics_cache(symbol_pair, metric_type, timestamp DESC)
            """)
            
            # Alerts configuration
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    symbol_pair TEXT NOT NULL,
                    condition TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    is_active INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_triggered TIMESTAMP
                )
            """)
            
            # Alert history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alert_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT NOT NULL,
                    triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    value REAL NOT NULL,
                    message TEXT,
                    FOREIGN KEY (alert_id) REFERENCES alerts(alert_id)
                )
            """)
            
            conn.commit()
            logger.info("Database schema initialized")
    
    # ========== TICK DATA OPERATIONS ==========
    
    def insert_tick_batch(self, ticks: List[Dict]):
        """Batch insert tick data for performance"""
        if not ticks:
            return
        
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT INTO tick_data 
                    (timestamp, symbol, price, quantity, is_buyer_maker, trade_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, [
                    (t['timestamp'], t['symbol'], t['price'], t['quantity'],
                     t.get('is_buyer_maker', 0), t.get('trade_id'))
                    for t in ticks
                ])
                conn.commit()
        
        logger.debug(f"Inserted {len(ticks)} ticks")
    
    def get_tick_data(self, symbol: str, start_time: Optional[float] = None,
                     end_time: Optional[float] = None, limit: int = 10000) -> pd.DataFrame:
        """Retrieve tick data"""
        query = "SELECT * FROM tick_data WHERE symbol = ?"
        params = [symbol.upper()]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def get_recent_ticks(self, symbol: str, seconds: int = 60) -> pd.DataFrame:
        """Get ticks from last N seconds"""
        cutoff = datetime.now().timestamp() - seconds
        return self.get_tick_data(symbol, start_time=cutoff)
    
    # ========== OHLCV OPERATIONS ==========
    
    def insert_ohlcv_batch(self, ohlcv_list: List[Dict]):
        """Batch insert OHLCV data"""
        if not ohlcv_list:
            return
        
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT OR REPLACE INTO ohlcv_data 
                    (timestamp, symbol, interval, open, high, low, close, volume, trade_count, vwap)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    (o['timestamp'], o['symbol'], o['interval'], o['open'],
                     o['high'], o['low'], o['close'], o['volume'],
                     o.get('trade_count', 0), o.get('vwap'))
                    for o in ohlcv_list
                ])
                conn.commit()
        
        logger.debug(f"Inserted {len(ohlcv_list)} OHLCV records")
    
    def get_ohlcv_data(self, symbol: str, interval: str,
                       start_time: Optional[float] = None,
                       limit: int = 1000) -> pd.DataFrame:
        """Retrieve OHLCV data"""
        query = "SELECT * FROM ohlcv_data WHERE symbol = ? AND interval = ?"
        params = [symbol.upper(), interval]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        return df
    
    def get_latest_ohlcv(self, symbol: str, interval: str) -> Optional[Dict]:
        """Get most recent OHLCV bar"""
        query = """
            SELECT * FROM ohlcv_data 
            WHERE symbol = ? AND interval = ?
            ORDER BY timestamp DESC LIMIT 1
        """
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            result = cursor.execute(query, [symbol.upper(), interval]).fetchone()
        
        if result:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, result))
        return None
    
    def upload_ohlc_file(self, df: pd.DataFrame, symbol: str, interval: str):
        """Upload OHLC data from CSV/DataFrame"""
        records = []
        
        for idx, row in df.iterrows():
            # Handle different timestamp formats
            if 'timestamp' in row:
                ts = row['timestamp']
            elif isinstance(idx, pd.Timestamp):
                ts = idx.timestamp()
            else:
                ts = pd.Timestamp(idx).timestamp()
            
            records.append({
                'timestamp': ts,
                'symbol': symbol.upper(),
                'interval': interval,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
                'trade_count': int(row.get('trade_count', 0)),
                'vwap': float(row['vwap']) if 'vwap' in row and pd.notna(row['vwap']) else None
            })
        
        self.insert_ohlcv_batch(records)
        logger.info(f"Uploaded {len(records)} OHLCV records for {symbol}")
    
    # ========== ANALYTICS CACHE ==========
    
    def save_analytics(self, symbol_pair: str, interval: str,
                      timestamp: float, metric_type: str, data: Dict):
        """Save analytics results"""
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO analytics_cache 
                    (timestamp, symbol_pair, interval, metric_type, metric_data)
                    VALUES (?, ?, ?, ?, ?)
                """, (timestamp, symbol_pair, interval, metric_type, json.dumps(data)))
                conn.commit()
    
    def get_analytics(self, symbol_pair: str, metric_type: str,
                     limit: int = 100) -> pd.DataFrame:
        """Retrieve analytics results"""
        query = """
            SELECT * FROM analytics_cache 
            WHERE symbol_pair = ? AND metric_type = ?
            ORDER BY timestamp DESC LIMIT ?
        """
        
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=[symbol_pair, metric_type, limit])
        
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
            df['metric_data'] = df['metric_data'].apply(json.loads)
        
        return df
    
    # ========== ALERT OPERATIONS ==========
    
    def create_alert(self, alert_data: Dict) -> str:
        """Create new alert"""
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO alerts 
                    (alert_id, name, symbol_pair, condition, threshold, is_active)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    alert_data['alert_id'],
                    alert_data['name'],
                    alert_data['symbol_pair'],
                    alert_data['condition'],
                    alert_data['threshold'],
                    alert_data.get('is_active', 1)
                ))
                conn.commit()
        
        logger.info(f"Alert created: {alert_data['alert_id']}")
        return alert_data['alert_id']
    
    def get_active_alerts(self) -> List[Dict]:
        """Get all active alerts"""
        query = "SELECT * FROM alerts WHERE is_active = 1"
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            results = cursor.execute(query).fetchall()
            columns = [desc[0] for desc in cursor.description]
        
        return [dict(zip(columns, row)) for row in results]
    
    def log_alert_trigger(self, alert_id: str, value: float, message: str):
        """Log alert trigger"""
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO alert_history (alert_id, value, message)
                    VALUES (?, ?, ?)
                """, (alert_id, value, message))
                
                cursor.execute("""
                    UPDATE alerts SET last_triggered = CURRENT_TIMESTAMP
                    WHERE alert_id = ?
                """, (alert_id,))
                
                conn.commit()
    
    def delete_alert(self, alert_id: str):
        """Delete alert"""
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM alerts WHERE alert_id = ?", (alert_id,))
                conn.commit()
    
    # ========== UTILITY OPERATIONS ==========
    
    def get_available_symbols(self) -> List[str]:
        """Get list of symbols with data"""
        query = "SELECT DISTINCT symbol FROM ohlcv_data ORDER BY symbol"
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            results = cursor.execute(query).fetchall()
        
        return [row[0] for row in results]
    
    def get_data_summary(self) -> Dict:
        """Get summary statistics of stored data"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            tick_count = cursor.execute("SELECT COUNT(*) FROM tick_data").fetchone()[0]
            ohlcv_count = cursor.execute("SELECT COUNT(*) FROM ohlcv_data").fetchone()[0]
            symbols = cursor.execute("SELECT COUNT(DISTINCT symbol) FROM ohlcv_data").fetchone()[0]
            
            oldest_tick = cursor.execute(
                "SELECT MIN(timestamp) FROM tick_data"
            ).fetchone()[0]
            
            latest_tick = cursor.execute(
                "SELECT MAX(timestamp) FROM tick_data"
            ).fetchone()[0]
        
        return {
            'tick_count': tick_count,
            'ohlcv_count': ohlcv_count,
            'symbol_count': symbols,
            'oldest_timestamp': oldest_tick,
            'latest_timestamp': latest_tick,
            'data_span_hours': (latest_tick - oldest_tick) / 3600 if oldest_tick and latest_tick else 0
        }
    
    def cleanup_old_data(self, days: int = 7):
        """Clean up data older than specified days"""
        cutoff = (datetime.now() - timedelta(days=days)).timestamp()
        
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                deleted_ticks = cursor.execute(
                    "DELETE FROM tick_data WHERE timestamp < ?", (cutoff,)
                ).rowcount
                
                deleted_ohlcv = cursor.execute(
                    "DELETE FROM ohlcv_data WHERE timestamp < ?", (cutoff,)
                ).rowcount
                
                conn.commit()
        
        logger.info(f"Cleaned up {deleted_ticks} ticks and {deleted_ohlcv} OHLCV records")
