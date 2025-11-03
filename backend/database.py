import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import asyncio
from threading import Lock
import json

class DatabaseManager:
    """
    Manages all database operations for tick data and analytics storage.
    Designed for easy migration to TimescaleDB or InfluxDB.
    """
    
    def __init__(self, db_path: str = "data/trading_data.db"):
        self.db_path = db_path
        self.lock = Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Tick data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tick_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    is_buyer_maker INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_tick_symbol_timestamp 
                ON tick_data(symbol, timestamp)
            """)
            
            # Resampled data table (OHLCV)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS resampled_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    trade_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(timestamp, symbol, interval)
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_resampled_symbol_interval_timestamp 
                ON resampled_data(symbol, interval, timestamp)
            """)
            
            # Analytics results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analytics_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    symbol_pair TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_analytics_symbol_timestamp 
                ON analytics_results(symbol_pair, timestamp)
            """)
            
            # Alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE NOT NULL,
                    symbol_pair TEXT NOT NULL,
                    condition TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    is_active INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Alert triggers table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alert_triggers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT NOT NULL,
                    triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    value REAL NOT NULL,
                    message TEXT
                )
            """)
            
            conn.commit()
    
    def insert_tick_data(self, data: List[Dict]):
        """Insert tick data in batch"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT INTO tick_data (timestamp, symbol, price, quantity, is_buyer_maker)
                    VALUES (?, ?, ?, ?, ?)
                """, [(d['timestamp'], d['symbol'], d['price'], d['quantity'], 
                       d.get('is_buyer_maker', 0)) for d in data])
                conn.commit()
    
    def insert_resampled_data(self, data: List[Dict]):
        """Insert resampled OHLCV data"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT OR REPLACE INTO resampled_data 
                    (timestamp, symbol, interval, open, high, low, close, volume, trade_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [(d['timestamp'], d['symbol'], d['interval'], d['open'], 
                       d['high'], d['low'], d['close'], d['volume'], d.get('trade_count', 0)) 
                       for d in data])
                conn.commit()
    
    def get_tick_data(self, symbol: str, start_time: Optional[float] = None, 
                      end_time: Optional[float] = None, limit: int = 10000) -> pd.DataFrame:
        """Retrieve tick data for a symbol"""
        query = "SELECT * FROM tick_data WHERE symbol = ?"
        params = [symbol]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        return df.sort_values('timestamp') if not df.empty else df
    
    def get_resampled_data(self, symbol: str, interval: str, 
                           start_time: Optional[float] = None, 
                           limit: int = 1000) -> pd.DataFrame:
        """Retrieve resampled OHLCV data"""
        query = "SELECT * FROM resampled_data WHERE symbol = ? AND interval = ?"
        params = [symbol, interval]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        return df.sort_values('timestamp') if not df.empty else df
    
    def save_analytics_result(self, symbol_pair: str, interval: str, 
                              timestamp: float, metric_name: str, metric_value: any):
        """Save analytics result"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO analytics_results 
                    (timestamp, symbol_pair, interval, metric_name, metric_value)
                    VALUES (?, ?, ?, ?, ?)
                """, (timestamp, symbol_pair, interval, metric_name, json.dumps(metric_value)))
                conn.commit()
    
    def get_latest_analytics(self, symbol_pair: str, metric_name: str, 
                            limit: int = 100) -> pd.DataFrame:
        """Get latest analytics results"""
        query = """
            SELECT * FROM analytics_results 
            WHERE symbol_pair = ? AND metric_name = ?
            ORDER BY timestamp DESC LIMIT ?
        """
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=[symbol_pair, metric_name, limit])
        
        return df.sort_values('timestamp') if not df.empty else df
    
    def upload_ohlc_data(self, df: pd.DataFrame, symbol: str, interval: str):
        """Upload OHLC data from CSV/DataFrame"""
        records = []
        for _, row in df.iterrows():
            records.append({
                'timestamp': row.get('timestamp', row.name.timestamp() if hasattr(row.name, 'timestamp') else 0),
                'symbol': symbol,
                'interval': interval,
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume'],
                'trade_count': row.get('trade_count', 0)
            })
        
        self.insert_resampled_data(records)
    
    def cleanup_old_data(self, days: int = 7):
        """Clean up data older than specified days"""
        cutoff_time = (datetime.now() - timedelta(days=days)).timestamp()
        
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM tick_data WHERE timestamp < ?", (cutoff_time,))
                cursor.execute("DELETE FROM resampled_data WHERE timestamp < ?", (cutoff_time,))
                cursor.execute("DELETE FROM analytics_results WHERE timestamp < ?", (cutoff_time,))
                conn.commit()
