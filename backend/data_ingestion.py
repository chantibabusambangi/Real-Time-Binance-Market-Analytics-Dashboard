import asyncio
import websockets
import json
from datetime import datetime
from typing import List, Dict, Callable
import logging
from collections import defaultdict, deque
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BinanceWebSocketClient:
    """
    Manages WebSocket connection to Binance and streams tick data.
    Implements reconnection logic and error handling.
    """
    
    def __init__(self, symbols: List[str], on_message_callback: Callable):
        self.symbols = [s.lower() for s in symbols]
        self.on_message_callback = on_message_callback
        self.ws_url = "wss://stream.binance.com:9443/stream"
        self.is_running = False
        self.reconnect_delay = 5
        
    async def connect(self):
        """Establish WebSocket connection"""
        # Subscribe to trade streams for all symbols
        streams = [f"{symbol}@trade" for symbol in self.symbols]
        params = {"method": "SUBSCRIBE", "params": streams, "id": 1}
        
        self.is_running = True
        
        while self.is_running:
            try:
                async with websockets.connect(self.ws_url) as websocket:
                    # Send subscription message
                    await websocket.send(json.dumps(params))
                    logger.info(f"Subscribed to {len(streams)} streams")
                    
                    # Receive messages
                    async for message in websocket:
                        if not self.is_running:
                            break
                        
                        try:
                            data = json.loads(message)
                            if 'data' in data:
                                await self.process_message(data['data'])
                        except json.JSONDecodeError:
                            logger.error(f"Failed to decode message: {message}")
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
                            
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed, reconnecting...")
                await asyncio.sleep(self.reconnect_delay)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await asyncio.sleep(self.reconnect_delay)
    
    async def process_message(self, data: Dict):
        """Process incoming trade message"""
        try:
            tick_data = {
                'timestamp': data['T'] / 1000.0,  # Convert to seconds
                'symbol': data['s'].lower(),
                'price': float(data['p']),
                'quantity': float(data['q']),
                'is_buyer_maker': 1 if data['m'] else 0,
                'trade_id': data['t']
            }
            
            await self.on_message_callback(tick_data)
            
        except KeyError as e:
            logger.error(f"Missing key in data: {e}, data: {data}")
        except Exception as e:
            logger.error(f"Error in process_message: {e}")
    
    async def stop(self):
        """Stop the WebSocket connection"""
        self.is_running = False


class DataAggregator:
    """
    Aggregates tick data into OHLCV bars for different timeframes.
    Implements efficient sampling and buffering.
    """
    
    def __init__(self, db_manager, intervals: Dict[str, int]):
        self.db_manager = db_manager
        self.intervals = intervals  # {'1s': 1, '1m': 60, '5m': 300}
        
        # Buffers for each symbol and interval
        self.tick_buffers = defaultdict(lambda: deque(maxlen=1000))
        self.ohlcv_buffers = defaultdict(lambda: defaultdict(lambda: {
            'open': None, 'high': None, 'low': None, 'close': None, 
            'volume': 0, 'trade_count': 0, 'start_time': None
        }))
        
        self.last_flush = defaultdict(lambda: defaultdict(float))
        
    async def process_tick(self, tick_data: Dict):
        """Process incoming tick data"""
        symbol = tick_data['symbol']
        timestamp = tick_data['timestamp']
        price = tick_data['price']
        quantity = tick_data['quantity']
        
        # Add to tick buffer
        self.tick_buffers[symbol].append(tick_data)
        
        # Update OHLCV for each interval
        for interval_name, interval_seconds in self.intervals.items():
            await self.update_ohlcv(symbol, interval_name, interval_seconds, 
                                   timestamp, price, quantity)
    
    async def update_ohlcv(self, symbol: str, interval_name: str, 
                          interval_seconds: int, timestamp: float, 
                          price: float, quantity: float):
        """Update OHLCV data for a specific interval"""
        # Calculate bar start time
        bar_start_time = int(timestamp / interval_seconds) * interval_seconds
        
        bar = self.ohlcv_buffers[symbol][interval_name]
        
        # Check if we need to flush the previous bar
        if bar['start_time'] is not None and bar_start_time != bar['start_time']:
            await self.flush_bar(symbol, interval_name, bar)
            # Reset for new bar
            bar = {
                'open': None, 'high': None, 'low': None, 'close': None,
                'volume': 0, 'trade_count': 0, 'start_time': bar_start_time
            }
            self.ohlcv_buffers[symbol][interval_name] = bar
        
        # Initialize bar if needed
        if bar['start_time'] is None:
            bar['start_time'] = bar_start_time
        
        # Update OHLCV
        if bar['open'] is None:
            bar['open'] = price
        bar['high'] = price if bar['high'] is None else max(bar['high'], price)
        bar['low'] = price if bar['low'] is None else min(bar['low'], price)
        bar['close'] = price
        bar['volume'] += quantity
        bar['trade_count'] += 1
    
    async def flush_bar(self, symbol: str, interval_name: str, bar: Dict):
        """Flush completed bar to database"""
        if bar['open'] is None:
            return
        
        record = {
            'timestamp': bar['start_time'],
            'symbol': symbol,
            'interval': interval_name,
            'open': bar['open'],
            'high': bar['high'],
            'low': bar['low'],
            'close': bar['close'],
            'volume': bar['volume'],
            'trade_count': bar['trade_count']
        }
        
        try:
            self.db_manager.insert_resampled_data([record])
            logger.debug(f"Flushed {interval_name} bar for {symbol} at {bar['start_time']}")
        except Exception as e:
            logger.error(f"Error flushing bar: {e}")
    
    async def flush_all_current_bars(self):
        """Flush all current bars (useful for shutdown)"""
        for symbol in self.ohlcv_buffers:
            for interval_name in self.ohlcv_buffers[symbol]:
                bar = self.ohlcv_buffers[symbol][interval_name]
                if bar['open'] is not None:
                    await self.flush_bar(symbol, interval_name, bar)
    
    def get_tick_buffer(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent ticks from buffer"""
        return list(self.tick_buffers[symbol])[-limit:]


class DataIngestionManager:
    """
    Orchestrates data ingestion, aggregation, and storage.
    Main entry point for the data pipeline.
    """
    
    def __init__(self, db_manager, symbols: List[str], intervals: Dict[str, int]):
        self.db_manager = db_manager
        self.symbols = symbols
        self.intervals = intervals
        
        self.aggregator = DataAggregator(db_manager, intervals)
        self.ws_client = None
        
        # Batch insertion for tick data
        self.tick_batch = []
        self.batch_size = 100
        self.last_batch_flush = time.time()
        self.batch_flush_interval = 5  # seconds
        
    async def on_tick_received(self, tick_data: Dict):
        """Callback when new tick data is received"""
        # Add to batch
        self.tick_batch.append(tick_data)
        
        # Process in aggregator
        await self.aggregator.process_tick(tick_data)
        
        # Flush batch if needed
        if len(self.tick_batch) >= self.batch_size or \
           time.time() - self.last_batch_flush > self.batch_flush_interval:
            await self.flush_tick_batch()
    
    async def flush_tick_batch(self):
        """Flush tick batch to database"""
        if not self.tick_batch:
            return
        
        try:
            self.db_manager.insert_tick_data(self.tick_batch)
            logger.debug(f"Flushed {len(self.tick_batch)} ticks to database")
            self.tick_batch = []
            self.last_batch_flush = time.time()
        except Exception as e:
            logger.error(f"Error flushing tick batch: {e}")
    
    async def start(self):
        """Start data ingestion"""
        self.ws_client = BinanceWebSocketClient(self.symbols, self.on_tick_received)
        
        # Start WebSocket connection
        ws_task = asyncio.create_task(self.ws_client.connect())
        
        # Start periodic batch flusher
        flush_task = asyncio.create_task(self.periodic_flush())
        
        await asyncio.gather(ws_task, flush_task)
    
    async def periodic_flush(self):
        """Periodically flush batches"""
        while True:
            await asyncio.sleep(self.batch_flush_interval)
            await self.flush_tick_batch()
    
    async def stop(self):
        """Stop data ingestion"""
        if self.ws_client:
            await self.ws_client.stop()
        await self.flush_tick_batch()
        await self.aggregator.flush_all_current_bars()
