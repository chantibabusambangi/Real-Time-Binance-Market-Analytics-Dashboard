"""
Alert management system
"""

import asyncio
from typing import List, Dict
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)


class AlertManager:
    """Manages custom alerts and notifications"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.running = False
        self.check_interval = 1
    
    async def check_alerts(self):
        """Check all active alerts"""
        active_alerts = self.db_manager.get_active_alerts()
        
        for alert in active_alerts:
            try:
                symbols = alert['symbol_pair'].split('/')
                if len(symbols) != 2:
                    continue
                
                data = self.db_manager.get_ohlcv_data(
                    symbols[0], '1m', limit=100
                )
                
                if data.empty:
                    continue
                
                triggered, value, message = self._evaluate_condition(alert, data)
                
                if triggered:
                    self.trigger_alert(alert, value, message)
            
            except Exception as e:
                logger.error(f"Error checking alert {alert['alert_id']}: {e}")
    
    def _evaluate_condition(self, alert: Dict, data) -> tuple:
        """Evaluate alert condition"""
        condition = alert['condition']
        threshold = alert['threshold']
        
        if condition == 'price_above':
            current_price = data['close'].iloc[-1]
            if current_price > threshold:
                return True, current_price, f"Price {current_price} > {threshold}"
        
        elif condition == 'price_below':
            current_price = data['close'].iloc[-1]
            if current_price < threshold:
                return True, current_price, f"Price {current_price} < {threshold}"
        
        elif condition == 'volume_spike':
            avg_volume = data['volume'].rolling(20).mean().iloc[-1]
            current_volume = data['volume'].iloc[-1]
            if current_volume > threshold * avg_volume:
                return True, current_volume, f"Volume spike: {current_volume}"
        
        return False, 0, ""
    
    def trigger_alert(self, alert: Dict, value: float, message: str):
        """Trigger an alert"""
        logger.info(f"🔔 ALERT: {alert['name']} - {message}")
        self.db_manager.log_alert_trigger(alert['alert_id'], value, message)
    
    async def start_monitoring(self):
        """Start alert monitoring"""
        self.running = True
        logger.info("Alert monitoring started")
        
        while self.running:
            await self.check_alerts()
            await asyncio.sleep(self.check_interval)
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
    
    def create_alert(self, name: str, symbol_pair: str, 
                    condition: str, threshold: float) -> str:
        """Create new alert"""
        alert_data = {
            'alert_id': str(uuid.uuid4()),
            'name': name,
            'symbol_pair': symbol_pair,
            'condition': condition,
            'threshold': threshold,
            'is_active': 1
        }
        
        return self.db_manager.create_alert(alert_data)
