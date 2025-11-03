"""
Real-Time Trading Analytics Application
Main entry point - orchestrates backend services and frontend dashboard
"""

import asyncio
import threading
import time
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from backend.data_ingestion import DataIngestionManager
from backend.database import DatabaseManager
from backend.alerts import AlertManager
from backend.api import create_api_app
from config.settings import Settings
from utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)


class TradingAnalyticsApp:
    """Main application orchestrator"""
    
    def __init__(self):
        self.settings = Settings()
        self.db_manager = DatabaseManager(self.settings.database_path)
        self.alert_manager = AlertManager(self.db_manager)
        self.ingestion_manager = None
        self.api_app = None
        self.running = False
        
    async def start_data_ingestion(self):
        """Start WebSocket data ingestion in background"""
        try:
            self.ingestion_manager = DataIngestionManager(
                db_manager=self.db_manager,
                symbols=self.settings.default_symbols,
                intervals=self.settings.sampling_intervals
            )
            
            logger.info(f"Starting data ingestion for symbols: {self.settings.default_symbols}")
            await self.ingestion_manager.start()
            
        except Exception as e:
            logger.error(f"Error in data ingestion: {e}")
    
    async def start_alert_monitoring(self):
        """Start alert monitoring service"""
        try:
            logger.info("Starting alert monitoring service")
            await self.alert_manager.start_monitoring()
        except Exception as e:
            logger.error(f"Error in alert monitoring: {e}")
    
    def start_api_server(self):
        """Start FastAPI server"""
        import uvicorn
        
        self.api_app = create_api_app(self.db_manager, self.alert_manager)
        
        logger.info(f"Starting API server on {self.settings.api_host}:{self.settings.api_port}")
        
        config = uvicorn.Config(
            self.api_app,
            host=self.settings.api_host,
            port=self.settings.api_port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        
        # Run in thread to not block
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        
        return thread
    
    def start_streamlit_dashboard(self):
        """Start Streamlit dashboard"""
        import subprocess
        
        logger.info(f"Starting Streamlit dashboard on port {self.settings.streamlit_port}")
        
        dashboard_path = Path(__file__).parent / "frontend" / "dashboard.py"
        
        process = subprocess.Popen([
            "streamlit", "run", str(dashboard_path),
            "--server.port", str(self.settings.streamlit_port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ])
        
        return process
    
    async def run_async_services(self):
        """Run all async services"""
        tasks = [
            asyncio.create_task(self.start_data_ingestion()),
            asyncio.create_task(self.start_alert_monitoring())
        ]
        
        await asyncio.gather(*tasks)
    
    def run(self):
        """Main run method"""
        try:
            self.running = True
            
            # Print startup banner
            print("\n" + "="*70)
            print("  🚀 REAL-TIME TRADING ANALYTICS PLATFORM")
            print("="*70)
            print(f"\n📊 Dashboard: http://localhost:{self.settings.streamlit_port}")
            print(f"🔌 API: http://localhost:{self.settings.api_port}")
            print(f"📡 WebSocket: Connecting to Binance...")
            print(f"💱 Symbols: {', '.join(self.settings.default_symbols)}")
            print("\n" + "="*70 + "\n")
            
            # Start API server (non-blocking)
            api_thread = self.start_api_server()
            time.sleep(2)  # Give API time to start
            
            # Start Streamlit dashboard (non-blocking)
            streamlit_process = self.start_streamlit_dashboard()
            time.sleep(3)  # Give Streamlit time to start
            
            # Run async services (blocking)
            logger.info("Starting async services (data ingestion & alerts)")
            asyncio.run(self.run_async_services())
            
        except KeyboardInterrupt:
            logger.info("\n⚠️  Shutting down gracefully...")
            self.shutdown()
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown"""
        self.running = False
        
        if self.ingestion_manager:
            logger.info("Stopping data ingestion...")
            # Flush any remaining data
            asyncio.run(self.ingestion_manager.stop())
        
        if self.alert_manager:
            logger.info("Stopping alert manager...")
            self.alert_manager.stop()
        
        logger.info("✅ Shutdown complete")


def main():
    """Entry point"""
    app = TradingAnalyticsApp()
    app.run()


if __name__ == "__main__":
    main()
