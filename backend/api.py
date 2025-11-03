"""
FastAPI REST API
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class HedgeRatioRequest(BaseModel):
    symbol_x: str
    symbol_y: str
    interval: str = "1m"
    window: Optional[int] = None


class AlertRequest(BaseModel):
    name: str
    symbol_pair: str
    condition: str
    threshold: float


def create_api_app(db_manager, alert_manager) -> FastAPI:
    """Create FastAPI application"""
    
    app = FastAPI(
        title="Trading Analytics API",
        description="Real-time trading analytics",
        version="1.0.0"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    def root():
        return {
            "message": "Trading Analytics API",
            "version": "1.0.0",
            "docs": "/docs"
        }
    
    @app.get("/api/health")
    def health_check():
        summary = db_manager.get_data_summary()
        return {"status": "healthy", "data_summary": summary}
    
    @app.get("/api/symbols")
    def get_symbols():
        symbols = db_manager.get_available_symbols()
        return {"symbols": symbols}
    
    @app.get("/api/ohlcv/{symbol}")
    def get_ohlcv(
        symbol: str,
        interval: str = Query("1m"),
        limit: int = Query(500)
    ):
        try:
            data = db_manager.get_ohlcv_data(symbol.upper(), interval, limit=limit)
            
            if data.empty:
                raise HTTPException(status_code=404, detail="No data found")
            
            return {
                "symbol": symbol.upper(),
                "interval": interval,
                "count": len(data),
                "data": data.to_dict(orient="records")
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/analytics/hedge-ratio")
    def calculate_hedge_ratio(request: HedgeRatioRequest):
        try:
            from backend.analytics import AnalyticsEngine
            import pandas as pd
            
            data_x = db_manager.get_ohlcv_data(
                request.symbol_x.upper(), request.interval, limit=1000
            )
            data_y = db_manager.get_ohlcv_data(
                request.symbol_y.upper(), request.interval, limit=1000
            )
            
            if data_x.empty or data_y.empty:
                raise HTTPException(status_code=404, detail="Insufficient data")
            
            merged = pd.merge(
                data_x[['timestamp', 'close']],
                data_y[['timestamp', 'close']],
                on='timestamp',
                suffixes=('_x', '_y')
            )
            
            result = AnalyticsEngine.ols_regression(
                merged['close_y'],
                merged['close_x'],
                window=request.window
            )
            
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/alerts")
    def create_alert(request: AlertRequest):
        try:
            alert_id = alert_manager.create_alert(
                name=request.name,
                symbol_pair=request.symbol_pair,
                condition=request.condition,
                threshold=request.threshold
            )
            return {"alert_id": alert_id, "status": "created"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/alerts")
    def get_alerts():
        try:
            alerts = db_manager.get_active_alerts()
            return {"alerts": alerts}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.delete("/api/alerts/{alert_id}")
    def delete_alert(alert_id: str):
        try:
            db_manager.delete_alert(alert_id)
            return {"status": "deleted"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app
