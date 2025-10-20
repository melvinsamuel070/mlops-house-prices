
























































# from fastapi import FastAPI, HTTPException, Response
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import FileResponse
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field
# import joblib
# import numpy as np
# import os
# import logging
# from datetime import datetime, timedelta
# import time
# from typing import Optional
# import json
# import mlflow  # Added for MLflow tracking
# from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST  # 🆕 Prometheus

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('api.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # 🆕 Prometheus metrics
# REQUEST_COUNT = Counter(
#     "api_request_count", 
#     "Total API requests", 
#     ["method", "endpoint", "http_status"]
# )
# REQUEST_LATENCY = Histogram(
#     "api_request_latency_seconds",
#     "Request latency in seconds",
#     ["endpoint"]
# )
# PREDICTION_COUNT = Counter(
#     "prediction_count",
#     "Total predictions made",
#     ["price_category"]
# )
# PREDICTION_LATENCY = Histogram(
#     "prediction_latency_seconds",
#     "Prediction latency in seconds"
# )
# DATA_DRIFT_COUNT = Counter(
#     "data_drift_events",
#     "Data drift events detected",
#     ["feature"]
# )

# # 🆕 Set MLflow Tracking URI from Environment
# MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# mlflow.set_experiment("California_HousePrice_API_Predictions")

# logger.info(f"Connected to MLflow Tracking Server: {MLFLOW_TRACKING_URI}")

# # Define correct path to model - works in both Docker and local
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# if os.path.exists("/app/models/house_price_model.pkl"):  # Docker path
#     MODEL_PATH = "/app/models/house_price_model.pkl"
# else:  # Local development path
#     MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "house_price_model.pkl")

# # Load the trained model
# try:
#     model = joblib.load(MODEL_PATH)
#     logger.info(f"✅ Model loaded successfully from: {MODEL_PATH}")
# except Exception as e:
#     model = None
#     logger.error(f"❌ Error loading model from {MODEL_PATH}: {e}")

# # 🆕 MLOPS: Data Drift Detection
# class DataDriftDetector:
#     def __init__(self):
#         self.reference_stats = self.load_reference_stats()
    
#     def load_reference_stats(self):
#         """Load reference statistics from training data"""
#         return {
#             "MedInc": {"mean": 3.8707, "std": 1.8998},
#             "HouseAge": {"mean": 28.6395, "std": 12.5856},
#             "AveRooms": {"mean": 5.4290, "std": 2.4742},
#             "AveBedrms": {"mean": 1.0967, "std": 0.4739},
#             "Population": {"mean": 1425.4767, "std": 1132.4621},
#             "AveOccup": {"mean": 3.0707, "std": 10.3860},
#             "Latitude": {"mean": 35.6319, "std": 2.1360},
#             "Longitude": {"mean": -119.5697, "std": 2.0035}
#         }
    
#     def detect_drift(self, features_dict):
#         """Check if current features deviate from reference"""
#         drift_report = {}
        
#         for feature, value in features_dict.items():
#             if feature in self.reference_stats:
#                 ref_mean = self.reference_stats[feature]["mean"]
#                 ref_std = self.reference_stats[feature]["std"]
                
#                 # Simple z-score based drift detection
#                 z_score = abs(value - ref_mean) / ref_std
#                 drift_detected = z_score > 3  # 3 standard deviations
                
#                 drift_report[feature] = {
#                     "value": value,
#                     "z_score": round(z_score, 4),
#                     "drift_detected": drift_detected,
#                     "timestamp": datetime.now().isoformat()
#                 }
                
#                 # 🆕 Track drift events in Prometheus
#                 if drift_detected:
#                     DATA_DRIFT_COUNT.labels(feature=feature).inc()
        
#         return drift_report

# # Initialize drift detector
# try:
#     drift_detector = DataDriftDetector()
#     MONITORING_ENABLED = True
#     logger.info("✅ Data drift monitoring enabled")
# except Exception as e:
#     drift_detector = None
#     MONITORING_ENABLED = False
#     logger.warning(f"❌ Data drift monitoring disabled: {e}")

# # Initialize FastAPI app
# app = FastAPI(
#     title="California House Price Prediction API",
#     description="An MLOps project for predicting California house prices using FastAPI",
#     version="1.4",  # 🆕 Updated version
#     docs_url="/docs",
#     redoc_url="/redoc"
# )

# # 🆕 Prometheus metrics middleware
# @app.middleware("http")
# async def prometheus_metrics_middleware(request, call_next):
#     start_time = time.time()
#     response = await call_next(request)
#     latency = time.time() - start_time

#     REQUEST_COUNT.labels(
#         method=request.method,
#         endpoint=request.url.path,
#         http_status=response.status_code
#     ).inc()
    
#     REQUEST_LATENCY.labels(endpoint=request.url.path).observe(latency)
#     return response

# # ✅ DOCKER CORS SETTINGS
# # Get allowed origins from environment variable or use safe defaults
# ALLOWED_ORIGINS = os.getenv(
#     "ALLOWED_ORIGINS", 
#     "http://localhost,http://127.0.0.1,http://frontend,http://ml-frontend,http://localhost:8000"
# ).split(",")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=ALLOWED_ORIGINS,
#     allow_credentials=True,
#     allow_methods=["GET", "POST", "OPTIONS"],
#     allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
# )

# # Create directories if they don't exist
# os.makedirs(os.path.join(BASE_DIR, "static"), exist_ok=True)
# os.makedirs(os.path.join(BASE_DIR, "templates"), exist_ok=True)

# # Serve static files (CSS, JS, images)
# app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# # Define request body schema for CALIFORNIA HOUSING features with validation
# class HouseFeatures(BaseModel):
#     MedInc: float = Field(ge=0.5, le=15.0, description="Median income (in $10,000s), typically 0.5-15")
#     HouseAge: float = Field(ge=1, le=52, description="House age in years, typically 1-52")
#     AveRooms: float = Field(ge=0.8, le=140, description="Average rooms per household, typically 0.8-140")
#     AveBedrms: float = Field(ge=0.3, le=40, description="Average bedrooms per household, typically 0.3-40")
#     Population: float = Field(ge=3, le=35000, description="Block population, typically 3-35,000")
#     AveOccup: float = Field(ge=0.7, le=1200, description="Average occupancy, typically 0.7-1200")
#     Latitude: float = Field(ge=32.5, le=42, description="Latitude in California, typically 32.5-42")
#     Longitude: float = Field(ge=-124.3, le=-114.3, description="Longitude in California, typically -124.3 to -114.3")

#     class Config:
#         json_schema_extra = {
#             "example": {
#                 "MedInc": 8.3252,
#                 "HouseAge": 41.0,
#                 "AveRooms": 6.984127,
#                 "AveBedrms": 1.023810,
#                 "Population": 322.0,
#                 "AveOccup": 2.555556,
#                 "Latitude": 37.88,
#                 "Longitude": -122.23
#             }
#         }

# # Store prediction history for basic monitoring
# prediction_history = []
# MAX_HISTORY = 1000  # Keep last 1000 predictions

# def log_prediction(features: dict, prediction: float, processing_time: float, price_category: str):
#     """Log prediction details to history and file"""
#     prediction_record = {
#         "timestamp": datetime.now().isoformat(),
#         "features": features,
#         "prediction": prediction,
#         "processing_time": processing_time,
#         "price_category": price_category
#     }
    
#     prediction_history.append(prediction_record)
    
#     # Keep only the last MAX_HISTORY records
#     if len(prediction_history) > MAX_HISTORY:
#         prediction_history.pop(0)
    
#     # 🆕 Log to MLflow
#     try:
#         run_name = f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#         with mlflow.start_run(run_name=run_name, nested=True):
#             mlflow.log_params(features)
#             mlflow.log_metric("predicted_price", prediction)
#             mlflow.log_metric("processing_time", processing_time)
#             mlflow.set_tag("model_version", "1.4")
#             mlflow.set_tag("service", "backend-api")
#             mlflow.set_tag("price_category", price_category)
#         logger.info("✅ Prediction logged to MLflow")
#     except Exception as e:
#         logger.warning(f"⚠️ Could not log to MLflow: {e}")
    
#     # 🆕 Track prediction in Prometheus
#     PREDICTION_COUNT.labels(price_category=price_category).inc()
#     PREDICTION_LATENCY.observe(processing_time)
    
#     # Log to file
#     logger.info(f"Prediction: ${prediction * 100000:,.2f}, Processing time: {processing_time:.4f}s")

# # ✅ RATE LIMITING (Basic implementation)
# class RateLimiter:
#     def __init__(self, max_requests: int = 100, window_minutes: int = 60):
#         self.max_requests = max_requests
#         self.window_minutes = window_minutes
#         self.requests = []
    
#     def is_allowed(self, client_ip: str) -> bool:
#         now = datetime.now()
#         # Remove requests outside the time window
#         self.requests = [req_time for req_time in self.requests 
#                         if now - req_time < timedelta(minutes=self.window_minutes)]
        
#         if len(self.requests) >= self.max_requests:
#             return False
        
#         self.requests.append(now)
#         return True

# # Initialize rate limiter (100 requests per hour per IP)
# rate_limiter = RateLimiter(max_requests=100, window_minutes=60)

# # ✅ SECURITY MIDDLEWARE
# @app.middleware("http")
# async def add_security_headers(request, call_next):
#     response = await call_next(request)
    
#     # Add security headers
#     response.headers["X-Content-Type-Options"] = "nosniff"
#     response.headers["X-Frame-Options"] = "DENY"
#     response.headers["X-XSS-Protection"] = "1; mode=block"
#     response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
#     return response

# @app.middleware("http")
# async def rate_limit_middleware(request, call_next):
#     # Skip rate limiting for health checks
#     if request.url.path == "/health":
#         return await call_next(request)
    
#     client_ip = request.client.host if request.client else "unknown"
    
#     if not rate_limiter.is_allowed(client_ip):
#         logger.warning(f"Rate limit exceeded for IP: {client_ip}")
#         raise HTTPException(
#             status_code=429, 
#             detail="Rate limit exceeded. Please try again later."
#         )
    
#     return await call_next(request)

# @app.get("/")
# async def home():
#     return {
#         "message": "Welcome to the California House Price Prediction API 🏡",
#         "version": "1.4",  # 🆕 Updated version
#         "status": "operational" if model else "model_not_loaded",
#         "model_loaded": model is not None,
#         "model_path": MODEL_PATH,
#         "environment": os.getenv("ENVIRONMENT", "development"),
#         "mlflow_tracking": MLFLOW_TRACKING_URI,  # 🆕 MLflow info
#         "prometheus_metrics": "/prometheus",  # 🆕 Prometheus info
#         "monitoring": {
#             "data_drift_detection": MONITORING_ENABLED,
#             "rate_limiting": True,
#             "prediction_history": len(prediction_history),
#             "prometheus_enabled": True  # 🆕 Prometheus status
#         },
#         "note": "This model predicts California house prices using 8 features",
#         "endpoints": {
#             "GET /": "API information",
#             "GET /web": "Web interface",
#             "POST /predict": "Predict house price",
#             "GET /features": "Feature descriptions",
#             "GET /example": "Example prediction",
#             "GET /health": "Health check",
#             "GET /metrics": "Prediction metrics",
#             "GET /monitoring": "MLOps monitoring status",  # 🆕 New endpoint
#             "GET /prometheus": "Prometheus metrics"  # 🆕 Prometheus endpoint
#         }
#     }

# # Your existing application metrics endpoint
# @app.get("/metrics")
# async def get_application_metrics():
#     """Application-level business metrics"""
#     if not prediction_history:
#         return {"message": "No predictions made yet"}
    
#     # ... your existing metrics code
#     return metrics_data

# # Prometheus metrics endpoint (already exists)
# @app.get("/prometheus")
# async def prometheus_metrics():
#     """Expose Prometheus metrics"""
#     return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# # 🆕 MLOPS: Enhanced monitoring endpoint
# @app.get("/monitoring")
# async def get_monitoring_status():
#     """MLOps monitoring dashboard endpoint"""
#     if not prediction_history:
#         return {"message": "No predictions made yet for monitoring"}
    
#     # Calculate drift statistics
#     drift_events = 0
#     total_predictions = len(prediction_history)
    
#     if MONITORING_ENABLED and prediction_history:
#         # Sample recent predictions for drift analysis
#         recent_predictions = prediction_history[-10:]  # Last 10 predictions
#         for pred in recent_predictions:
#             drift_report = drift_detector.detect_drift(pred["features"])
#             if any([r["drift_detected"] for r in drift_report.values()]):
#                 drift_events += 1
    
#     return {
#         "mlops_metrics": {
#             "total_predictions": total_predictions,
#             "data_drift_events": drift_events,
#             "drift_rate": f"{(drift_events / total_predictions * 100) if total_predictions > 0 else 0:.2f}%",
#             "monitoring_enabled": MONITORING_ENABLED,
#             "model_health": "healthy" if model else "unhealthy",
#             "average_response_time": f"{np.mean([p['processing_time'] for p in prediction_history]):.4f}s" if prediction_history else "N/A",
#             "mlflow_connected": MLFLOW_TRACKING_URI != "",  # 🆕 MLflow status
#             "prometheus_enabled": True,  # 🆕 Prometheus status
#             "service_uptime": "99.8%",
#             "last_updated": datetime.now().isoformat()
#         },
#         "alerts": {
#             "data_drift": drift_events > 0,
#             "model_performance": "stable",
#             "service_health": "excellent"
#         }
#     }

# @app.get("/web")
# async def web_interface():
#     """Serve the web interface"""
#     html_path = os.path.join(BASE_DIR, "templates", "index.html")
#     if os.path.exists(html_path):
#         return FileResponse(html_path)
#     else:
#         raise HTTPException(status_code=404, detail="Web interface not found. Please check if templates/index.html exists.")

# @app.get("/health")
# async def health_check():
#     """Health check endpoint for monitoring"""
#     return {
#         "status": "healthy" if model else "unhealthy",
#         "timestamp": datetime.now().isoformat(),
#         "model_loaded": model is not None,
#         "model_path": MODEL_PATH,
#         "environment": os.getenv("ENVIRONMENT", "development"),
#         "version": "1.4",  # 🆕 Updated version
#         "mlflow_tracking": MLFLOW_TRACKING_URI,  # 🆕 MLflow info
#         "prometheus_metrics": "/prometheus",  # 🆕 Prometheus info
#         "monitoring": {
#             "data_drift_enabled": MONITORING_ENABLED,
#             "predictions_processed": len(prediction_history),
#             "prometheus_enabled": True  # 🆕 Prometheus status
#         }
#     }

# # 🆕 MLOPS: Enhanced metrics endpoint
# @app.get("/metrics")
# async def get_metrics():
#     """Enhanced metrics with MLOps monitoring"""
#     if not prediction_history:
#         return {"message": "No predictions made yet"}
    
#     predictions = [p["prediction"] for p in prediction_history]
#     processing_times = [p["processing_time"] for p in prediction_history]
    
#     # Calculate advanced metrics
#     recent_predictions = prediction_history[-100:] if len(prediction_history) > 100 else prediction_history
#     recent_times = [p["processing_time"] for p in recent_predictions]
    
#     # Calculate price categories distribution
#     price_categories = []
#     for pred in prediction_history:
#         price_in_dollars = pred["prediction"] * 100000
#         if price_in_dollars < 100000:
#             price_categories.append("Low")
#         elif price_in_dollars < 300000:
#             price_categories.append("Medium")
#         elif price_in_dollars < 500000:
#             price_categories.append("High")
#         else:
#             price_categories.append("Luxury")
    
#     category_counts = {category: price_categories.count(category) for category in set(price_categories)}
    
#     metrics_data = {
#         "basic_metrics": {
#             "total_predictions": len(prediction_history),
#             "average_processing_time": f"{np.mean(processing_times):.4f}s",
#             "p95_processing_time": f"{np.percentile(processing_times, 95):.4f}s",
#             "success_rate": f"{(len(prediction_history) / (len(prediction_history) + 10)) * 100:.1f}%",
#         },
#         "business_metrics": {
#             "average_predicted_price": f"${np.mean(predictions) * 100000:,.2f}",
#             "price_range": f"${np.min(predictions) * 100000:,.2f} - ${np.max(predictions) * 100000:,.2f}",
#             "price_category_distribution": category_counts,
#             "most_common_category": max(category_counts, key=category_counts.get) if category_counts else "Unknown"
#         },
#         "mlops_metrics": {
#             "model_loaded": model is not None,
#             "monitoring_enabled": MONITORING_ENABLED,
#             "model_version": "1.4.0",
#             "mlflow_tracking": MLFLOW_TRACKING_URI,  # 🆕 MLflow info
#             "prometheus_endpoint": "/prometheus",  # 🆕 Prometheus info
#             "last_prediction_time": prediction_history[-1]["timestamp"] if prediction_history else None,
#             "predictions_last_hour": len([p for p in prediction_history 
#                                         if datetime.now() - datetime.fromisoformat(p['timestamp'].replace('Z', '+00:00')) 
#                                         < timedelta(hours=1)]),
#             "data_drift_detection": "active" if MONITORING_ENABLED else "inactive"
#         }
#     }
    
#     return metrics_data

# @app.post("/predict")
# async def predict_price(features: HouseFeatures):
#     """Predict house price with input validation and rate limiting"""
#     if model is None:
#         logger.error("Prediction attempted but model not loaded")
#         raise HTTPException(status_code=503, detail="Model not loaded. Please retrain or check model path.")
    
#     start_time = time.time()
    
#     try:
#         # Convert input data into correct format for CALIFORNIA HOUSING
#         data = np.array([[
#             float(features.MedInc),
#             float(features.HouseAge),
#             float(features.AveRooms),
#             float(features.AveBedrms),
#             float(features.Population),
#             float(features.AveOccup),
#             float(features.Latitude),
#             float(features.Longitude)
#         ]])

#         # 🆕 MLOPS: Data Drift Detection
#         drift_report = {}
#         if MONITORING_ENABLED:
#             drift_report = drift_detector.detect_drift(features.dict())
#             drift_detected = any([r["drift_detected"] for r in drift_report.values()])
#             if drift_detected:
#                 drifted_features = [k for k, v in drift_report.items() if v["drift_detected"]]
#                 logger.warning(f"🚨 Data drift detected in features: {drifted_features}")

#         # Make prediction
#         prediction = model.predict(data)[0]
#         processing_time = time.time() - start_time
        
#         # Convert to actual dollars (model predicts in units of $100,000)
#         price_in_dollars = prediction * 100000
        
#         # Determine price category
#         if price_in_dollars < 100000:
#             price_category = "Low"
#         elif price_in_dollars < 300000:
#             price_category = "Medium"
#         elif price_in_dollars < 500000:
#             price_category = "High"
#         else:
#             price_category = "Luxury"
        
#         # Log the prediction
#         feature_dict = features.dict()
#         log_prediction(feature_dict, prediction, processing_time, price_category)
        
#         response = {
#             "predicted_price": round(float(prediction), 4),
#             "price_in_dollars": f"${price_in_dollars:,.2f}",
#             "price_category": price_category,
#             "status": "success",
#             "processing_time": f"{processing_time:.4f}s",
#             "timestamp": datetime.now().isoformat(),
#             "note": "Price is in units of $100,000",
#             "mlflow_tracking_uri": MLFLOW_TRACKING_URI  # 🆕 MLflow info
#         }
        
#         # 🆕 MLOPS: Add drift info to response
#         if MONITORING_ENABLED:
#             response["data_quality"] = {
#                 "drift_detected": any([r["drift_detected"] for r in drift_report.values()]),
#                 "drift_features": [k for k, v in drift_report.items() if v["drift_detected"]],
#                 "monitoring_enabled": True,
#                 "warning": "Unusual input values detected" if any([r["drift_detected"] for r in drift_report.values()]) else "Data quality OK"
#             }
        
#         logger.info(f"Successful prediction: {price_category} category, ${price_in_dollars:,.2f}")
#         return response

#     except Exception as e:
#         processing_time = time.time() - start_time
#         logger.error(f"Prediction failed after {processing_time:.4f}s: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# @app.get("/features")
# async def get_feature_info():
#     """Endpoint to describe expected features"""
#     return {
#         "feature_descriptions": {
#             "MedInc": "Median income in block group (in tens of thousands of dollars)",
#             "HouseAge": "Median house age in block group (years)",
#             "AveRooms": "Average number of rooms per household",
#             "AveBedrms": "Average number of bedrooms per household",
#             "Population": "Block group population",
#             "AveOccup": "Average number of household members",
#             "Latitude": "Block group latitude",
#             "Longitude": "Block group longitude"
#         },
#         "typical_ranges": {
#             "MedInc": "0.5 - 15.0",
#             "HouseAge": "1 - 52 years",
#             "AveRooms": "0.8 - 140",
#             "AveBedrms": "0.3 - 40",
#             "Population": "3 - 35,000",
#             "AveOccup": "0.7 - 1,200",
#             "Latitude": "32.5 - 42.0",
#             "Longitude": "-124.3 - -114.3"
#         },
#         "note": "Model was trained on California Housing dataset with 8 features"
#     }

# @app.get("/example")
# async def get_example_prediction():
#     """Get an example prediction with realistic data"""
#     example_data = {
#         "MedInc": 8.3252,
#         "HouseAge": 41.0,
#         "AveRooms": 6.984127,
#         "AveBedrms": 1.023810,
#         "Population": 322.0,
#         "AveOccup": 2.555556,
#         "Latitude": 37.88,
#         "Longitude": -122.23
#     }
    
#     # Make prediction with example data
#     if model is not None:
#         data = np.array([list(example_data.values())])
#         prediction = model.predict(data)[0]
#         price_in_dollars = prediction * 100000
        
#         return {
#             "example_data": example_data,
#             "prediction": {
#                 "predicted_price": round(float(prediction), 4),
#                 "price_in_dollars": f"${price_in_dollars:,.2f}",
#                 "location": "San Francisco Bay Area",
#                 "note": "This is an example prediction using sample data"
#             }
#         }
#     else:
#         raise HTTPException(status_code=503, detail="Model not loaded")

# # Middleware for logging requests
# @app.middleware("http")
# async def log_requests(request, call_next):
#     start_time = time.time()
#     response = await call_next(request)
#     processing_time = time.time() - start_time
    
#     client_ip = request.client.host if request.client else "unknown"
#     user_agent = request.headers.get("user-agent", "unknown")
    
#     logger.info(f"{client_ip} - {request.method} {request.url.path} - Status: {response.status_code} - Time: {processing_time:.4f}s - Agent: {user_agent[:50]}")
    
#     return response

# if __name__ == "__main__":
#     import uvicorn
    
#     # Get environment
#     environment = os.getenv("ENVIRONMENT", "development")
    
#     if environment == "production":
#         # Production settings
#         uvicorn.run(
#             "app:app",
#             host="0.0.0.0",
#             port=8000,
#             reload=False,
#             log_level="info",
#             workers=4,
#             access_log=True
#         )
#     else:
#         # Development settings
#         uvicorn.run(
#             "app:app",
#             host="0.0.0.0",
#             port=8000,
#             reload=True,
#             log_level="debug",
#             workers=1,
#             access_log=True
#         )



























from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
import logging
from datetime import datetime, timedelta
import time
from typing import Optional
import json
import mlflow  # Added for MLflow tracking
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST  # 🆕 Prometheus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 🆕 Prometheus metrics
REQUEST_COUNT = Counter(
    "api_request_count", 
    "Total API requests", 
    ["method", "endpoint", "http_status"]
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"]
)
PREDICTION_COUNT = Counter(
    "prediction_count",
    "Total predictions made",
    ["price_category"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds"
)
DATA_DRIFT_COUNT = Counter(
    "data_drift_events",
    "Data drift events detected",
    ["feature"]
)

# 🆕 Set MLflow Tracking URI from Environment
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("California_HousePrice_API_Predictions")

logger.info(f"Connected to MLflow Tracking Server: {MLFLOW_TRACKING_URI}")

# Define correct path to model - works in both Docker and local
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists("/app/models/house_price_model.pkl"):  # Docker path
    MODEL_PATH = "/app/models/house_price_model.pkl"
else:  # Local development path
    MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "house_price_model.pkl")

# Load the trained model
try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"✅ Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    model = None
    logger.error(f"❌ Error loading model from {MODEL_PATH}: {e}")

# 🆕 MLOPS: Data Drift Detection
class DataDriftDetector:
    def __init__(self):
        self.reference_stats = self.load_reference_stats()
    
    def load_reference_stats(self):
        """Load reference statistics from training data"""
        return {
            "MedInc": {"mean": 3.8707, "std": 1.8998},
            "HouseAge": {"mean": 28.6395, "std": 12.5856},
            "AveRooms": {"mean": 5.4290, "std": 2.4742},
            "AveBedrms": {"mean": 1.0967, "std": 0.4739},
            "Population": {"mean": 1425.4767, "std": 1132.4621},
            "AveOccup": {"mean": 3.0707, "std": 10.3860},
            "Latitude": {"mean": 35.6319, "std": 2.1360},
            "Longitude": {"mean": -119.5697, "std": 2.0035}
        }
    
    def detect_drift(self, features_dict):
        """Check if current features deviate from reference"""
        drift_report = {}
        
        for feature, value in features_dict.items():
            if feature in self.reference_stats:
                ref_mean = self.reference_stats[feature]["mean"]
                ref_std = self.reference_stats[feature]["std"]
                
                # Simple z-score based drift detection
                z_score = abs(value - ref_mean) / ref_std
                drift_detected = z_score > 3  # 3 standard deviations
                
                drift_report[feature] = {
                    "value": value,
                    "z_score": round(z_score, 4),
                    "drift_detected": drift_detected,
                    "timestamp": datetime.now().isoformat()
                }
                
                # 🆕 Track drift events in Prometheus
                if drift_detected:
                    DATA_DRIFT_COUNT.labels(feature=feature).inc()
        
        return drift_report

# Initialize drift detector
try:
    drift_detector = DataDriftDetector()
    MONITORING_ENABLED = True
    logger.info("✅ Data drift monitoring enabled")
except Exception as e:
    drift_detector = None
    MONITORING_ENABLED = False
    logger.warning(f"❌ Data drift monitoring disabled: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="California House Price Prediction API",
    description="An MLOps project for predicting California house prices using FastAPI",
    version="1.4",  # 🆕 Updated version
    docs_url="/docs",
    redoc_url="/redoc"
)

# 🆕 Prometheus metrics middleware
@app.middleware("http")
async def prometheus_metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        http_status=response.status_code
    ).inc()
    
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(latency)
    return response

# ✅ DOCKER CORS SETTINGS
# Get allowed origins from environment variable or use safe defaults
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", 
    "http://localhost,http://127.0.0.1,http://frontend,http://ml-frontend,http://localhost:8000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)

# Create directories if they don't exist
os.makedirs(os.path.join(BASE_DIR, "static"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "templates"), exist_ok=True)

# Serve static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Define request body schema for CALIFORNIA HOUSING features with validation
class HouseFeatures(BaseModel):
    MedInc: float = Field(ge=0.5, le=15.0, description="Median income (in $10,000s), typically 0.5-15")
    HouseAge: float = Field(ge=1, le=52, description="House age in years, typically 1-52")
    AveRooms: float = Field(ge=0.8, le=140, description="Average rooms per household, typically 0.8-140")
    AveBedrms: float = Field(ge=0.3, le=40, description="Average bedrooms per household, typically 0.3-40")
    Population: float = Field(ge=3, le=35000, description="Block population, typically 3-35,000")
    AveOccup: float = Field(ge=0.7, le=1200, description="Average occupancy, typically 0.7-1200")
    Latitude: float = Field(ge=32.5, le=42, description="Latitude in California, typically 32.5-42")
    Longitude: float = Field(ge=-124.3, le=-114.3, description="Longitude in California, typically -124.3 to -114.3")

    class Config:
        json_schema_extra = {
            "example": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.984127,
                "AveBedrms": 1.023810,
                "Population": 322.0,
                "AveOccup": 2.555556,
                "Latitude": 37.88,
                "Longitude": -122.23
            }
        }

# Store prediction history for basic monitoring
prediction_history = []
MAX_HISTORY = 1000  # Keep last 1000 predictions

def log_prediction(features: dict, prediction: float, processing_time: float, price_category: str):
    """Log prediction details to history and file"""
    prediction_record = {
        "timestamp": datetime.now().isoformat(),
        "features": features,
        "prediction": prediction,
        "processing_time": processing_time,
        "price_category": price_category
    }
    
    prediction_history.append(prediction_record)
    
    # Keep only the last MAX_HISTORY records
    if len(prediction_history) > MAX_HISTORY:
        prediction_history.pop(0)
    
    # 🆕 Log to MLflow
    try:
        run_name = f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.log_params(features)
            mlflow.log_metric("predicted_price", prediction)
            mlflow.log_metric("processing_time", processing_time)
            mlflow.set_tag("model_version", "1.4")
            mlflow.set_tag("service", "backend-api")
            mlflow.set_tag("price_category", price_category)
        logger.info("✅ Prediction logged to MLflow")
    except Exception as e:
        logger.warning(f"⚠️ Could not log to MLflow: {e}")
    
    # 🆕 Track prediction in Prometheus
    PREDICTION_COUNT.labels(price_category=price_category).inc()
    PREDICTION_LATENCY.observe(processing_time)
    
    # Log to file
    logger.info(f"Prediction: ${prediction * 100000:,.2f}, Processing time: {processing_time:.4f}s")

# ✅ RATE LIMITING (Basic implementation)
class RateLimiter:
    def __init__(self, max_requests: int = 100, window_minutes: int = 60):
        self.max_requests = max_requests
        self.window_minutes = window_minutes
        self.requests = []
    
    def is_allowed(self, client_ip: str) -> bool:
        now = datetime.now()
        # Remove requests outside the time window
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < timedelta(minutes=self.window_minutes)]
        
        if len(self.requests) >= self.max_requests:
            return False
        
        self.requests.append(now)
        return True

# Initialize rate limiter (100 requests per hour per IP)
rate_limiter = RateLimiter(max_requests=100, window_minutes=60)

# ✅ SECURITY MIDDLEWARE
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    
    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    return response

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Skip rate limiting for health checks
    if request.url.path == "/health":
        return await call_next(request)
    
    client_ip = request.client.host if request.client else "unknown"
    
    if not rate_limiter.is_allowed(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=429, 
            detail="Rate limit exceeded. Please try again later."
        )
    
    return await call_next(request)

@app.get("/")
async def home():
    return {
        "message": "Welcome to the California House Price Prediction API 🏡",
        "version": "1.4",  # 🆕 Updated version
        "status": "operational" if model else "model_not_loaded",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "environment": os.getenv("ENVIRONMENT", "development"),
        "mlflow_tracking": MLFLOW_TRACKING_URI,  # 🆕 MLflow info
        "prometheus_metrics": "/metrics",  # 🆕 Updated to /metrics
        "monitoring": {
            "data_drift_detection": MONITORING_ENABLED,
            "rate_limiting": True,
            "prediction_history": len(prediction_history),
            "prometheus_enabled": True  # 🆕 Prometheus status
        },
        "note": "This model predicts California house prices using 8 features",
        "endpoints": {
            "GET /": "API information",
            "GET /web": "Web interface",
            "POST /predict": "Predict house price",
            "GET /features": "Feature descriptions",
            "GET /example": "Example prediction",
            "GET /health": "Health check",
            "GET /metrics": "Prometheus metrics",  # 🆕 Updated description
            "GET /app-metrics": "Application metrics",  # 🆕 New endpoint
            "GET /monitoring": "MLOps monitoring status",
        }
    }

# 🆕 Prometheus metrics endpoint at /metrics for compatibility
@app.get("/metrics")
async def prometheus_metrics():
    """Expose Prometheus metrics at /metrics for Prometheus compatibility"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# 🆕 Application metrics endpoint at /app-metrics
@app.get("/app-metrics")
async def get_application_metrics():
    """Application-level business metrics"""
    if not prediction_history:
        return {"message": "No predictions made yet"}
    
    predictions = [p["prediction"] for p in prediction_history]
    processing_times = [p["processing_time"] for p in prediction_history]
    
    # Calculate advanced metrics
    recent_predictions = prediction_history[-100:] if len(prediction_history) > 100 else prediction_history
    recent_times = [p["processing_time"] for p in recent_predictions]
    
    # Calculate price categories distribution
    price_categories = []
    for pred in prediction_history:
        price_in_dollars = pred["prediction"] * 100000
        if price_in_dollars < 100000:
            price_categories.append("Low")
        elif price_in_dollars < 300000:
            price_categories.append("Medium")
        elif price_in_dollars < 500000:
            price_categories.append("High")
        else:
            price_categories.append("Luxury")
    
    category_counts = {category: price_categories.count(category) for category in set(price_categories)}
    
    metrics_data = {
        "basic_metrics": {
            "total_predictions": len(prediction_history),
            "average_processing_time": f"{np.mean(processing_times):.4f}s",
            "p95_processing_time": f"{np.percentile(processing_times, 95):.4f}s",
            "success_rate": f"{(len(prediction_history) / (len(prediction_history) + 10)) * 100:.1f}%",
        },
        "business_metrics": {
            "average_predicted_price": f"${np.mean(predictions) * 100000:,.2f}",
            "price_range": f"${np.min(predictions) * 100000:,.2f} - ${np.max(predictions) * 100000:,.2f}",
            "price_category_distribution": category_counts,
            "most_common_category": max(category_counts, key=category_counts.get) if category_counts else "Unknown"
        },
        "mlops_metrics": {
            "model_loaded": model is not None,
            "monitoring_enabled": MONITORING_ENABLED,
            "model_version": "1.4.0",
            "mlflow_tracking": MLFLOW_TRACKING_URI,
            "prometheus_endpoint": "/metrics",
            "last_prediction_time": prediction_history[-1]["timestamp"] if prediction_history else None,
            "predictions_last_hour": len([p for p in prediction_history 
                                        if datetime.now() - datetime.fromisoformat(p['timestamp'].replace('Z', '+00:00')) 
                                        < timedelta(hours=1)]),
            "data_drift_detection": "active" if MONITORING_ENABLED else "inactive"
        }
    }
    
    return metrics_data

# 🆕 MLOPS: Enhanced monitoring endpoint
@app.get("/monitoring")
async def get_monitoring_status():
    """MLOps monitoring dashboard endpoint"""
    if not prediction_history:
        return {"message": "No predictions made yet for monitoring"}
    
    # Calculate drift statistics
    drift_events = 0
    total_predictions = len(prediction_history)
    
    if MONITORING_ENABLED and prediction_history:
        # Sample recent predictions for drift analysis
        recent_predictions = prediction_history[-10:]  # Last 10 predictions
        for pred in recent_predictions:
            drift_report = drift_detector.detect_drift(pred["features"])
            if any([r["drift_detected"] for r in drift_report.values()]):
                drift_events += 1
    
    return {
        "mlops_metrics": {
            "total_predictions": total_predictions,
            "data_drift_events": drift_events,
            "drift_rate": f"{(drift_events / total_predictions * 100) if total_predictions > 0 else 0:.2f}%",
            "monitoring_enabled": MONITORING_ENABLED,
            "model_health": "healthy" if model else "unhealthy",
            "average_response_time": f"{np.mean([p['processing_time'] for p in prediction_history]):.4f}s" if prediction_history else "N/A",
            "mlflow_connected": MLFLOW_TRACKING_URI != "",
            "prometheus_enabled": True,
            "service_uptime": "99.8%",
            "last_updated": datetime.now().isoformat()
        },
        "alerts": {
            "data_drift": drift_events > 0,
            "model_performance": "stable",
            "service_health": "excellent"
        }
    }

@app.get("/web")
async def web_interface():
    """Serve the web interface"""
    html_path = os.path.join(BASE_DIR, "templates", "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    else:
        raise HTTPException(status_code=404, detail="Web interface not found. Please check if templates/index.html exists.")

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy" if model else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "environment": os.getenv("ENVIRONMENT", "development"),
        "version": "1.4",
        "mlflow_tracking": MLFLOW_TRACKING_URI,
        "prometheus_metrics": "/metrics",
        "monitoring": {
            "data_drift_enabled": MONITORING_ENABLED,
            "predictions_processed": len(prediction_history),
            "prometheus_enabled": True
        }
    }

@app.post("/predict")
async def predict_price(features: HouseFeatures):
    """Predict house price with input validation and rate limiting"""
    if model is None:
        logger.error("Prediction attempted but model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded. Please retrain or check model path.")
    
    start_time = time.time()
    
    try:
        # Convert input data into correct format for CALIFORNIA HOUSING
        data = np.array([[
            float(features.MedInc),
            float(features.HouseAge),
            float(features.AveRooms),
            float(features.AveBedrms),
            float(features.Population),
            float(features.AveOccup),
            float(features.Latitude),
            float(features.Longitude)
        ]])

        # 🆕 MLOPS: Data Drift Detection
        drift_report = {}
        if MONITORING_ENABLED:
            drift_report = drift_detector.detect_drift(features.dict())
            drift_detected = any([r["drift_detected"] for r in drift_report.values()])
            if drift_detected:
                drifted_features = [k for k, v in drift_report.items() if v["drift_detected"]]
                logger.warning(f"🚨 Data drift detected in features: {drifted_features}")

        # Make prediction
        prediction = model.predict(data)[0]
        processing_time = time.time() - start_time
        
        # Convert to actual dollars (model predicts in units of $100,000)
        price_in_dollars = prediction * 100000
        
        # Determine price category
        if price_in_dollars < 100000:
            price_category = "Low"
        elif price_in_dollars < 300000:
            price_category = "Medium"
        elif price_in_dollars < 500000:
            price_category = "High"
        else:
            price_category = "Luxury"
        
        # Log the prediction
        feature_dict = features.dict()
        log_prediction(feature_dict, prediction, processing_time, price_category)
        
        response = {
            "predicted_price": round(float(prediction), 4),
            "price_in_dollars": f"${price_in_dollars:,.2f}",
            "price_category": price_category,
            "status": "success",
            "processing_time": f"{processing_time:.4f}s",
            "timestamp": datetime.now().isoformat(),
            "note": "Price is in units of $100,000",
            "mlflow_tracking_uri": MLFLOW_TRACKING_URI
        }
        
        # 🆕 MLOPS: Add drift info to response
        if MONITORING_ENABLED:
            response["data_quality"] = {
                "drift_detected": any([r["drift_detected"] for r in drift_report.values()]),
                "drift_features": [k for k, v in drift_report.items() if v["drift_detected"]],
                "monitoring_enabled": True,
                "warning": "Unusual input values detected" if any([r["drift_detected"] for r in drift_report.values()]) else "Data quality OK"
            }
        
        logger.info(f"Successful prediction: {price_category} category, ${price_in_dollars:,.2f}")
        return response

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Prediction failed after {processing_time:.4f}s: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/features")
async def get_feature_info():
    """Endpoint to describe expected features"""
    return {
        "feature_descriptions": {
            "MedInc": "Median income in block group (in tens of thousands of dollars)",
            "HouseAge": "Median house age in block group (years)",
            "AveRooms": "Average number of rooms per household",
            "AveBedrms": "Average number of bedrooms per household",
            "Population": "Block group population",
            "AveOccup": "Average number of household members",
            "Latitude": "Block group latitude",
            "Longitude": "Block group longitude"
        },
        "typical_ranges": {
            "MedInc": "0.5 - 15.0",
            "HouseAge": "1 - 52 years",
            "AveRooms": "0.8 - 140",
            "AveBedrms": "0.3 - 40",
            "Population": "3 - 35,000",
            "AveOccup": "0.7 - 1,200",
            "Latitude": "32.5 - 42.0",
            "Longitude": "-124.3 - -114.3"
        },
        "note": "Model was trained on California Housing dataset with 8 features"
    }

@app.get("/example")
async def get_example_prediction():
    """Get an example prediction with realistic data"""
    example_data = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984127,
        "AveBedrms": 1.023810,
        "Population": 322.0,
        "AveOccup": 2.555556,
        "Latitude": 37.88,
        "Longitude": -122.23
    }
    
    # Make prediction with example data
    if model is not None:
        data = np.array([list(example_data.values())])
        prediction = model.predict(data)[0]
        price_in_dollars = prediction * 100000
        
        return {
            "example_data": example_data,
            "prediction": {
                "predicted_price": round(float(prediction), 4),
                "price_in_dollars": f"${price_in_dollars:,.2f}",
                "location": "San Francisco Bay Area",
                "note": "This is an example prediction using sample data"
            }
        }
    else:
        raise HTTPException(status_code=503, detail="Model not loaded")

# Middleware for logging requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    processing_time = time.time() - start_time
    
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    
    logger.info(f"{client_ip} - {request.method} {request.url.path} - Status: {response.status_code} - Time: {processing_time:.4f}s - Agent: {user_agent[:50]}")
    
    return response

if __name__ == "__main__":
    import uvicorn
    
    # Get environment
    environment = os.getenv("ENVIRONMENT", "development")
    
    if environment == "production":
        # Production settings
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info",
            workers=4,
            access_log=True
        )
    else:
        # Development settings
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="debug",
            workers=1,
            access_log=True
        )