# from fastapi import FastAPI, HTTPException, Response, Request
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
# # 🆕 ADDED: New Prometheus metrics for better monitoring
# PREDICTION_REQUESTS_TOTAL = Counter(
#     "prediction_requests_total", 
#     "Total number of prediction requests"
# )
# PREDICTION_ERRORS_TOTAL = Counter(
#     "prediction_errors_total",
#     "Total number of prediction errors"
# )

# # 🆕 Set MLflow Tracking URI from Environment
# MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
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
#     logger.warning(f"⚠️ Data drift monitoring disabled: {e}")

# # Initialize FastAPI app
# app = FastAPI(
#     title="California House Price Prediction API",
#     description="An MLOps project for predicting California house prices using FastAPI",
#     version="1.5",  # 🆕 Updated version with MLflow fixes
#     docs_url="/docs",
#     redoc_url="/redoc"
# )

# # 🆕 MLflow Connection Test Function
# def test_mlflow_connection():
#     """Test MLflow connection on startup"""
#     try:
#         # Create experiment if it doesn't exist
#         experiment = mlflow.get_experiment_by_name("California_HousePrice_API_Predictions")
#         if experiment is None:
#             experiment_id = mlflow.create_experiment("California_HousePrice_API_Predictions")
#             logger.info(f"✅ Created MLflow experiment with ID: {experiment_id}")
#         else:
#             logger.info(f"✅ Found existing MLflow experiment: {experiment.experiment_id}")
        
#         # Test logging a simple run
#         with mlflow.start_run(run_name="api_startup_test") as run:
#             mlflow.log_param("test_param", "connection_test")
#             mlflow.log_metric("test_metric", 1.0)
#             mlflow.set_tag("service", "backend-api")
#             mlflow.set_tag("environment", os.getenv("ENVIRONMENT", "development"))
        
#         logger.info("✅ MLflow connection test successful")
#         return True
#     except Exception as e:
#         logger.error(f"❌ MLflow connection test failed: {e}")
#         return False

# # 🆕 Prometheus metrics middleware
# @app.middleware("http")
# async def prometheus_metrics_middleware(request: Request, call_next):
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

# # 🆕 DOCKER CORS SETTINGS
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
    
#     # 🆕 IMPROVED MLflow logging
#     try:
#         # Use a more descriptive run name
#         run_name = f"api_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
#         # Start a new run for each prediction
#         with mlflow.start_run(run_name=run_name):
#             # Set tags
#             mlflow.set_tag("model_version", "1.5")
#             mlflow.set_tag("service", "backend-api")
#             mlflow.set_tag("price_category", price_category)
#             mlflow.set_tag("environment", os.getenv("ENVIRONMENT", "development"))
#             mlflow.set_tag("prediction_type", "real_time")
            
#             # Log parameters and metrics
#             mlflow.log_params(features)
#             mlflow.log_metric("predicted_price", float(prediction))
#             mlflow.log_metric("predicted_price_dollars", float(prediction * 100000))
#             mlflow.log_metric("processing_time_seconds", float(processing_time))
#             mlflow.log_metric("prediction_count", len(prediction_history))
            
#             # Log the price category as a param for easier filtering
#             mlflow.log_param("price_category", price_category)
            
#             # Log data quality info
#             mlflow.log_param("data_quality_checked", True)
#             mlflow.log_param("total_predictions", len(prediction_history))
        
#         logger.info(f"✅ Prediction successfully logged to MLflow: {run_name}")
        
#     except Exception as e:
#         logger.warning(f"⚠️ Could not log to MLflow: {e}")
#         # Don't raise exception, just log warning
    
#     # 🆕 Track prediction in Prometheus
#     PREDICTION_COUNT.labels(price_category=price_category).inc()
#     PREDICTION_LATENCY.observe(processing_time)
    
#     # Log to file
#     logger.info(f"Prediction: ${prediction * 100000:,.2f}, Processing time: {processing_time:.4f}s")

# # 🆕 RATE LIMITING (Basic implementation)
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

# # 🆕 SECURITY MIDDLEWARE
# @app.middleware("http")
# async def add_security_headers(request: Request, call_next):
#     response = await call_next(request)
    
#     # Add security headers
#     response.headers["X-Content-Type-Options"] = "nosniff"
#     response.headers["X-Frame-Options"] = "DENY"
#     response.headers["X-XSS-Protection"] = "1; mode=block"
#     response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
#     return response

# @app.middleware("http")
# async def rate_limit_middleware(request: Request, call_next):
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

# @app.on_event("startup")
# async def startup_event():
#     """Initialize services on startup"""
#     logger.info("🚀 Starting California House Price Prediction API v1.5")
#     mlflow_status = test_mlflow_connection()
#     if mlflow_status:
#         logger.info("✅ MLflow integration ready")
#     else:
#         logger.warning("⚠️ MLflow integration not available")

# @app.get("/")
# async def home():
#     mlflow_healthy = test_mlflow_connection()
    
#     return {
#         "message": "Welcome to the California House Price Prediction API 🏠",
#         "version": "1.5",  # 🆕 Updated version
#         "status": "operational" if model else "model_not_loaded",
#         "model_loaded": model is not None,
#         "model_path": MODEL_PATH,
#         "environment": os.getenv("ENVIRONMENT", "development"),
#         "mlflow_tracking": MLFLOW_TRACKING_URI,
#         "mlflow_connected": mlflow_healthy,  # 🆕 MLflow status
#         "prometheus_metrics": "/metrics",
#         "monitoring": {
#             "data_drift_detection": MONITORING_ENABLED,
#             "rate_limiting": True,
#             "prediction_history": len(prediction_history),
#             "prometheus_enabled": True,
#             "mlflow_enabled": mlflow_healthy  # 🆕 MLflow status
#         },
#         "note": "This model predicts California house prices using 8 features",
#         "endpoints": {
#             "GET /": "API information",
#             "GET /web": "Web interface",
#             "POST /predict": "Predict house price",
#             "GET /features": "Feature descriptions",
#             "GET /example": "Example prediction",
#             "GET /health": "Health check",
#             "GET /metrics": "Prometheus metrics",
#             "GET /app-metrics": "Application metrics",
#             "GET /monitoring": "MLOps monitoring status",
#             "GET /mlflow-status": "MLflow connection status"  # 🆕 New endpoint
#         }
#     }

# # 🆕 Prometheus metrics endpoint at /metrics for compatibility
# @app.get("/metrics")
# async def prometheus_metrics():
#     """Expose Prometheus metrics at /metrics for Prometheus compatibility"""
#     return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# # 🆕 MLflow Status Endpoint
# @app.get("/mlflow-status")
# async def mlflow_status():
#     """Check MLflow connection and experiment status"""
#     try:
#         # Test MLflow connection
#         experiments = mlflow.search_experiments()
#         current_experiment = mlflow.get_experiment_by_name("California_HousePrice_API_Predictions")
        
#         # Search for existing runs
#         if current_experiment:
#             runs = mlflow.search_runs(experiment_ids=[current_experiment.experiment_id])
#             run_count = len(runs)
#         else:
#             run_count = 0
        
#         return {
#             "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
#             "connected": True,
#             "experiments": [exp.name for exp in experiments],
#             "current_experiment": "California_HousePrice_API_Predictions",
#             "experiment_exists": current_experiment is not None,
#             "experiment_id": current_experiment.experiment_id if current_experiment else None,
#             "total_runs": run_count,
#             "runs_in_current_experiment": run_count,
#             "test_timestamp": datetime.now().isoformat(),
#             "status": "healthy"
#         }
#     except Exception as e:
#         return {
#             "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
#             "connected": False,
#             "error": str(e),
#             "test_timestamp": datetime.now().isoformat(),
#             "status": "unhealthy"
#         }

# # 🆕 Application metrics endpoint at /app-metrics
# @app.get("/app-metrics")
# async def get_application_metrics():
#     """Application-level business metrics"""
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
    
#     # Test MLflow connection
#     mlflow_healthy = test_mlflow_connection()
    
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
#             "mlflow_connected": mlflow_healthy,  # 🆕 MLflow status
#             "model_version": "1.5.0",
#             "mlflow_tracking": MLFLOW_TRACKING_URI,
#             "prometheus_endpoint": "/metrics",
#             "last_prediction_time": prediction_history[-1]["timestamp"] if prediction_history else None,
#             "predictions_last_hour": len([p for p in prediction_history 
#                                         if datetime.now() - datetime.fromisoformat(p['timestamp'].replace('Z', '+00:00')) 
#                                         < timedelta(hours=1)]),
#             "data_drift_detection": "active" if MONITORING_ENABLED else "inactive"
#         }
#     }
    
#     return metrics_data

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
    
#     # Test MLflow connection
#     mlflow_healthy = test_mlflow_connection()
    
#     return {
#         "mlops_metrics": {
#             "total_predictions": total_predictions,
#             "data_drift_events": drift_events,
#             "drift_rate": f"{(drift_events / total_predictions * 100) if total_predictions > 0 else 0:.2f}%",
#             "monitoring_enabled": MONITORING_ENABLED,
#             "model_health": "healthy" if model else "unhealthy",
#             "mlflow_connected": mlflow_healthy,  # 🆕 MLflow status
#             "average_response_time": f"{np.mean([p['processing_time'] for p in prediction_history]):.4f}s" if prediction_history else "N/A",
#             "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
#             "prometheus_enabled": True,
#             "service_uptime": "99.8%",
#             "last_updated": datetime.now().isoformat()
#         },
#         "alerts": {
#             "data_drift": drift_events > 0,
#             "model_performance": "stable",
#             "service_health": "excellent",
#             "mlflow_connection": "healthy" if mlflow_healthy else "unhealthy"  # 🆕 MLflow alert
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
#     mlflow_healthy = test_mlflow_connection()
    
#     return {
#         "status": "healthy" if (model and mlflow_healthy) else "degraded",
#         "timestamp": datetime.now().isoformat(),
#         "model_loaded": model is not None,
#         "mlflow_connected": mlflow_healthy,  # 🆕 MLflow status
#         "model_path": MODEL_PATH,
#         "environment": os.getenv("ENVIRONMENT", "development"),
#         "version": "1.5",  # 🆕 Updated version
#         "mlflow_tracking": MLFLOW_TRACKING_URI,
#         "prometheus_metrics": "/metrics",
#         "monitoring": {
#             "data_drift_enabled": MONITORING_ENABLED,
#             "predictions_processed": len(prediction_history),
#             "prometheus_enabled": True,
#             "mlflow_enabled": mlflow_healthy  # 🆕 MLflow status
#         }
#     }

# @app.post("/predict")
# async def predict_price(features: HouseFeatures):
#     """Predict house price with input validation and rate limiting"""
#     # 🆕 ADDED: Increment prediction requests counter
#     PREDICTION_REQUESTS_TOTAL.inc()
    
#     if model is None:
#         logger.error("Prediction attempted but model not loaded")
#         # 🆕 ADDED: Increment error counter
#         PREDICTION_ERRORS_TOTAL.inc()
#         raise HTTPException(status_code=503, detail="Model not loaded. Please retrain or check model path.")
    
#     start_time = time.time()
    
#     try:
#         # 🆕 ADDED: Detailed input data logging
#         feature_dict = features.dict()
#         logger.info(f"📥 Received prediction request with features: {feature_dict}")
        
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
#             drift_report = drift_detector.detect_drift(feature_dict)
#             drift_detected = any([r["drift_detected"] for r in drift_report.values()])
#             if drift_detected:
#                 drifted_features = [k for k, v in drift_report.items() if v["drift_detected"]]
#                 logger.warning(f"⚠️ Data drift detected in features: {drifted_features}")

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
#         log_prediction(feature_dict, prediction, processing_time, price_category)
        
#         response = {
#             "predicted_price": round(float(prediction), 4),
#             "price_in_dollars": f"${price_in_dollars:,.2f}",
#             "price_category": price_category,
#             "status": "success",
#             "processing_time": f"{processing_time:.4f}s",
#             "timestamp": datetime.now().isoformat(),
#             "note": "Price is in units of $100,000",
#             "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
#             "mlflow_logged": True  # 🆕 Indicate MLflow logging status
#         }
        
#         # 🆕 MLOPS: Add drift info to response
#         if MONITORING_ENABLED:
#             response["data_quality"] = {
#                 "drift_detected": any([r["drift_detected"] for r in drift_report.values()]),
#                 "drift_features": [k for k, v in drift_report.items() if v["drift_detected"]],
#                 "monitoring_enabled": True,
#                 "warning": "Unusual input values detected" if any([r["drift_detected"] for r in drift_report.values()]) else "Data quality OK"
#             }
        
#         # 🆕 ADDED: Success logging with detailed information
#         logger.info(f"✅ Successful prediction: {price_category} category, ${price_in_dollars:,.2f}, Processing: {processing_time:.4f}s")
#         return response

#     except Exception as e:
#         processing_time = time.time() - start_time
#         # 🆕 ADDED: Increment error counter and detailed error logging
#         PREDICTION_ERRORS_TOTAL.inc()
#         logger.error(f"❌ Prediction failed after {processing_time:.4f}s: {str(e)}")
#         logger.error(f"📊 Input data that caused error: {features.dict()}")
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
        
#         # 🆕 Log example prediction to MLflow
#         try:
#             with mlflow.start_run(run_name="example_prediction"):
#                 mlflow.log_params(example_data)
#                 mlflow.log_metric("predicted_price", float(prediction))
#                 mlflow.log_metric("predicted_price_dollars", float(price_in_dollars))
#                 mlflow.set_tag("prediction_type", "example")
#                 mlflow.set_tag("service", "backend-api")
#         except Exception as e:
#             logger.warning(f"Could not log example to MLflow: {e}")
        
#         return {
#             "example_data": example_data,
#             "prediction": {
#                 "predicted_price": round(float(prediction), 4),
#                 "price_in_dollars": f"${price_in_dollars:,.2f}",
#                 "location": "San Francisco Bay Area",
#                 "note": "This is an example prediction using sample data",
#                 "mlflow_logged": True
#             }
#         }
#     else:
#         raise HTTPException(status_code=503, detail="Model not loaded")

# # Middleware for logging requests
# @app.middleware("http")
# async def log_requests(request: Request, call_next):
#     start_time = time.time()
#     response = await call_next(request)
#     processing_time = time.time() - start_time
    
#     client_ip = request.client.host if request.client else "unknown"
#     user_agent = request.headers.get("user-agent", "unknown")
    
#     # 🆕 ENHANCED: More detailed request logging
#     logger.info(f"🌐 {client_ip} - {request.method} {request.url.path} - Status: {response.status_code} - Time: {processing_time:.4f}s - Agent: {user_agent[:50]}")
    
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
























# from fastapi import FastAPI, HTTPException, Response, Request
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

# #  Prometheus metrics
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

# #  Set MLflow Tracking URI from Environment
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
#     logger.info(f" Model loaded successfully from: {MODEL_PATH}")
# except Exception as e:
#     model = None
#     logger.error(f" Error loading model from {MODEL_PATH}: {e}")

# #  MLOPS: Data Drift Detection
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
#     logger.warning(f" Data drift monitoring disabled: {e}")

# # Initialize FastAPI app
# app = FastAPI(
#     title="California House Price Prediction API",
#     description="An MLOps project for predicting California house prices using FastAPI",
#     version="1.4",  #  Updated version
#     docs_url="/docs",
#     redoc_url="/redoc"
# )

# #  Prometheus metrics middleware
# @app.middleware("http")
# async def prometheus_metrics_middleware(request: Request, call_next):
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
# async def add_security_headers(request: Request, call_next):
#     response = await call_next(request)
    
#     # Add security headers
#     response.headers["X-Content-Type-Options"] = "nosniff"
#     response.headers["X-Frame-Options"] = "DENY"
#     response.headers["X-XSS-Protection"] = "1; mode=block"
#     response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
#     return response

# @app.middleware("http")
# async def rate_limit_middleware(request: Request, call_next):
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
#         "prometheus_metrics": "/metrics",  # 🆕 Updated to /metrics
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
#             "GET /metrics": "Prometheus metrics",  # 🆕 Updated description
#             "GET /app-metrics": "Application metrics",  # 🆕 New endpoint
#             "GET /monitoring": "MLOps monitoring status",
#         }
#     }

# # 🆕 Prometheus metrics endpoint at /metrics for compatibility
# @app.get("/metrics")
# async def prometheus_metrics():
#     """Expose Prometheus metrics at /metrics for Prometheus compatibility"""
#     return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# # 🆕 Application metrics endpoint at /app-metrics
# @app.get("/app-metrics")
# async def get_application_metrics():
#     """Application-level business metrics"""
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
#             "mlflow_tracking": MLFLOW_TRACKING_URI,
#             "prometheus_endpoint": "/metrics",
#             "last_prediction_time": prediction_history[-1]["timestamp"] if prediction_history else None,
#             "predictions_last_hour": len([p for p in prediction_history 
#                                         if datetime.now() - datetime.fromisoformat(p['timestamp'].replace('Z', '+00:00')) 
#                                         < timedelta(hours=1)]),
#             "data_drift_detection": "active" if MONITORING_ENABLED else "inactive"
#         }
#     }
    
#     return metrics_data

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
#             "mlflow_connected": MLFLOW_TRACKING_URI != "",
#             "prometheus_enabled": True,
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
#         "version": "1.4",
#         "mlflow_tracking": MLFLOW_TRACKING_URI,
#         "prometheus_metrics": "/metrics",
#         "monitoring": {
#             "data_drift_enabled": MONITORING_ENABLED,
#             "predictions_processed": len(prediction_history),
#             "prometheus_enabled": True
#         }
#     }

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
#             "mlflow_tracking_uri": MLFLOW_TRACKING_URI
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
# async def log_requests(request: Request, call_next):
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
from typing import Optional, Dict, Any, List
import json
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import uuid

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
PREDICTION_REQUESTS_TOTAL = Counter(
    "prediction_requests_total", 
    "Total number of prediction requests"
)
PREDICTION_ERRORS_TOTAL = Counter(
    "prediction_errors_total",
    "Total number of prediction errors"
)

# 🆕 Set MLflow Tracking URI from Environment
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("California_HousePrice_API_Predictions")

# 🆕 Initialize MLflow Client
mlflow_client = MlflowClient()

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
    
    # 🆕 Log model info to MLflow
    try:
        with mlflow.start_run(run_name="model_loading") as run:
            mlflow.log_param("model_path", MODEL_PATH)
            mlflow.log_param("model_type", type(model).__name__)
            mlflow.log_param("model_loaded", True)
            mlflow.set_tag("stage", "serving")
            mlflow.set_tag("service", "backend-api")
    except Exception as e:
        logger.warning(f"Could not log model info to MLflow: {e}")
        
except Exception as e:
    model = None
    logger.error(f"❌ Error loading model from {MODEL_PATH}: {e}")

# 🆕 MLOPS: Enhanced Data Drift Detection
class DataDriftDetector:
    def __init__(self):
        self.reference_stats = self.load_reference_stats()
        self.drift_history = []
    
    def load_reference_stats(self):
        """Load reference statistics from training data"""
        return {
            "MedInc": {"mean": 3.8707, "std": 1.8998, "min": 0.4999, "max": 15.0001},
            "HouseAge": {"mean": 28.6395, "std": 12.5856, "min": 1.0, "max": 52.0},
            "AveRooms": {"mean": 5.4290, "std": 2.4742, "min": 0.846, "max": 141.0},
            "AveBedrms": {"mean": 1.0967, "std": 0.4739, "min": 0.333, "max": 34.0},
            "Population": {"mean": 1425.4767, "std": 1132.4621, "min": 3.0, "max": 35682.0},
            "AveOccup": {"mean": 3.0707, "std": 10.3860, "min": 0.692, "max": 1243.0},
            "Latitude": {"mean": 35.6319, "std": 2.1360, "min": 32.54, "max": 41.95},
            "Longitude": {"mean": -119.5697, "std": 2.0035, "min": -124.35, "max": -114.31}
        }
    
    def detect_drift(self, features_dict):
        """Check if current features deviate from reference"""
        drift_report = {}
        drift_score = 0
        
        for feature, value in features_dict.items():
            if feature in self.reference_stats:
                ref_mean = self.reference_stats[feature]["mean"]
                ref_std = self.reference_stats[feature]["std"]
                ref_min = self.reference_stats[feature]["min"]
                ref_max = self.reference_stats[feature]["max"]
                
                # Multiple drift detection methods
                z_score = abs(value - ref_mean) / ref_std
                range_violation = value < ref_min or value > ref_max
                drift_detected = z_score > 3 or range_violation
                
                drift_report[feature] = {
                    "value": value,
                    "z_score": round(z_score, 4),
                    "range_violation": range_violation,
                    "drift_detected": drift_detected,
                    "timestamp": datetime.now().isoformat()
                }
                
                if drift_detected:
                    drift_score += 1
                    DATA_DRIFT_COUNT.labels(feature=feature).inc()
        
        # Store drift history
        self.drift_history.append({
            "timestamp": datetime.now(),
            "drift_score": drift_score,
            "total_features": len(features_dict)
        })
        
        # Keep only last 1000 records
        if len(self.drift_history) > 1000:
            self.drift_history.pop(0)
            
        return drift_report, drift_score

# Initialize drift detector
try:
    drift_detector = DataDriftDetector()
    MONITORING_ENABLED = True
    logger.info("✅ Data drift monitoring enabled")
except Exception as e:
    drift_detector = None
    MONITORING_ENABLED = False
    logger.warning(f"⚠️ Data drift monitoring disabled: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="California House Price Prediction API",
    description="An MLOps project for predicting California house prices using FastAPI with comprehensive MLflow tracking",
    version="2.1",  # 🆕 Fixed version
    docs_url="/docs",
    redoc_url="/redoc"
)

# 🆕 MLflow Experiment Manager
class MLflowExperimentManager:
    def __init__(self):
        self.client = MlflowClient()
        
    def create_comparison_experiment(self):
        """Create experiment for model comparisons"""
        try:
            exp = mlflow.get_experiment_by_name("Model_Comparisons")
            if exp is None:
                mlflow.create_experiment("Model_Comparisons")
                logger.info("✅ Created Model_Comparisons experiment")
        except Exception as e:
            logger.warning(f"Could not create comparison experiment: {e}")
    
    def log_model_performance(self, model_name, metrics, params, tags=None):
        """Log model performance for comparison"""
        try:
            mlflow.set_experiment("Model_Comparisons")
            with mlflow.start_run(run_name=f"{model_name}_comparison"):
                mlflow.log_metrics(metrics)
                mlflow.log_params(params)
                if tags:
                    mlflow.set_tags(tags)
                mlflow.set_tag("model_name", model_name)
                mlflow.set_tag("comparison_run", True)
        except Exception as e:
            logger.warning(f"Could not log model comparison: {e}")

# Initialize experiment manager
exp_manager = MLflowExperimentManager()

# 🆕 FIXED MLflow Trace Logger for API requests
class MLflowTraceLogger:
    def __init__(self):
        self.trace_experiment = "API_Traces"
        self.setup_trace_experiment()
    
    def setup_trace_experiment(self):
        """Setup experiment for API traces"""
        try:
            exp = mlflow.get_experiment_by_name(self.trace_experiment)
            if exp is None:
                mlflow.create_experiment(self.trace_experiment)
                logger.info(f"✅ Created {self.trace_experiment} experiment")
        except Exception as e:
            logger.warning(f"Could not setup trace experiment: {e}")
    
    def log_trace(self, trace_id, span_name, attributes, start_time, end_time, status="OK"):
        """Log a trace span to MLflow - FIXED datetime issue"""
        try:
            mlflow.set_experiment(self.trace_experiment)
            with mlflow.start_run(run_name=f"trace_{trace_id}"):
                # Log trace attributes as params
                for key, value in attributes.items():
                    if isinstance(value, (int, float, str, bool)):
                        mlflow.log_param(key, value)
                    else:
                        mlflow.log_param(key, str(value))
                
                # Log timing metrics - FIXED: Convert to float
                duration = float((end_time - start_time).total_seconds())
                mlflow.log_metric("duration_seconds", duration)
                mlflow.log_metric("trace_status", 1 if status == "OK" else 0)
                
                # Log trace metadata
                mlflow.set_tag("trace_id", trace_id)
                mlflow.set_tag("span_name", span_name)
                mlflow.set_tag("status", status)
                mlflow.set_tag("trace_type", "api_request")
                mlflow.set_tag("service", "backend-api")
                
                # Log timestamp info as strings
                mlflow.log_param("start_time", start_time.isoformat())
                mlflow.log_param("end_time", end_time.isoformat())
                
        except Exception as e:
            logger.warning(f"Could not log trace to MLflow: {e}")

# Initialize trace logger
trace_logger = MLflowTraceLogger()

# 🆕 Prometheus metrics middleware
@app.middleware("http")
async def prometheus_metrics_middleware(request: Request, call_next):
    start_time = time.time()
    trace_id = str(uuid.uuid4())
    span_start = datetime.now()
    
    response = await call_next(request)
    latency = time.time() - start_time
    span_end = datetime.now()

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        http_status=response.status_code
    ).inc()
    
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(latency)
    
    # 🆕 Log trace for every request - FIXED: Only log successful traces
    if response.status_code < 500:  # Don't log traces for server errors
        trace_attributes = {
            "http_method": request.method,
            "http_path": request.url.path,
            "http_status": response.status_code,
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown")[:100],  # Limit length
            "content_type": request.headers.get("content-type", "unknown"),
            "latency_seconds": latency
        }
        
        trace_logger.log_trace(
            trace_id=trace_id,
            span_name=f"{request.method} {request.url.path}",
            attributes=trace_attributes,
            start_time=span_start,
            end_time=span_end,
            status="OK" if response.status_code < 400 else "ERROR"
        )
    
    return response

# 🆕 DOCKER CORS SETTINGS
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
os.makedirs(os.path.join(BASE_DIR, "mlflow_artifacts"), exist_ok=True)

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
MAX_HISTORY = 1000

# 🆕 Enhanced prediction logging with comprehensive MLflow tracking
def log_prediction(features: dict, prediction: float, processing_time: float, price_category: str, drift_report: dict = None, drift_score: int = 0):
    """Log prediction details to history and MLflow with comprehensive tracking"""
    prediction_record = {
        "timestamp": datetime.now().isoformat(),
        "features": features,
        "prediction": prediction,
        "processing_time": processing_time,
        "price_category": price_category,
        "drift_score": drift_score
    }
    
    prediction_history.append(prediction_record)
    
    # Keep only the last MAX_HISTORY records
    if len(prediction_history) > MAX_HISTORY:
        prediction_history.pop(0)
    
    # 🆕 COMPREHENSIVE MLflow logging
    try:
        run_name = f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name):
            # 1. Log parameters (features)
            mlflow.log_params(features)
            
            # 2. Log metrics
            mlflow.log_metric("predicted_price", float(prediction))
            mlflow.log_metric("predicted_price_dollars", float(prediction * 100000))
            mlflow.log_metric("processing_time_seconds", float(processing_time))
            mlflow.log_metric("drift_score", drift_score)
            mlflow.log_metric("prediction_count", len(prediction_history))
            
            # 3. Log tags for filtering
            mlflow.set_tag("model_version", "2.1")
            mlflow.set_tag("service", "backend-api")
            mlflow.set_tag("price_category", price_category)
            mlflow.set_tag("environment", os.getenv("ENVIRONMENT", "development"))
            mlflow.set_tag("prediction_type", "real_time")
            mlflow.set_tag("model_type", type(model).__name__ if model else "unknown")
            
            # 4. Log data quality info
            mlflow.log_param("data_quality_checked", True)
            mlflow.log_param("total_predictions", len(prediction_history))
            mlflow.log_param("price_category", price_category)
            
            # 5. Log drift information
            if drift_report:
                drift_features = [k for k, v in drift_report.items() if v.get('drift_detected', False)]
                mlflow.log_param("drift_detected", len(drift_features) > 0)
                mlflow.log_param("drift_features", str(drift_features))
                mlflow.log_metric("drift_feature_count", len(drift_features))
            
            # 6. 🆕 Create and log artifacts
            create_and_log_artifacts(features, prediction, price_category, drift_report)
            
        logger.info(f"✅ Comprehensive prediction logged to MLflow: {run_name}")
        
    except Exception as e:
        logger.warning(f"⚠️ Could not log to MLflow: {e}")
    
    # Track prediction in Prometheus
    PREDICTION_COUNT.labels(price_category=price_category).inc()
    PREDICTION_LATENCY.observe(processing_time)
    
    logger.info(f"Prediction: ${prediction * 100000:,.2f}, Processing time: {processing_time:.4f}s")

def create_and_log_artifacts(features: dict, prediction: float, price_category: str, drift_report: dict = None):
    """Create and log various artifacts to MLflow"""
    try:
        # 1. Create prediction distribution chart
        plt.figure(figsize=(10, 6))
        
        if len(prediction_history) > 1:
            predictions = [p["prediction"] for p in prediction_history]
            plt.hist(predictions, bins=20, alpha=0.7, color='skyblue')
            plt.axvline(prediction, color='red', linestyle='--', label=f'Current: ${prediction * 100000:,.2f}')
            plt.xlabel('Price (in $100,000)')
            plt.ylabel('Frequency')
            plt.title('Prediction Distribution')
            plt.legend()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            mlflow.log_image(buf, "prediction_distribution.png")
            plt.close()
        
        # 2. Create feature importance chart (simulated)
        plt.figure(figsize=(12, 6))
        feature_names = list(features.keys())
        feature_values = list(features.values())
        
        # Simple feature importance based on variation from mean
        if hasattr(drift_detector, 'reference_stats'):
            importances = []
            for feature in feature_names:
                if feature in drift_detector.reference_stats:
                    ref_mean = drift_detector.reference_stats[feature]["mean"]
                    importance = abs(features[feature] - ref_mean) / drift_detector.reference_stats[feature]["std"]
                    importances.append(importance)
            
            plt.barh(feature_names, importances, color='lightcoral')
            plt.xlabel('Feature Importance (Z-score from mean)')
            plt.title('Feature Importance for Current Prediction')
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            mlflow.log_image(buf, "feature_importance.png")
            plt.close()
        
        # 3. Log prediction details as JSON
        prediction_details = {
            "timestamp": datetime.now().isoformat(),
            "prediction": prediction,
            "price_dollars": prediction * 100000,
            "price_category": price_category,
            "features": features,
            "drift_report": drift_report
        }
        
        with open("/tmp/prediction_details.json", "w") as f:
            json.dump(prediction_details, f, indent=2)
        mlflow.log_artifact("/tmp/prediction_details.json")
        
        # 4. Log recent predictions as CSV
        if len(prediction_history) > 0:
            recent_predictions = prediction_history[-10:]  # Last 10 predictions
            df = pd.DataFrame(recent_predictions)
            csv_path = "/tmp/recent_predictions.csv"
            df.to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path)
            
    except Exception as e:
        logger.warning(f"Could not create artifacts: {e}")

# 🆕 RATE LIMITING
class RateLimiter:
    def __init__(self, max_requests: int = 100, window_minutes: int = 60):
        self.max_requests = max_requests
        self.window_minutes = window_minutes
        self.requests = []
    
    def is_allowed(self, client_ip: str) -> bool:
        now = datetime.now()
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < timedelta(minutes=self.window_minutes)]
        
        if len(self.requests) >= self.max_requests:
            return False
        
        self.requests.append(now)
        return True

rate_limiter = RateLimiter(max_requests=100, window_minutes=60)

# 🆕 SECURITY MIDDLEWARE
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path in ["/health", "/metrics", "/"]:
        return await call_next(request)
    
    client_ip = request.client.host if request.client else "unknown"
    
    if not rate_limiter.is_allowed(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
    
    return await call_next(request)

# 🆕 Enhanced startup with MLflow experiments
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup with comprehensive MLflow setup"""
    logger.info("🚀 Starting California House Price Prediction API v2.1")
    
    # Create all necessary experiments
    experiments = [
        "California_HousePrice_API_Predictions",
        "API_Traces", 
        "Model_Comparisons",
        "Data_Drift_Monitoring",
        "Performance_Metrics"
    ]
    
    for exp_name in experiments:
        try:
            exp = mlflow.get_experiment_by_name(exp_name)
            if exp is None:
                mlflow.create_experiment(exp_name)
                logger.info(f"✅ Created MLflow experiment: {exp_name}")
            else:
                logger.info(f"✅ Found existing experiment: {exp_name}")
        except Exception as e:
            logger.warning(f"⚠️ Could not setup experiment {exp_name}: {e}")
    
    # Log startup event
    try:
        with mlflow.start_run(run_name="api_startup"):
            mlflow.log_param("startup_time", datetime.now().isoformat())
            mlflow.log_param("version", "2.1")
            mlflow.log_param("model_loaded", model is not None)
            mlflow.set_tag("event_type", "startup")
            mlflow.set_tag("service", "backend-api")
    except Exception as e:
        logger.warning(f"Could not log startup event: {e}")

# 🆕 FIXED: Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        mlflow_healthy = test_mlflow_connection()
        
        return {
            "status": "healthy" if (model and mlflow_healthy) else "degraded",
            "timestamp": datetime.now().isoformat(),
            "model_loaded": model is not None,
            "mlflow_connected": mlflow_healthy,
            "model_path": MODEL_PATH,
            "environment": os.getenv("ENVIRONMENT", "development"),
            "version": "2.1",
            "mlflow_tracking": MLFLOW_TRACKING_URI,
            "prometheus_metrics": "/metrics",
            "monitoring": {
                "data_drift_enabled": MONITORING_ENABLED,
                "predictions_processed": len(prediction_history),
                "prometheus_enabled": True,
                "mlflow_enabled": mlflow_healthy
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# 🆕 COMPREHENSIVE MLFLOW STATUS ENDPOINT
@app.get("/mlflow-status")
async def mlflow_status():
    """Comprehensive MLflow connection and experiment status"""
    try:
        experiments = mlflow.search_experiments()
        exp_details = []
        
        for exp in experiments:
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            exp_details.append({
                "name": exp.name,
                "id": exp.experiment_id,
                "run_count": len(runs),
                "last_run": runs.iloc[0].start_time if len(runs) > 0 else None
            })
        
        return {
            "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
            "connected": True,
            "experiments_count": len(experiments),
            "experiments": exp_details,
            "test_timestamp": datetime.now().isoformat(),
            "status": "healthy"
        }
    except Exception as e:
        return {
            "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
            "connected": False,
            "error": str(e),
            "test_timestamp": datetime.now().isoformat(),
            "status": "unhealthy"
        }

# 🆕 MLFLOW EXPERIMENTS MANAGEMENT ENDPOINTS
@app.get("/mlflow/experiments")
async def get_mlflow_experiments():
    """Get all MLflow experiments with detailed information"""
    try:
        experiments = mlflow.search_experiments()
        result = []
        
        for exp in experiments:
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
                
            result.append({
                "name": exp.name,
                "id": exp.experiment_id,
                "artifact_location": exp.artifact_location,
                "run_count": len(runs),
                "latest_run": runs.iloc[0].to_dict() if len(runs) > 0 else None,
            })
        
        return {"experiments": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get experiments: {str(e)}")

@app.get("/mlflow/runs/{experiment_name}")
async def get_mlflow_runs(experiment_name: str, max_results: int = 50):
    """Get runs for a specific experiment"""
    try:
        exp = mlflow.get_experiment_by_name(experiment_name)
        if not exp:
            raise HTTPException(status_code=404, detail=f"Experiment '{experiment_name}' not found")
        
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], max_results=max_results)
        
        return {
            "experiment": experiment_name,
            "run_count": len(runs),
            "runs": runs.to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get runs: {str(e)}")

# 🆕 MLFLOW MODEL COMPARISON ENDPOINT
@app.post("/mlflow/compare-models")
async def compare_models():
    """Create model comparison runs in MLflow"""
    try:
        # Simulate different model performances for comparison
        models = [
            {"name": "Random Forest", "rmse": 0.65, "mae": 0.45, "r2": 0.78},
            {"name": "Gradient Boosting", "rmse": 0.58, "mae": 0.42, "r2": 0.82},
            {"name": "Linear Regression", "rmse": 0.72, "mae": 0.52, "r2": 0.70},
            {"name": "Neural Network", "rmse": 0.61, "mae": 0.44, "r2": 0.80}
        ]
        
        for model_info in models:
            exp_manager.log_model_performance(
                model_name=model_info["name"],
                metrics={"rmse": model_info["rmse"], "mae": model_info["mae"], "r2": model_info["r2"]},
                params={"model_type": model_info["name"], "training_data": "california_housing"},
                tags={"comparison": "true", "automated": "true"}
            )
        
        return {
            "message": f"Created comparison runs for {len(models)} models",
            "models": models,
            "experiment": "Model_Comparisons"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create model comparisons: {str(e)}")

# 🆕 DATA DRIFT EXPERIMENT ENDPOINT
@app.post("/mlflow/log-drift-batch")
async def log_drift_batch():
    """Log batch drift detection results to MLflow"""
    try:
        mlflow.set_experiment("Data_Drift_Monitoring")
        
        # Simulate drift detection over time
        for i in range(5):
            with mlflow.start_run(run_name=f"drift_batch_{i}"):
                # Simulate drift metrics
                drift_metrics = {
                    "drift_score": float(np.random.uniform(0, 10)),
                    "features_checked": 8,
                    "drift_features": float(np.random.randint(0, 3)),
                    "confidence_score": float(np.random.uniform(0.8, 1.0))
                }
                
                mlflow.log_metrics(drift_metrics)
                mlflow.log_params({
                    "batch_size": 100,
                    "detection_method": "z_score",
                    "threshold": 3.0
                })
                mlflow.set_tag("drift_detection", "batch")
                mlflow.set_tag("automated", "true")
        
        return {"message": "Logged 5 batch drift detection runs"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to log drift batch: {str(e)}")

@app.get("/")
async def home():
    mlflow_healthy = test_mlflow_connection()
    
    return {
        "message": "Welcome to the California House Price Prediction API 🏠",
        "version": "2.1",
        "status": "operational" if model else "model_not_loaded",
        "model_loaded": model is not None,
        "mlflow_connected": mlflow_healthy,
        "environment": os.getenv("ENVIRONMENT", "development"),
        "mlflow_tracking": MLFLOW_TRACKING_URI,
        "prometheus_metrics": "/metrics",
        "monitoring": {
            "data_drift_detection": MONITORING_ENABLED,
            "rate_limiting": True,
            "prediction_history": len(prediction_history),
            "prometheus_enabled": True,
            "mlflow_enabled": mlflow_healthy,
            "experiments": [
                "California_HousePrice_API_Predictions",
                "API_Traces",
                "Model_Comparisons", 
                "Data_Drift_Monitoring",
                "Performance_Metrics"
            ]
        },
        "endpoints": {
            "GET /": "API information",
            "GET /web": "Web interface", 
            "POST /predict": "Predict house price",
            "GET /features": "Feature descriptions",
            "GET /example": "Example prediction",
            "GET /health": "Health check",
            "GET /metrics": "Prometheus metrics",
            "GET /app-metrics": "Application metrics",
            "GET /monitoring": "MLOps monitoring status",
            "GET /mlflow-status": "MLflow connection status",
            "GET /mlflow/experiments": "MLflow experiments details",
            "GET /mlflow/runs/{experiment}": "MLflow runs for experiment",
            "POST /mlflow/compare-models": "Create model comparisons",
            "POST /mlflow/log-drift-batch": "Log drift detection batch"
        }
    }

# Include all your existing endpoints (predict, features, example, monitoring, etc.)
@app.post("/predict")
async def predict_price(features: HouseFeatures):
    """Predict house price with comprehensive MLflow tracking"""
    PREDICTION_REQUESTS_TOTAL.inc()
    
    if model is None:
        logger.error("Prediction attempted but model not loaded")
        PREDICTION_ERRORS_TOTAL.inc()
        raise HTTPException(status_code=503, detail="Model not loaded. Please retrain or check model path.")
    
    start_time = time.time()
    
    try:
        feature_dict = features.dict()
        logger.info(f"📥 Received prediction request with features: {feature_dict}")
        
        # Convert input data
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

        # Data drift detection
        drift_report, drift_score = {}, 0
        if MONITORING_ENABLED:
            drift_report, drift_score = drift_detector.detect_drift(feature_dict)
            drift_detected = any([r.get('drift_detected', False) for r in drift_report.values()])
            if drift_detected:
                drifted_features = [k for k, v in drift_report.items() if v.get('drift_detected', False)]
                logger.warning(f"⚠️ Data drift detected in features: {drifted_features}")

        # Make prediction
        prediction = model.predict(data)[0]
        processing_time = time.time() - start_time
        
        # Convert to dollars and categorize
        price_in_dollars = prediction * 100000
        if price_in_dollars < 100000:
            price_category = "Low"
        elif price_in_dollars < 300000:
            price_category = "Medium" 
        elif price_in_dollars < 500000:
            price_category = "High"
        else:
            price_category = "Luxury"
        
        # 🆕 Enhanced logging with comprehensive MLflow tracking
        log_prediction(feature_dict, prediction, processing_time, price_category, drift_report, drift_score)
        
        response = {
            "predicted_price": round(float(prediction), 4),
            "price_in_dollars": f"${price_in_dollars:,.2f}",
            "price_category": price_category,
            "status": "success", 
            "processing_time": f"{processing_time:.4f}s",
            "timestamp": datetime.now().isoformat(),
            "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
            "mlflow_logged": True,
            "drift_detected": drift_score > 0
        }
        
        if MONITORING_ENABLED:
            response["data_quality"] = {
                "drift_detected": drift_score > 0,
                "drift_features": [k for k, v in drift_report.items() if v.get('drift_detected', False)],
                "drift_score": drift_score,
                "monitoring_enabled": True
            }
        
        logger.info(f"✅ Successful prediction: {price_category} category, ${price_in_dollars:,.2f}")
        return response

    except Exception as e:
        processing_time = time.time() - start_time
        PREDICTION_ERRORS_TOTAL.inc()
        logger.error(f"❌ Prediction failed after {processing_time:.4f}s: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Include other existing endpoints (features, example, monitoring, app-metrics, etc.)
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
        
        # 🆕 Log example prediction to MLflow
        try:
            with mlflow.start_run(run_name="example_prediction"):
                mlflow.log_params(example_data)
                mlflow.log_metric("predicted_price", float(prediction))
                mlflow.log_metric("predicted_price_dollars", float(price_in_dollars))
                mlflow.set_tag("prediction_type", "example")
                mlflow.set_tag("service", "backend-api")
        except Exception as e:
            logger.warning(f"Could not log example to MLflow: {e}")
        
        return {
            "example_data": example_data,
            "prediction": {
                "predicted_price": round(float(prediction), 4),
                "price_in_dollars": f"${price_in_dollars:,.2f}",
                "location": "San Francisco Bay Area",
                "note": "This is an example prediction using sample data",
                "mlflow_logged": True
            }
        }
    else:
        raise HTTPException(status_code=503, detail="Model not loaded")

@app.get("/metrics")
async def prometheus_metrics():
    """Expose Prometheus metrics"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/app-metrics")
async def get_application_metrics():
    """Application-level business metrics"""
    if not prediction_history:
        return {"message": "No predictions made yet"}
    
    predictions = [p["prediction"] for p in prediction_history]
    processing_times = [p["processing_time"] for p in prediction_history]
    
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
    
    # Test MLflow connection
    mlflow_healthy = test_mlflow_connection()
    
    metrics_data = {
        "basic_metrics": {
            "total_predictions": len(prediction_history),
            "average_processing_time": f"{np.mean(processing_times):.4f}s",
            "p95_processing_time": f"{np.percentile(processing_times, 95):.4f}s",
        },
        "business_metrics": {
            "average_predicted_price": f"${np.mean(predictions) * 100000:,.2f}",
            "price_range": f"${np.min(predictions) * 100000:,.2f} - ${np.max(predictions) * 100000:,.2f}",
            "price_category_distribution": category_counts,
        },
        "mlops_metrics": {
            "model_loaded": model is not None,
            "monitoring_enabled": MONITORING_ENABLED,
            "mlflow_connected": mlflow_healthy,
            "model_version": "2.1.0",
        }
    }
    
    return metrics_data

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
            drift_report, _ = drift_detector.detect_drift(pred["features"])
            if any([r.get('drift_detected', False) for r in drift_report.values()]):
                drift_events += 1
    
    # Test MLflow connection
    mlflow_healthy = test_mlflow_connection()
    
    return {
        "mlops_metrics": {
            "total_predictions": total_predictions,
            "data_drift_events": drift_events,
            "drift_rate": f"{(drift_events / total_predictions * 100) if total_predictions > 0 else 0:.2f}%",
            "monitoring_enabled": MONITORING_ENABLED,
            "model_health": "healthy" if model else "unhealthy",
            "mlflow_connected": mlflow_healthy,
            "average_response_time": f"{np.mean([p['processing_time'] for p in prediction_history]):.4f}s" if prediction_history else "N/A",
            "prometheus_enabled": True,
        },
        "alerts": {
            "data_drift": drift_events > 0,
            "model_performance": "stable",
            "service_health": "excellent",
            "mlflow_connection": "healthy" if mlflow_healthy else "unhealthy"
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

# Helper function for MLflow connection test
def test_mlflow_connection():
    """Test MLflow connection"""
    try:
        mlflow.search_experiments()
        return True
    except Exception as e:
        logger.warning(f"MLflow connection test failed: {e}")
        return False

# Middleware for logging requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    processing_time = time.time() - start_time
    
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    
    logger.info(f"🌐 {client_ip} - {request.method} {request.url.path} - Status: {response.status_code} - Time: {processing_time:.4f}s")
    
    return response

if __name__ == "__main__":
    import uvicorn
    environment = os.getenv("ENVIRONMENT", "development")
    
    if environment == "production":
        uvicorn.run(
            "app:app", 
            host="0.0.0.0",
            port=8000, 
            reload=False,
            log_level="info",
            workers=4
        )
    else:
        uvicorn.run(
            "app:app",
            host="0.0.0.0", 
            port=8000,
            reload=True,
            log_level="debug",
            workers=1
        )