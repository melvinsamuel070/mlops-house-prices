#  MLOps House Price Prediction System

A complete MLOps pipeline for predicting California house prices with real-time monitoring, experiment tracking, and production deployment.

##  Features

- **Real-time Prediction API** - FastAPI backend with automatic validation
- **ML Experiment Tracking** - MLflow for model versioning and comparison  
- **Production Monitoring** - Prometheus metrics & Grafana dashboards
- **Web Interface** - Clean frontend for business users
- **Containerized Deployment** - Docker compose for easy scaling

##  Tech Stack

- **Backend**: FastAPI + Python + Scikit-learn
- **Monitoring**: Prometheus + Custom metrics
- **MLOps**: MLflow for experiment tracking
- **Infrastructure**: Docker + Nginx
- **Frontend**: HTML/CSS + JavaScript

##  Quick Start

```bash
# Clone and run
git clone <your-repo>
cd mlops-house-price
docker-compose up --build

# Access services:
# Frontend: http://localhost
# API Docs: http://localhost:8000/docs  
# MLflow:   http://localhost:5000
# Prometheus: http://localhost:9090
# grafana 