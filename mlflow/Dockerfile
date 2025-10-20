# FROM python:3.10-slim

# WORKDIR /app

# RUN pip install --no-cache-dir mlflow==2.14.1

# EXPOSE 5000

# ENTRYPOINT ["mlflow", "server"]

# Dockerfile.mlflow
FROM python:3.10-slim

# Set working directory
WORKDIR /mlflow

# Install dependencies (MLflow and Gunicorn)
RUN pip install --no-cache-dir mlflow==2.14.1 gunicorn

# Expose MLflow port
EXPOSE 5000

# Set environment variables
ENV MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlruns
ENV MLFLOW_HOST=0.0.0.0

# Define volumes for persistent storage
VOLUME ["/mlruns", "/mlflow.db"]

# Start MLflow server
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "/mlruns", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
