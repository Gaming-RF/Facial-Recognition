# Multi-stage build for Facial Recognition System
# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code (models excluded — downloaded at runtime)
COPY config.py main.py ./
COPY app/ app/
COPY templates/ templates/
COPY static/ static/

# Create data directory for persistence (mounted as volume)
RUN mkdir -p /app/data /app/data/mugshots

EXPOSE 5000

CMD ["python", "main.py"]
