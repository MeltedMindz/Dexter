# Dockerfile enhancements
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node exporter for system metrics
RUN curl -LO https://github.com/prometheus/node_exporter/releases/download/v1.3.1/node_exporter-1.3.1.linux-amd64.tar.gz \
    && tar xvfz node_exporter-1.3.1.linux-amd64.tar.gz \
    && mv node_exporter-1.3.1.linux-amd64/node_exporter /usr/local/bin/ \
    && rm -rf node_exporter-1.3.1.linux-amd64*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Start both API and Node exporter
CMD ["sh", "-c", "node_exporter & uvicorn app.main:app --host 0.0.0.0 --port 8000"]
