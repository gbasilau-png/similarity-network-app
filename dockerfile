FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for pandas and kaleido
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8050

# Gunicorn configuration optimized for Dash
CMD ["gunicorn", "--bind", "0.0.0.0:8050", "--workers", "2", "--threads", "2", "--timeout", "60", "app:server"]
