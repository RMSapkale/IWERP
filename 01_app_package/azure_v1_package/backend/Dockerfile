FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ROOT=/app/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements-v1.txt /tmp/requirements-v1.txt
RUN pip install --no-cache-dir --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements-v1.txt

COPY app /app/app
COPY backend/startup.sh /app/backend/startup.sh
RUN chmod +x /app/backend/startup.sh

ENV PYTHONPATH="/app/app/backend:/app/app" \
    IWERP_BASE_DIR="/app/app"

EXPOSE 8000
CMD ["/app/backend/startup.sh"]
