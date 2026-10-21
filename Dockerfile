# syntax=docker/dockerfile:1
# Two-stage: wheels resolved first, runtime installs only wheels (no compiler).
FROM python:3.11-slim AS wheels
ENV PIP_NO_CACHE_DIR=1
COPY requirements.txt .
WORKDIR /build
RUN pip wheel --wheel-dir /wheels -r requirements.txt

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY --from=wheels /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

# Copy only what the app needs - never `COPY . .` (uploads/db/reports stay out).
COPY app ./app

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p uploads reports models deployments \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4); sys.exit(0)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
