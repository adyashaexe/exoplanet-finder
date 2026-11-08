FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# CPU-only torch keeps the image small
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[data]"

# Mount or copy trained artifacts into /app/artifacts (model.pt, baseline.joblib)
COPY artifacts ./artifacts
ENV EXOFINDER_HOME=/app

EXPOSE 8000
CMD ["exofinder", "serve", "--host", "0.0.0.0", "--port", "8000"]
