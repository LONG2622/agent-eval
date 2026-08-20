# ============================================================
# Agent Eval - Docker Deployment
# Multi-stage build for optimized image size
# ============================================================

# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Install the package itself (in editable mode for development, or regular for production)
COPY pyproject.toml .
COPY src/ ./src/
RUN pip install --no-cache-dir --user -e .


# Stage 2: Runtime
FROM python:3.11-slim AS runtime

WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 appuser

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY examples/ ./examples/
COPY pyproject.toml .
COPY README.md .

# Ensure CLI entry point is executable
RUN ls -la /home/appuser/.local/bin/agent 2>/dev/null && chmod +x /home/appuser/.local/bin/agent || true

# Create data directories
RUN mkdir -p /app/outputs /app/data && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Add user bin to PATH
ENV PATH=/home/appuser/.local/bin:$PATH

# Environment variables (override in docker-compose.yml)
ENV PYTHONUNBUFFERED=1
ENV AGENT_OUTPUT_DIR=/app/outputs
ENV AGENT_DATA_DIR=/app/data

# Expose the web server port
EXPOSE 8000

# Health check - uses the /api/health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# Default command: start the web server
CMD ["uvicorn", "agent_eval.server.app:app", "--host", "0.0.0.0", "--port", "8000"]