# Use Python slim (Debian-based but much lighter than full Ubuntu)
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/TTS:$PYTHONPATH

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libsndfile1-dev \
    ffmpeg \
    gcc \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy project files (excluding .git)
COPY app.py requirements.txt ./
COPY assets/ ./assets/

# Clone TTS and install dependencies in single optimized layer
RUN git clone --depth 1 --branch 0.1.1 https://github.com/Min9802/TTS.git TTS && \
    # Apply numpy compatibility fix
    sed -i 's/np\.dtypes\.Float64DType/np.dtype("float64")/g' /app/TTS/TTS/__init__.py && \
    pip install --upgrade pip setuptools wheel && \
    cd TTS && \
    pip install --use-deprecated=legacy-resolver -e . --verbose && \
    cd .. && \
    pip install -r requirements.txt && \
    # Debug: Check if TTS is installed
    python -c "import sys; print('Python path:', sys.path)" && \
    python -c "import os; print('TTS dir exists:', os.path.exists('/app/TTS'))" && \
    python -c "import os; print('TTS/TTS dir exists:', os.path.exists('/app/TTS/TTS'))" && \
    ls -la /app/TTS/ && \
    pip list | grep -i tts && \
    pip cache purge && \
    rm -rf ~/.cache/pip && \
    rm -rf /tmp/* && \
    find /usr/local/lib/python${PYTHON_VERSION}/site-packages -name "*.pyc" -delete && \
    find /usr/local/lib/python${PYTHON_VERSION}/site-packages -name "__pycache__" -type d -exec rm -rf {} + || true

# Expose port for Gradio app (default port from app.py)
EXPOSE 5003

# Set entrypoint to run Python directly
CMD ["python", "app.py"]