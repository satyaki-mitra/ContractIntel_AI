FROM python:3.10-slim-bullseye

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV DOCKER_CONTAINER=true  
ENV SPACE_APP_DATA=/data
ENV HF_HOME=/data/huggingface

# Optimize llama-cpp-python build for CPU only
ENV CMAKE_ARGS="-DLLAMA_BLAS=0 -DLLAMA_CUBLAS=0"
ENV FORCE_CMAKE=1

WORKDIR /app

# System deps - minimal for HuggingFace Spaces
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libjpeg62-turbo \
    poppler-utils \
    libmagic1 \
    curl \
    git \             
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better layer caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies with specific versions
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir

# Download spaCy model (after dependencies)
RUN python -m spacy download en_core_web_sm

# Create directories that your app expects
RUN mkdir -p /data/models /data/uploads /data/cache /data/logs /data/huggingface

# Copy app code
COPY . .

# Set proper permissions
RUN chmod -R 755 /app && \
    chmod -R 755 /data

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/docs || exit 1  # Changed to /docs endpoint

EXPOSE 7860

# Use multiple workers for better performance
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "2"]