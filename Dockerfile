FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for llama-cpp-python and PDF processing
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    cmake \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install with optimizations
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application
COPY . .

# Create directories (HF Spaces uses /data for persistent storage)
RUN mkdir -p uploads cache logs /data/models

# Expose port (HF Spaces uses 7860 by default)
EXPOSE 7860

# Environment variables for CPU-only operation
ENV LLAMA_CPP_N_GPU_LAYERS=0
ENV CUDA_VISIBLE_DEVICES=""  
ENV OMP_NUM_THREADS=4        
ENV NUMEXPR_MAX_THREADS=4

# HEALTH CHECK for HF Spaces
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/api/v1/health || exit 1

# CMD for HuggingFace Spaces (NO Ollama!)
CMD uvicorn app:app --host 0.0.0.0 --port 7860 --workers 1 --timeout-keep-alive 30