FROM python:3.10-slim-bullseye

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV DOCKER_CONTAINER=true  
ENV SPACE_APP_DATA=/data
ENV HF_HOME=/data/huggingface
ENV LLAMA_CPP_MODEL_PATH=/data/models/Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf

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

# Download GGUF model during build
RUN python -c "from huggingface_hub import hf_hub_download; \
    import shutil; \
    downloaded = hf_hub_download( \
        repo_id='NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF', \
        filename='Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf', \
        cache_dir='/data/huggingface' \
    ); \
    shutil.copy(downloaded, '/data/models/Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf')" && \
    echo "Model downloaded to /data/models/"

# Copy app code
COPY . .

# Set proper permissions
RUN chmod -R 755 /app && \
    chmod -R 755 /data

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/api/v1/health || exit 1

EXPOSE 7860

# Use multiple workers for better performance
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "2"]