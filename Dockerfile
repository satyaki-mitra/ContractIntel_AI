FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy application
COPY . .

# Create directories
RUN mkdir -p uploads cache logs

# Expose port
EXPOSE 7860

# Simple CMD - start Ollama in background, then start FastAPI
CMD ollama serve & sleep 20 && ollama pull llama3:8b & uvicorn app:app --host 0.0.0.0 --port 7860