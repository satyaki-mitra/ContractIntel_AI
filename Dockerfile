# Base Image (HF-recommended for Python apps)-
FROM python:3.10-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
    
# Reduce pip cache to save space on HF Spaces
ENV PIP_NO_CACHE_DIR=1

# System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libpoppler-cpp-dev \
    libglib2.0-0 \
    libjpeg-dev \
    libpng-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy App Code
WORKDIR /app
COPY . /app
    
# Python Dependencies
COPY requirements.txt /app/requirements.txt
    
# Upgrade pip
RUN pip install --upgrade pip
    
# Install requirements
RUN pip install -r /app/requirements.txt
    
# Download spaCy model (CPU-friendly)
RUN python -m spacy download en_core_web_sm

# Expose & Run
EXPOSE 7860
    
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]    