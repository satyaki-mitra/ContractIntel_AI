FROM python:3.10-slim-bullseye

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (lightweight) - ADD pthread development headers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libjpeg62-turbo \
    poppler-utils \
    libpthread-stubs0-dev \
    && rm -rf /var/lib/apt/lists/*

    # Optimize llama-cpp-python build for CPU only to reduce build time/memory usage
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=off"
ENV FORCE_CMAKE=on

COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

COPY . .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]