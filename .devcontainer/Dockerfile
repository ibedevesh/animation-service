FROM python:3.9

RUN apt-get update && apt-get install -y \
    ffmpeg \
    cairo \
    libcairo2-dev \
    pkg-config \
    python3-dev \
    build-essential \
    libpango1.0-dev \
    libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
