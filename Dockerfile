FROM python:3.10-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install package in development mode
RUN pip install -e .

# Create directories
RUN mkdir -p data experiments/logs experiments/checkpoints experiments/results

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MUJOCO_GL=egl

# Default command
CMD ["/bin/bash"]
