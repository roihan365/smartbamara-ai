# FROM python:3.10-slim
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Makassar

# Install dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip ffmpeg gcc git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Set environment variable for uvicorn
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose FastAPI default port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
