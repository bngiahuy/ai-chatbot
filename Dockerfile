# Use official Python 3.12 slim image
FROM python:3.13.3-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

RUN mkdir -p data chroma_storage models
# Set volume mounts for persistence
VOLUME /app/data
VOLUME /app/chroma_storage
VOLUME /app/models
EXPOSE 5000
# Run the pipeline
CMD ["python", "main.py"]
