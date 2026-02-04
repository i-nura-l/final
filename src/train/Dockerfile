# Training Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Create data and outputs directories
RUN mkdir -p data/raw data/processed outputs/models outputs/predictions outputs/figures

# Set environment variable
ENV PYTHONUNBUFFERED=1

# Run training script
CMD ["python", "src/train/train.py", "--output-dir", "/app/outputs"]
