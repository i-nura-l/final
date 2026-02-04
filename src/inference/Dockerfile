# Inference Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Copy source code
COPY src/ ./src/

# Create necessary directories
RUN mkdir -p data/raw outputs/predictions

# Set environment variable
ENV PYTHONUNBUFFERED=1

# Run inference script
CMD ["python", "src/inference/run_inference.py", "--model-dir", "/app/outputs/models", "--output-dir", "/app/outputs"]
