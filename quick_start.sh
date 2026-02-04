#!/bin/bash
# Quick start script for sentiment analysis project

echo "=========================================="
echo "Sentiment Analysis Project - Quick Start"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

echo "✓ Docker is installed"
echo ""

# Step 1: Build Docker images
echo "Step 1: Building Docker images..."
echo "--------------------------------"

echo "Building training image..."
docker build -f src/train/Dockerfile -t sentiment-training . || {
    echo "❌ Failed to build training image"
    exit 1
}
echo "✓ Training image built successfully"

echo "Building inference image..."
docker build -f src/inference/Dockerfile -t sentiment-inference . || {
    echo "❌ Failed to build inference image"
    exit 1
}
echo "✓ Inference image built successfully"
echo ""

# Step 2: Run training
echo "Step 2: Running training pipeline..."
echo "-----------------------------------"
docker run -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs sentiment-training || {
    echo "❌ Training failed"
    exit 1
}
echo "✓ Training completed successfully"
echo ""

# Step 3: Run inference
echo "Step 3: Running inference pipeline..."
echo "------------------------------------"
docker run -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs sentiment-inference || {
    echo "❌ Inference failed"
    exit 1
}
echo "✓ Inference completed successfully"
echo ""

# Summary
echo "=========================================="
echo "All Done! 🎉"
echo "=========================================="
echo ""
echo "Results are available in:"
echo "  - outputs/models/          (trained models)"
echo "  - outputs/predictions/     (predictions and metrics)"
echo "  - outputs/figures/         (visualizations)"
echo ""
echo "To view metrics:"
echo "  cat outputs/predictions/test_metrics.txt"
echo ""
echo "To view predictions:"
echo "  cat outputs/predictions/predictions.csv"
echo ""
