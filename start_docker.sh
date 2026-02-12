#!/bin/bash
# Fashion Search - Docker Quick Start (Linux/Mac)

echo "================================"
echo "Fashion Search - Docker Setup"
echo "================================"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker is not installed!"
    echo "Please install Docker from: https://docs.docker.com/get-docker/"
    exit 1
fi

echo "[OK] Docker is installed"
echo ""

# Build Docker image
echo "[1/3] Building Docker image..."
docker build -t fashion-search:latest .
if [ $? -ne 0 ]; then
    echo "[ERROR] Docker build failed!"
    exit 1
fi

echo "[OK] Docker image built successfully"
echo ""

# Start container
echo "[2/3] Starting container..."
docker-compose up -d
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to start container!"
    exit 1
fi

echo "[OK] Container started"
echo ""

# Wait for server to be ready
echo "[3/3] Waiting for server to be ready..."
sleep 5

# Check health
if curl -f http://localhost:8001/health > /dev/null 2>&1; then
    echo "[OK] Server is healthy!"
else
    echo "[WARNING] Server might not be ready yet"
    echo "Check logs with: docker-compose logs -f"
fi

echo ""
echo "================================"
echo "  Fashion Search is running!"
echo "================================"
echo ""
echo "Web UI:   http://localhost:8001/"
echo "API Docs: http://localhost:8001/docs"
echo ""
echo "Commands:"
echo "  - View logs:   docker-compose logs -f"
echo "  - Stop server: docker-compose down"
echo "  - Restart:     docker-compose restart"
echo ""
