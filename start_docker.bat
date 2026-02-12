@echo off
REM Fashion Search - Docker Quick Start (Windows)

echo ================================
echo Fashion Search - Docker Setup
echo ================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not installed!
    echo Please install Docker Desktop from: https://www.docker.com/products/docker-desktop/
    pause
    exit /b 1
)

echo [OK] Docker is installed
echo.

REM Build Docker image
echo [1/3] Building Docker image...
docker build -t fashion-search:latest .
if %errorlevel% neq 0 (
    echo [ERROR] Docker build failed!
    pause
    exit /b 1
)

echo [OK] Docker image built successfully
echo.

REM Start container
echo [2/3] Starting container...
docker-compose up -d
if %errorlevel% neq 0 (
    echo [ERROR] Failed to start container!
    pause
    exit /b 1
)

echo [OK] Container started
echo.

REM Wait for server to be ready
echo [3/3] Waiting for server to be ready...
timeout /t 5 /nobreak >nul

REM Check health
curl -f http://localhost:8001/health >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Server might not be ready yet
    echo Check logs with: docker-compose logs -f
) else (
    echo [OK] Server is healthy!
)

echo.
echo ================================
echo   Fashion Search is running!
echo ================================
echo.
echo Web UI:  http://localhost:8001/
echo API Docs: http://localhost:8001/docs
echo.
echo Commands:
echo   - View logs:   docker-compose logs -f
echo   - Stop server: docker-compose down
echo   - Restart:     docker-compose restart
echo.

pause
