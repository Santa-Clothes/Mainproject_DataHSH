@echo off
REM ===========================================
REM Kiro GPU 학습 환경 세팅 스크립트
REM Windows + Conda + Python 3.10 + CUDA PyTorch
REM 환경 이름: cuda_gpu
REM ===========================================

echo.
echo 1️⃣ Conda 환경 확인
conda --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Conda가 설치되어 있지 않습니다. Anaconda/Miniconda 설치 후 다시 실행하세요.
    pause
    exit /b
)

echo.
echo 2️⃣ Python 3.10 기반 CUDA 환경 생성
conda create -n cuda_gpu python=3.10 -y

echo.
echo 3️⃣ 환경 활성화
call conda activate cuda_gpu

echo.
echo 4️⃣ pip 최신화
python -m pip install --upgrade pip

echo.
echo 5️⃣ PyTorch CUDA 설치 (CUDA 12.2 기준)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu122

echo.
echo 6️⃣ 설치 확인
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('Torch Version:', torch.__version__); print('CUDA Version:', torch.version.cuda)"

echo.
echo 7️⃣ Kfashion 학습 스크립트 실행 (GPU)
REM 경로를 실제 Kfashion 학습 스크립트로 바꿔주세요
python C:\Mainproject_DataHSH\scripts\training\create_baseline_v3_fashionclip.py --device cuda

echo.
echo ✅ 모든 과정 완료!
pause
