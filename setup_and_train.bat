@echo off
REM Anaconda Prompt에서 실행하세요!

echo ========================================
echo CUDA 환경 설정 및 학습 시작
echo ========================================

echo.
echo 1. Python 3.10 환경 생성
call conda create -n cuda_gpu python=3.10 -y

echo.
echo 2. 환경 활성화
call conda activate cuda_gpu

echo.
echo 3. pip 업그레이드
python -m pip install --upgrade pip

echo.
echo 4. PyTorch CUDA 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu122

echo.
echo 5. 필요한 패키지 설치
pip install transformers fashion-clip Pillow numpy pandas matplotlib tqdm tensorboard psutil accelerate optuna pytest hypothesis

echo.
echo 6. CUDA 확인
python -c "import torch; print('='*50); print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('PyTorch Version:', torch.__version__); print('CUDA Version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('='*50)"

echo.
echo 7. 데이터셋 경로 확인
set /p DATASET_PATH="데이터셋 경로를 입력하세요 (예: C:/sample/라벨링데이터): "

echo.
echo 8. 학습 시작
python train.py --dataset_path "%DATASET_PATH%" --epochs 20 --batch_size 16

echo.
echo ========================================
echo 완료!
echo ========================================
pause
