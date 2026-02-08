@echo off
echo CUDA 환경 설정 시작...
echo.

echo 1. Python 3.10 환경 생성 (conda-forge 사용)
conda create -n cuda_gpu python=3.10 -y -c conda-forge --override-channels

echo.
echo 2. 환경 활성화
call conda activate cuda_gpu

echo.
echo 3. PyTorch CUDA 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu122

echo.
echo 4. 필수 패키지 설치
pip install transformers fashion-clip Pillow numpy pandas matplotlib tqdm tensorboard psutil

echo.
echo 5. CUDA 확인
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo.
echo 완료! 이제 학습을 시작할 수 있습니다.
pause
