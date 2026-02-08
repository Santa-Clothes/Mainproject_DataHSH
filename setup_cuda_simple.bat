@echo off
REM ===========================================
REM CUDA PyTorch 설치 및 학습 스크립트
REM Python 3.10 이하 버전 필요
REM ===========================================

echo.
echo 현재 Python 버전 확인
python --version

echo.
echo ⚠️ 경고: Python 3.13은 PyTorch CUDA를 지원하지 않습니다.
echo Python 3.10 또는 3.11 버전을 사용해주세요.
echo.

echo 1️⃣ pip 업그레이드
python -m pip install --upgrade pip

echo.
echo 2️⃣ PyTorch CUDA 12.2 설치 시도
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu122

echo.
echo 3️⃣ CUDA 확인
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('Torch Version:', torch.__version__)"

echo.
echo 4️⃣ 필요한 패키지 설치
pip install transformers fashion-clip Pillow numpy pandas matplotlib tqdm tensorboard

echo.
echo ✅ 설치 완료!
pause
