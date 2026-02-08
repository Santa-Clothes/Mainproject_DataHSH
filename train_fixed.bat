@echo off
chcp 65001
call conda activate cuda_gpu
echo Starting training with fixed CUDA device handling...
python train.py --dataset_path "D:/K-Fashion/Training/라벨링데이터" --epochs 20 --batch_size 16 --standalone_epochs 5
pause
