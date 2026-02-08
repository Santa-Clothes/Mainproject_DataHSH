#!/usr/bin/env python3
"""
학습 결과 시각화 스크립트
TensorBoard 로그와 체크포인트를 분석하여 그래프 생성
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI 없이 실행
import json
from pathlib import Path
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

def load_tensorboard_logs(log_dir):
    """TensorBoard 로그 파일 읽기"""
    log_path = Path(log_dir)
    
    # 가장 최근 로그 파일 찾기
    event_files = list(log_path.glob("events.out.tfevents.*"))
    if not event_files:
        print(f"No TensorBoard logs found in {log_dir}")
        return None
    
    latest_log = max(event_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading logs from: {latest_log}")
    
    # 로그 읽기
    ea = event_accumulator.EventAccumulator(str(latest_log))
    ea.Reload()
    
    return ea

def plot_training_results():
    """학습 결과 시각화"""
    
    print("="*60)
    print("학습 결과 시각화")
    print("="*60)
    
    # 로그 디렉토리
    log_dir = Path("logs")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # TensorBoard 로그 로드
    ea = load_tensorboard_logs(log_dir)
    
    if ea is None:
        print("TensorBoard 로그를 찾을 수 없습니다.")
        return
    
    # 사용 가능한 태그 확인
    print("\n사용 가능한 메트릭:")
    for tag in ea.Tags()['scalars']:
        print(f"  - {tag}")
    
    # Figure 생성
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('K-Fashion Training Results', fontsize=16, fontweight='bold')
    
    # 1. Stage 1 Loss (Standalone)
    ax1 = axes[0, 0]
    try:
        train_loss = ea.Scalars('Standalone/Train_Loss')
        val_loss = ea.Scalars('Standalone/Val_Loss')
        
        train_steps = [s.step for s in train_loss]
        train_values = [s.value for s in train_loss]
        val_steps = [s.step for s in val_loss]
        val_values = [s.value for s in val_loss]
        
        ax1.plot(train_steps, train_values, label='Train Loss', linewidth=2)
        ax1.plot(val_steps, val_values, label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Stage 1: JSON Encoder Standalone')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    except:
        ax1.text(0.5, 0.5, 'No Stage 1 data', ha='center', va='center')
    
    # 2. Stage 2 Loss (Contrastive)
    ax2 = axes[0, 1]
    try:
        train_loss = ea.Scalars('Training/Loss')
        val_loss = ea.Scalars('Validation/Loss')
        
        train_steps = [s.step for s in train_loss]
        train_values = [s.value for s in train_loss]
        val_steps = [s.step for s in val_loss]
        val_values = [s.value for s in val_loss]
        
        ax2.plot(train_steps, train_values, label='Train Loss', linewidth=2)
        ax2.plot(val_steps, val_values, label='Val Loss', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Stage 2: Contrastive Learning')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    except:
        ax2.text(0.5, 0.5, 'No Stage 2 data', ha='center', va='center')
    
    # 3. Validation Metrics
    ax3 = axes[1, 0]
    try:
        top1 = ea.Scalars('Validation/top1_accuracy')
        top5 = ea.Scalars('Validation/top5_accuracy')
        mrr = ea.Scalars('Validation/mean_reciprocal_rank')
        
        steps = [s.step for s in top1]
        top1_values = [s.value for s in top1]
        top5_values = [s.value for s in top5]
        mrr_values = [s.value for s in mrr]
        
        ax3.plot(steps, top1_values, label='Top-1 Accuracy', linewidth=2, marker='o')
        ax3.plot(steps, top5_values, label='Top-5 Accuracy', linewidth=2, marker='s')
        ax3.plot(steps, mrr_values, label='MRR', linewidth=2, marker='^')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Score')
        ax3.set_title('Validation Metrics')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    except:
        ax3.text(0.5, 0.5, 'No validation metrics', ha='center', va='center')
    
    # 4. Learning Rate
    ax4 = axes[1, 1]
    try:
        lr = ea.Scalars('Training/Learning_Rate')
        
        steps = [s.step for s in lr]
        values = [s.value for s in lr]
        
        ax4.plot(steps, values, linewidth=2, color='green')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
    except:
        ax4.text(0.5, 0.5, 'No LR data', ha='center', va='center')
    
    plt.tight_layout()
    
    # 저장
    output_path = results_dir / "training_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 시각화 저장: {output_path}")
    
    plt.close()
    
    # 요약 통계 생성
    generate_summary_stats(ea, results_dir)

def generate_summary_stats(ea, results_dir):
    """학습 요약 통계 생성"""
    
    summary = {
        "stage1": {},
        "stage2": {},
        "final_metrics": {}
    }
    
    try:
        # Stage 1 통계
        val_loss = ea.Scalars('Standalone/Val_Loss')
        if val_loss:
            summary["stage1"]["final_val_loss"] = val_loss[-1].value
            summary["stage1"]["best_val_loss"] = min(s.value for s in val_loss)
            summary["stage1"]["epochs"] = len(val_loss)
    except:
        pass
    
    try:
        # Stage 2 통계
        val_loss = ea.Scalars('Validation/Loss')
        if val_loss:
            summary["stage2"]["final_val_loss"] = val_loss[-1].value
            summary["stage2"]["best_val_loss"] = min(s.value for s in val_loss)
            summary["stage2"]["epochs"] = len(val_loss)
        
        # 최종 메트릭
        top1 = ea.Scalars('Validation/top1_accuracy')
        top5 = ea.Scalars('Validation/top5_accuracy')
        mrr = ea.Scalars('Validation/mean_reciprocal_rank')
        
        if top1:
            summary["final_metrics"]["top1_accuracy"] = top1[-1].value
        if top5:
            summary["final_metrics"]["top5_accuracy"] = top5[-1].value
        if mrr:
            summary["final_metrics"]["mrr"] = mrr[-1].value
    except:
        pass
    
    # JSON 저장
    summary_path = results_dir / "training_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 요약 통계 저장: {summary_path}")
    
    # 콘솔 출력
    print("\n" + "="*60)
    print("학습 요약")
    print("="*60)
    
    if summary["stage1"]:
        print("\n[Stage 1: JSON Encoder Standalone]")
        for key, value in summary["stage1"].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    if summary["stage2"]:
        print("\n[Stage 2: Contrastive Learning]")
        for key, value in summary["stage2"].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    if summary["final_metrics"]:
        print("\n[Final Metrics]")
        for key, value in summary["final_metrics"].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

if __name__ == "__main__":
    plot_training_results()
