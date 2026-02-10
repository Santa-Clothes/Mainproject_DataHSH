#!/usr/bin/env python3
"""
패션 JSON 인코더 - 학습 스크립트

합리적인 기본값으로 패션 JSON 인코더 시스템을 학습하기 위한
쉬운 인터페이스를 제공하는 간소화된 학습 스크립트입니다.

사용법:
    python scripts/train.py --dataset_path /path/to/kfashion
    python scripts/train.py --dataset_path /path/to/kfashion --epochs 50 --batch_size 64
    python scripts/train.py --dataset_path /path/to/kfashion --config my_config.json
"""

import argparse
import sys
import json
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 메인 시스템 가져오기
from scripts.main import FashionEncoderSystem, create_config_file


def main():
    """간소화된 학습 인터페이스입니다."""
    parser = argparse.ArgumentParser(
        description="패션 JSON 인코더 - 간소화된 학습 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python scripts/train.py --dataset_path /path/to/kfashion
  python scripts/train.py --dataset_path /path/to/kfashion --epochs 50
  python scripts/train.py --dataset_path /path/to/kfashion --batch_size 32 --lr 0.001
  python scripts/train.py --dataset_path /path/to/kfashion --config config.json
  python scripts/train.py --sanity_check  # 합성 데이터로 정상성 검사 실행
        """
    )
    
    # 주요 인수
    parser.add_argument('--dataset_path', 
                       help='K-Fashion 데이터셋 경로 (--sanity_check가 아닌 경우 필수)')
    parser.add_argument('--config', 
                       help='설정 파일 경로')
    
    # 학습 매개변수
    parser.add_argument('--epochs', type=int, default=20,
                       help='학습 에포크 수 (기본값: 20)')
    parser.add_argument('--standalone_epochs', type=int, default=5,
                       help='독립 학습 에포크 수 (기본값: 5)')
    parser.add_argument('--batch_size', type=int,
                       help='배치 크기 (설정 재정의)')
    parser.add_argument('--lr', '--learning_rate', type=float, dest='learning_rate',
                       help='학습률 (설정 재정의)')
    
    # 출력 디렉토리
    parser.add_argument('--output_dir', default='training_output',
                       help='기본 출력 디렉토리 (기본값: training_output)')
    
    # 특수 모드
    parser.add_argument('--sanity_check', action='store_true',
                       help='전체 학습 대신 정상성 검사 실행')
    parser.add_argument('--create_config', 
                       help='샘플 설정 파일 생성 후 종료')
    
    # 고급 옵션
    parser.add_argument('--no_standalone', action='store_true',
                       help='독립 학습 단계 건너뛰기')
    parser.add_argument('--gpu', type=int,
                       help='사용할 GPU 장치 ID')
    
    args = parser.parse_args()
    
    # 특수 명령어 처리
    if args.create_config:
        create_config_file(args.create_config)
        return
    
    if args.sanity_check:
        print("합성 데이터로 정상성 검사를 실행합니다...")
        run_sanity_check(args)
        return
    
    # 필수 인수 검증
    if not args.dataset_path:
        print("오류: --dataset_path가 필요합니다 (--sanity_check 사용하지 않는 경우)")
        parser.print_help()
        sys.exit(1)
    
    if not Path(args.dataset_path).exists():
        print(f"오류: 데이터셋 경로가 존재하지 않습니다: {args.dataset_path}")
        sys.exit(1)
    
    # 학습 실행
    run_training(args)


def run_training(args):
    """메인 학습 파이프라인을 실행합니다."""
    print("패션 JSON 인코더 - 학습")
    print("=" * 50)
    
    # 출력 디렉토리 설정
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    
    output_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    
    print(f"데이터셋: {args.dataset_path}")
    print(f"출력 디렉토리: {output_dir}")
    print(f"총 에포크: {args.epochs}")
    print(f"독립 학습 에포크: {args.standalone_epochs}")
    
    try:
        # 시스템 초기화
        system = FashionEncoderSystem(config_path=args.config)
        
        # 명령줄 인수로 설정 재정의
        if args.batch_size:
            system.config.batch_size = args.batch_size
            print(f"배치 크기: {args.batch_size}")
        
        if args.learning_rate:
            system.config.learning_rate = args.learning_rate
            print(f"학습률: {args.learning_rate}")
        
        # 데이터 및 트레이너 설정
        print("\n데이터를 설정합니다...")
        system.setup_data(args.dataset_path)
        
        print("트레이너를 설정합니다...")
        system.setup_trainer(
            checkpoint_dir=str(checkpoint_dir),
            log_dir=str(log_dir)
        )
        
        # 학습 실행
        print("\n학습을 시작합니다...")
        
        standalone_epochs = 0 if args.no_standalone else args.standalone_epochs
        contrastive_epochs = args.epochs - standalone_epochs
        
        results = system.train(
            standalone_epochs=standalone_epochs,
            contrastive_epochs=contrastive_epochs,
            save_results=True
        )
        
        # 결과 요약 출력
        print_training_summary(results, output_dir)
        
        # 정리
        system.cleanup()
        
        print(f"\n학습이 성공적으로 완료되었습니다!")
        print(f"체크포인트 저장 위치: {checkpoint_dir}")
        print(f"로그 저장 위치: {log_dir}")
        print(f"학습 진행 상황 보기: tensorboard --logdir {log_dir}")
        
    except KeyboardInterrupt:
        print("\n사용자에 의해 학습이 중단되었습니다")
        sys.exit(1)
    except Exception as e:
        print(f"\n학습 실패: {e}")
        sys.exit(1)


def run_sanity_check(args):
    """정상성 검사 모드를 실행합니다."""
    print("패션 JSON 인코더 - 정상성 검사")
    print("=" * 50)
    
    try:
        # 시스템 초기화
        system = FashionEncoderSystem(config_path=args.config)
        
        # 정상성 검사 실행
        results = system.sanity_check(
            dataset_path=args.dataset_path,  # 합성 데이터의 경우 None 가능
            num_epochs=3
        )
        
        # 요약 출력
        print_sanity_check_summary(results)
        
        # 정리
        system.cleanup()
        
    except Exception as e:
        print(f"정상성 검사 실패: {e}")
        sys.exit(1)


def print_training_summary(results, output_dir):
    """학습 결과 요약을 출력합니다."""
    print("\n" + "=" * 60)
    print("학습 요약")
    print("=" * 60)
    
    if 'standalone' in results:
        standalone = results['standalone']
        print(f"독립 학습:")
        print(f"  에포크: {len(standalone['train_losses'])}")
        print(f"  초기 손실: {standalone['train_losses'][0]:.4f}")
        print(f"  최종 손실: {standalone['train_losses'][-1]:.4f}")
        
        final_analysis = standalone.get('final_analysis', {})
        if final_analysis:
            print(f"  출력 정규화됨: {final_analysis.get('is_normalized', '알 수 없음')}")
            print(f"  평균 노름: {final_analysis.get('norm_mean', 0):.4f}")
    
    if 'contrastive' in results:
        contrastive = results['contrastive']
        print(f"\n대조 학습:")
        print(f"  에포크: {contrastive['total_epochs']}")
        print(f"  최고 검증 손실: {contrastive['best_val_loss']:.4f}")
        
        final_metrics = contrastive.get('final_metrics', {})
        if final_metrics:
            print(f"  Top-1 정확도: {final_metrics.get('top1_accuracy', 0):.4f}")
            print(f"  Top-5 정확도: {final_metrics.get('top5_accuracy', 0):.4f}")
            print(f"  평균 역순위: {final_metrics.get('mean_reciprocal_rank', 0):.4f}")
    
    print(f"\n결과 저장 위치: {output_dir}")


def print_sanity_check_summary(results):
    """정상성 검사 결과 요약을 출력합니다."""
    print("\n" + "=" * 60)
    print("정상성 검사 요약")
    print("=" * 60)
    
    validation = results.get('validation_results', {})
    
    print("검증 확인:")
    checks = [
        ('dimension_check', '출력 차원'),
        ('normalization_check', 'L2 정규화'),
        ('gradient_check', '그래디언트 계산'),
        ('field_processing_check', '필드 처리'),
        ('batch_consistency_check', '배치 일관성')
    ]
    
    all_passed = True
    for check_key, check_name in checks:
        passed = validation.get(check_key, False)
        status = "✓ 통과" if passed else "✗ 실패"
        print(f"  {status} {check_name}")
        if not passed:
            all_passed = False
    
    if validation.get('errors'):
        print("\n오류:")
        for error in validation['errors']:
            print(f"  - {error}")
    
    # 학습 진행 상황
    training = results.get('training_results', {})
    if training:
        print(f"\n학습 진행 상황:")
        print(f"  손실 감소: {training['train_losses'][0]:.4f} → {training['train_losses'][-1]:.4f}")
    
    # 최종 평가
    final_analysis = results.get('final_analysis', {})
    is_normalized = final_analysis.get('is_normalized', False)
    correct_dim = final_analysis.get('embedding_dim', 0) == 512
    
    print(f"\n{'='*40}")
    if all_passed and is_normalized and correct_dim:
        print("🎉 정상성 검사 통과")
        print("JSON 인코더가 올바르게 작동합니다!")
    else:
        print("⚠️  정상성 검사에서 문제가 감지되었습니다")
        print("자세한 내용은 위의 결과를 검토하세요.")
    print("="*40)


if __name__ == "__main__":
    main()
