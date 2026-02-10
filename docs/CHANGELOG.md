# Changelog
## Fashion JSON Encoder 변경 이력

---

## [v5.0.0] - 2026-02-06 (프로덕션 배포)

### 🎉 주요 성과
- **Top-1 Accuracy**: 47.1% (v3 대비 +16.8%p)
- **Top-5 Accuracy**: 87.1% (v3 대비 +15.6%p)
- **MRR**: 0.638 (v3 대비 +31.8%)
- **재현성 검증**: 4개 seed 테스트 완료 (CV < 5%)

### ✨ 새로운 기능
- 강화된 데이터 증강 (6가지 augmentation)
- 클래스 밸런싱 샘플러 (레트로 불균형 해결)
- Seed 검증 시스템 (재현성 확보)

### 🔧 개선 사항
- RandomResizedCrop 추가 (scale 0.8-1.0)
- ColorJitter 강화 (0.1 → 0.2)
- RandomRotation 증가 (5° → 10°)
- RandomAffine 추가 (translate 5%, scale 5%)
- GaussianBlur 추가 (30% probability)
- RandomErasing 추가 (20% probability)

### 📚 문서
- README.md 전면 개편 (v5 기준)
- PROJECT_SUMMARY.md 추가 (프로젝트 요약)
- v5_seed_validation_report.md 추가 (재현성 검증)
- baseline_v5_final_report.md 추가 (상세 분석)

### 🗑️ 제거
- 오래된 v1, v2 결과 파일 삭제
- CODE_REVIEW_가이드.md 삭제
- CLEANUP_SUMMARY.md 삭제
- 중복 및 불필요한 분석 파일 정리

---

## [v4.0.0] - 2026-02-06 (실험 실패)

### ❌ 실험 결과
- **Top-1 Accuracy**: 29.4% (v3보다 낮음)
- **Top-5 Accuracy**: 72.0% (v3: 71.5%)
- **결론**: Fine-tuning 효과 없음

### 🔬 시도한 기능
- FashionCLIP 마지막 2개 레이어 fine-tuning
- Differential learning rate (JSON: 1e-4, CLIP: 1e-5)

### 📝 교훈
- FashionCLIP은 이미 충분히 최적화됨
- Frozen 상태가 fine-tuned보다 우수
- 2,172개 샘플로는 87M 파라미터 fine-tuning 부족

### 📚 문서
- baseline_v3_vs_v4_analysis.md 추가 (실패 원인 분석)

---

## [v3.0.0] - 2026-02-05

### 🎉 주요 성과
- **Top-1 Accuracy**: 30.3% (v2 대비 +8.1%p)
- **Top-5 Accuracy**: 71.5% (v2 대비 +7.4%p)
- **MRR**: 0.484 (v2 대비 +18.9%)

### ✨ 새로운 기능
- FashionCLIP 통합 (`patrickjohncyh/fashion-clip`)
- 패션 도메인 특화 이미지 인코더

### 🔧 개선 사항
- Standard CLIP → FashionCLIP 교체
- 패션 특화 특징 추출 능력 향상

### 📚 문서
- baseline_v3_final_report.md 추가
- baseline_v2_vs_v3_comparison.json 추가

---

## [v2.0.0] - 2026-02-04

### 🎉 주요 성과
- **Top-1 Accuracy**: 22.2%
- **Top-5 Accuracy**: 64.1%
- **MRR**: 0.407

### ✨ 새로운 기능
- Standard CLIP 통합 (`openai/clip-vit-base-patch32`)
- Contrastive Learning 시스템 구현
- InfoNCE Loss 적용

### 🔧 개선 사항
- JSON Encoder 아키텍처 최적화
- Temperature 0.1로 설정

---

## [v1.0.0] - 2026-02-03

### 🎉 초기 릴리스
- **Top-1 Accuracy**: 1.0%
- **Top-5 Accuracy**: N/A

### ✨ 새로운 기능
- JSON Encoder 기본 구현
- K-Fashion 데이터 로더
- 기본 학습 파이프라인

### 📚 문서
- 초기 README.md
- 프로젝트 구조 설정

---

## 버전 비교 요약

| 버전 | Top-1 | Top-5 | MRR | 주요 변경 | 상태 |
|------|-------|-------|-----|-----------|------|
| v1 | 1.0% | - | - | 초기 구현 | deprecated |
| v2 | 22.2% | 64.1% | 0.407 | Standard CLIP | deprecated |
| v3 | 30.3% | 71.5% | 0.484 | FashionCLIP | baseline |
| v4 | 29.4% | 72.0% | 0.481 | Fine-tuning (실패) | deprecated |
| **v5** | **47.1%** | **87.1%** | **0.638** | **Augmentation + Balancing** | **production ✅** |

---

## 향후 계획

### v6 (계획 중)
- 배치 크기 증가 (16 → 32-64)
- 예상 효과: Top-5 +1-2%

### v7 (계획 중)
- Temperature 튜닝 (0.07, 0.15)
- 예상 효과: Top-1 +1-2%

### v8 (계획 중)
- Ensemble 시스템 (v3 + v5)
- 예상 효과: Top-1 +2-3%

---

**Last Updated**: 2026-02-06  
**Current Version**: v5.0.0 (Production Ready)
