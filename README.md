# Semantic Decision Tree (SDT) for Molecular Classification

## Overview
SMILES 기반 분자 구조를 화학 온톨로지로 변환하고, Semantic Decision Tree를 사용하여 Blood-Brain Barrier Penetration (BBBP) 예측을 수행합니다.

이 레포에는 아래 2가지 계열이 함께 들어 있습니다.

- **Legacy SDT (feature 기반 단일 트리)**: RDKit 피처를 바로 사용
- **Logic SDT (DTO/OWL 기반 단일 트리)** + **Semantic Random Forest (Logic SDT 배깅 앙상블)**

따라서 사용자가 목적에 맞게 **단일 트리(SDT)** 또는 **배깅된 앙상블(Semantic Random Forest)** 중 선택해서 실행할 수 있습니다.

## Project Structure
```
SDT/
├── data/
│   └── bbbp/
│       └── BBBP.csv
├── src/
│   ├── ontology/
│   │   ├── __init__.py
│   │   ├── molecule_ontology.py    # 온톨로지 구축
│   │   └── smiles_converter.py     # SMILES → 온톨로지 변환
│   ├── sdt/
│   │   ├── __init__.py
│   │   ├── refinement.py           # Refinement 연산자
│   │   ├── tree.py                 # SDT 구조
│   │   └── learner.py              # SDT 학습 알고리즘
│   └── utils/
│       ├── __init__.py
│       └── evaluation.py           # 평가 지표 (AUC-ROC)
├── experiments/
│   └── bbbp_experiment.py          # 메인 실행 스크립트
├── output/
│   └── (결과 저장 디렉토리)
├── requirements.txt
└── README.md
```

## Installation
```bash
pip install -r requirements.txt
```

## How to run (사용자가 선택)

### A) Legacy SDT (feature 기반 단일 트리)

여러 데이터셋을 feature 기반 SDT로 벤치마크하려면:

```bash
python experiments/benchmark_runner.py --dataset all
```

### B) Logic SDT (DTO/OWL 기반 단일 트리)

BBBP 단일 실행(기본은 **dynamic**: 매번 refinement 생성):

```bash
python verify_logic_sdt_bbbp.py
```

#### B-1) Dynamic vs Static refinement 선택

- **dynamic**: 실행할 때마다 refinements를 생성
- **static**: 미리 추출해둔 refinements(JSON)를 로드해서 재사용

Static 모드를 쓰려면 먼저 refinements를 추출해서 JSON을 만들어야 합니다.

1) (한 번만) refinement 추출 + JSON 생성

```bash
python experiments/extract_dto_refinements.py --dataset bbbp --target p_np --limit 200
```

생성 파일:

- `output/dto_refinements/bbbp/p_np.txt`
- `output/dto_refinements/bbbp/p_np.json`

2) Static 모드로 Logic SDT 실행

```bash
python verify_logic_sdt_bbbp.py --refinement-mode static --refinement-file output/dto_refinements/bbbp/p_np.json
```

### C) Semantic Random Forest (Logic SDT 배깅 앙상블)

배깅(bootstrap aggregating)으로 Logic SDT 여러 개를 학습해 성능을 올리고 싶으면:

```bash
python experiments/verify_semantic_forest.py
```

내부 구현은 `src/sdt/logic_forest.py`의 `SemanticRandomForest`이며, 각 트리는 `LogicSDTLearner`로 학습됩니다.

## Evaluation Metric
- AUC-ROC (Area Under the ROC Curve)
