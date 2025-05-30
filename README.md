
# DirectNet-HSI

> 🌈 Hyperspectral Image Self-Supervised Reconstruction Network  
> 재구성을 기반으로 이상 영역을 탐지하는 DirectNet 구조 구현 (PyTorch)

---

## 📌 프로젝트 개요

본 프로젝트는 초분광 영상(Hyperspectral Imaging, HSI)에서 정상 스펙트럼 복원을 통해 **이상 탐지**를 수행하는 **Self-Supervised Reconstruction Model – DirectNet**을 구현한 것입니다.

학습은 오직 정상(normal) 라벨만 사용하며, 이상(anomaly)은 재구성 오류 기반으로 추론됩니다.

---

## 📁 디렉토리 구성

```
DirectNet-HSI/
├── checkpoints/             # 학습된 모델 저장 디렉토리
├── figs/                    # 시각화된 예시 결과 저장
├── directnet.ipynb          # Jupyter 기반 전체 실행 파이프라인
├── direct_train.py          # 학습 스크립트
├── direct_predict.py        # 추론 스크립트
├── evaluate_directnet.py    # 정량적 평가 (AUC, F1 등)
├── directnet_config.yaml    # 설정 파일
├── model.py                 # DirectNet 모델 정의
├── layer.py                 # 모델 구성 Layer 정의
├── utils.py                 # 데이터 처리, 보조 함수
└── README.md                # 설명 파일 (현재 문서)
```

---

## ⚙️ 설치 환경

```bash
conda create -n directnet python=3.9
conda activate directnet
pip install torch torchvision numpy matplotlib scikit-learn tqdm
```

PyTorch >= 1.10 이상 권장 (CUDA 지원)

---

## 🧪 실행 방법 (노트북 기반)

Jupyter 환경에서 다음 노트북을 실행합니다:

```bash
directnet.ipynb
```


## 📌 주요 특징

| 항목          | 설명 |
|---------------|------|
| 입력          | 정규화된 HSI 데이터 (H×W×C, 예: 224 bands) |
| 모델 구조     | Patch-level CNN 기반 재구성 네트워크 |
| 학습 방식     | 정상 영역만 사용한 self-supervised 방식 |
| 이상 탐지     | 재구성 오차 기반 |
| 시각화 지원   | RGB 변환, 에러맵, 마스크 |
| 평가 지표     | AUC, F1-score, Accuracy 등 지원 |
| AMP           | 지원 (자동 Mixed Precision 학습) |

---

## 🔗 원작자 및 참고 자료

본 프로젝트는 아래의 논문 및 코드 구현을 기반으로 재구성 및 일부 수정되었습니다.

- 📘 논문: Wang et al., "DirectNet: End-to-End Anomaly Localization Network for Hyperspectral Imagery", IEEE Transactions on Geoscience and Remote Sensing, 2022.
- 🔗 공식 코드: [DegangWang97/IEEE_TGRS_DirectNet](https://github.com/DegangWang97/IEEE_TGRS_DirectNet)

> 본 리포지토리는 상기 코드를 기반으로 **직관적 실행 구조(Jupyter/Script), 시각화 기능, 평가 자동화**를 추가한 파생 프로젝트입니다.

---

## 🖼️ 결과 예시

> (추론 결과 이미지 예시 삽입 가능)

---

## 📝 설정 파일 예시 (directnet_config.yaml)

```yaml
epochs: 100
batch_size: 64
patch_size: 17
lr: 0.001
save_freq: 10
patience: 30
dataset: YourDatasetName
```
