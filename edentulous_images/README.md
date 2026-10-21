# Edentulous & Implant Hard Negative Images (`edentulous_images/`)

## 출처 및 라이선스 고지 (Provenance & Usage Policy)
- 구성: 무치악(Edentulous) 악궁 및 다수 임플란트 식립 파노라마 방사선 영상 75장 (`edentulous_0001.jpg` ~ `edentulous_0079.jpg`, 고유 SHA-256 해시 75장) 및 공개 학술 증례 참조 이미지 1장 (`dentistry-oraldisorders-therapy38-g007.gif`).
- 수집 목적: `YOLOv8m-seg` 영구치 분할 모델이 무치악 치조골 융기나 임플란트 픽스처를 자연치아로 오탐(False Positive)하는 현상을 억제하기 위한 배경(Background / 0-object) Hard Negative 연구 샘플.
- 분할 정책 (`scripts/prepare_v3_dataset.py`):
  - 고정 시드(`seed=42`) 기준 `Train 60장` (학습 배경 데이터 병합) 및 `Holdout Val 15장` (학습에 일절 노출되지 않는 독립 검증셋)으로 분리(파일명 및 SHA-256 바이트 해시 교집합 `0건`).
- 이용 제한:
  - 본 폴더의 영상은 웹 공개 학술/교육 증례에서 비식별 상태로 수집·정규화된 비영리 학술 연구 전용 참고 자료이며, 일부 영상에는 원 출처 워터마크가 포함되어 있습니다.
  - 상업적 재배포 또는 임상 진단 목적의 사용을 금하며, 권리자의 삭제 요청 시 즉시 제외됩니다.
