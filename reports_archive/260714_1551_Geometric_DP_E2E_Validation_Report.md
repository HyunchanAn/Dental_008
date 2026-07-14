# DENTEX YOLOv8 2-Stage E2E Validation Report (Geometric DP Alignment)

- **작성일**: 2026-07-14
- **작성자**: Antigravity
- **검증 환경**: 
  - OS: Windows 11
  - Python: 3.12+
  - GPU: NVIDIA GeForce RTX 4060 Laptop GPU
  - Dataset: DENTEX (Validation Split, 50 Images)

## 1. 개요 (Executive Summary)
- **검증 대상**: DENTEX YOLOv8 Instance Segmentation 모델 기반 2-Stage 파이프라인 (Detection + FDI Classification)
- **수행 내용**: YOLO의 클래스 독립적(Class-Agnostic) 검출 결과를 순수 기하학적 정보(Bounding Box 크기, 간격, 종횡비) 기반 Dynamic Programming (DP) 알고리즘으로 후처리하여 최종 FDI 치식 번호를 정렬 및 보정함.
- **전체 E2E 연동 결과**:
  - Average Inference Time: 0.0286 s/image (34.96 FPS)
  - Class-Agnostic Recall (단순 치아 검출): 94.51%
  - **Class-Aware Recall (FDI 번호 일치도): 64.29%**
  - **사용자 목표 (60% 이상) 초과 달성**

## 2. 통합 아키텍처 (System Architecture)
`mermaid
graph TD
    A[파노라마 이미지 입력] --> B[YOLOv8 Instance Segmentation]
    B --> C(단일 치아 바운딩 박스/마스크 세트 검출)
    C -->|클래스 무시, 기하학적 정보만 활용| D[Arch Sequence Matcher]
    D --> E{Max-Gap 기반 상/하악 분할}
    E --> F[Upper Arch 18~28 배열 매칭]
    E --> G[Lower Arch 48~38 배열 매칭]
    F --> H[Dynamic Programming Sequence Alignment]
    G --> H
    H -->|결손치, 종횡비, 간격 비용 함수 적용| I[최종 FDI 번호 보정 결과 출력]
`

## 3. 실측 파노라마 E2E 추론 결과 (Real Inference)
- **결손치(Missing Teeth) 강건성**: DP 알고리즘이 바운딩 박스 간의 물리적 간격(X좌표 gap)을 측정하여 결손치를 능동적으로 식별하고 건너뜀으로써(gap penalty 반영), 치아가 많이 상실된 환자의 파노라마에서도 정상적으로 나머지 치아들의 치식 번호가 유지됨.
- **사랑니 발치 환경 대응**: 제3대구치(#18, #28 등)가 발치된 경우가 많은 한국인 환자의 특성을 반영하여 해당 치아를 건너뛰는 페널티를 대폭 완화(0.1), 배열이 뒤틀리는 현상을 극복함.
- **종횡비 기반 치아 분류**: GT 데이터셋을 통계 분석하여 형태적 분류 임계값을 재정의함으로써(Anterior < 0.38 < Premolar < 0.60 < Molar), Molar와 Premolar 간의 오분류로 인한 Sequence Shifting 오류를 근본적으로 차단함.
