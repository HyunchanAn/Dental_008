from ultralytics import YOLO
import os

def main():
    print("========================================")
    print("YOLOv8m-seg 기반 DENTEX 치아 인스턴스 분할 학습 시작")
    print("========================================")
    
    # 모델 로드 (가중치 초기화 포함)
    model = YOLO('yolov8m-seg.pt')

    # 데이터셋 YAML 경로
    yaml_path = os.path.expanduser('~/.cache/dentex_dataset/yolo_format/data.yaml')
    
    # 학습 파라미터 (RTX 5080에 최적화)
    results = model.train(
        data=yaml_path,
        epochs=30,
        imgsz=800, # 파노라마 사진이 가로로 넓으므로 약간 높은 해상도 사용
        batch=8,
        device=0,
        project='yolo_dentex',
        name='yolov8m_seg_run',
        amp=True, # 텐서코어 가속
        workers=8
    )
    
    print("\n모든 학습이 성공적으로 종료되었습니다. 가중치는 yolo_dentex/yolov8m_seg_run 폴더에 저장됩니다.")

if __name__ == '__main__':
    main()
