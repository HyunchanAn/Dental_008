from ultralytics import YOLO
import os

def main():
    print("========================================")
    print("YOLOv8m-seg Hard Negative 오탐지 억제 파인튜닝 시작")
    print("========================================")
    
    # 기존에 학습한 베스트 가중치 로드
    weight_path = r"\\rtx4060laptop-hc\Users\chema\Github\Dental_008\runs\segment\yolo_dentex\yolov8m_seg_run\weights\best.pt"
    model = YOLO(weight_path)

    # 데이터셋 YAML 경로 (Hard negative 크롭 이미지가 추가되어 있음)
    yaml_path = os.path.expanduser('~/.cache/dentex_dataset/yolo_format/data.yaml')
    
    # 파인튜닝 파라미터 (학습률을 낮춤)
    results = model.train(
        data=yaml_path,
        epochs=30,
        imgsz=800,
        batch=8,
        device=0,
        project='yolo_dentex',
        name='yolov8m_seg_finetune',
        amp=True,
        workers=8,
        lr0=0.001
    )
    
    print("\n파인튜닝이 성공적으로 종료되었습니다. 가중치는 yolo_dentex/yolov8m_seg_finetune 폴더에 저장됩니다.")

if __name__ == '__main__':
    main()
