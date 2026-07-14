from ultralytics import YOLO
import os

def train_yolo():
    # COCO 사전학습된 순수 가중치(yolov8m-seg.pt)를 베이스로 처음부터 재학습
    model = YOLO('yolov8m-seg.pt')
    
    yaml_path = os.path.expanduser('~/.cache/dentex_dataset/yolo_format_v3/data.yaml')
    
    results = model.train(
        data=yaml_path,
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,
        project=r"\\rtx4060laptop-hc\Users\chema\Github\Dental_000",
        name="yolov8m_seg_scratch",
        device='0',
        exist_ok=True
    )
    print("Training from scratch complete.")

if __name__ == '__main__':
    train_yolo()
