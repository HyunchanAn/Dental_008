import os
from pathlib import Path
import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    print("========================================")
    print("YOLOv8m-seg Hard Negative 오탐지 억제 파인튜닝 시작")
    print("========================================")

    weight_path = Path(
        os.getenv(
            "DENTAL_008_BASE_WEIGHTS",
            str(REPO_ROOT / "weights" / "best.pt"),
        )
    )
    model = YOLO(str(weight_path))

    yaml_path = os.path.expanduser("~/.cache/dentex_dataset/yolo_format_v3/data.yaml")
    default_device = 0 if torch.cuda.is_available() else "cpu"

    model.train(
        data=yaml_path,
        epochs=30,
        imgsz=800,
        batch=8,
        device=default_device,
        project="yolo_dentex",
        name="yolov8m_seg_finetune",
        amp=torch.cuda.is_available(),
        workers=8,
        lr0=0.001,
    )

    print("\n파인튜닝이 성공적으로 종료되었습니다. 가중치는 yolo_dentex/yolov8m_seg_finetune 폴더에 저장됩니다.")


if __name__ == "__main__":
    main()
