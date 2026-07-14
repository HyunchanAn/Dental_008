from ultralytics import YOLO
import glob
import os

def test():
    model_v4 = YOLO(r"\\rtx4060laptop-hc\Users\chema\Github\Dental_000\yolov8m_seg_scratch\weights\best.pt")
    model_v2 = YOLO(r"\\rtx4060laptop-hc\Users\chema\Github\Dental_008\runs\segment\yolo_dentex\yolov8m_seg_run\weights\best.pt")
    
    imgs = glob.glob(r"\\rtx4060laptop-hc\Users\chema\Github\Dental_008\edentulous_images\*.jpg")[:10]
    
    total_v2 = 0
    total_v4 = 0
    print("Testing on 10 Edentulous Images:")
    for img in imgs:
        res_v2 = model_v2(img, conf=0.25, verbose=False)[0]
        res_v4 = model_v4(img, conf=0.25, verbose=False)[0]
        
        c2 = len(res_v2.boxes) if res_v2.boxes else 0
        c4 = len(res_v4.boxes) if res_v4.boxes else 0
        
        print(f"{os.path.basename(img)} - v2(Old) predicted: {c2}, v4(New) predicted: {c4}")
        total_v2 += c2
        total_v4 += c4
        
    print(f"\nTotal false detections on 10 images - v2: {total_v2}, v4: {total_v4}")

if __name__ == "__main__":
    test()
