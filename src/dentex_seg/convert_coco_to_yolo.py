import os
import json
import shutil
from tqdm import tqdm
from pathlib import Path

def convert_coco_json_to_yolo_seg(json_path, img_dir, output_dir, split_name):
    print(f"Converting {split_name}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    images = {img['id']: img for img in data['images']}
    
    img_out_dir = os.path.join(output_dir, 'images', split_name)
    label_out_dir = os.path.join(output_dir, 'labels', split_name)
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(label_out_dir, exist_ok=True)
    
    img_to_anns = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
        
    for img_id, img_info in tqdm(images.items()):
        file_name = img_info['file_name']
        width = img_info['width']
        height = img_info['height']
        
        src_img = os.path.join(img_dir, file_name)
        dst_img = os.path.join(img_out_dir, file_name)
        
        # YOLO labels
        anns = img_to_anns.get(img_id, [])
        label_file = os.path.splitext(file_name)[0] + '.txt'
        label_path = os.path.join(label_out_dir, label_file)
        
        valid_boxes = False
        with open(label_path, 'w') as f:
            for ann in anns:
                # Polygons to normalized YOLO seg format
                polygons = ann.get('segmentation', [])
                if len(polygons) == 0:
                    continue
                    
                for poly in polygons:
                    # poly is [x1, y1, x2, y2, ...]
                    normalized_poly = []
                    for i in range(0, len(poly), 2):
                        x = poly[i] / width
                        y = poly[i+1] / height
                        # 클램핑 (0~1)
                        x = max(0.0, min(1.0, x))
                        y = max(0.0, min(1.0, y))
                        normalized_poly.append(f"{x:.6f} {y:.6f}")
                        
                    if len(normalized_poly) >= 3:
                        class_id = 0 # Force all to 0 (Tooth)
                        f.write(f"{class_id} " + " ".join(normalized_poly) + "\n")
                        valid_boxes = True
                        
        if valid_boxes:
            if not os.path.exists(dst_img):
                shutil.copy2(src_img, dst_img)
        else:
            if os.path.exists(label_path):
                os.remove(label_path)

if __name__ == '__main__':
    cache_dir = os.path.expanduser("~/.cache/dentex_dataset")
    yolo_dir = os.path.join(cache_dir, "yolo_format")
    
    # Train
    train_json = os.path.join(cache_dir, "training_data/quadrant_enumeration/train_quadrant_enumeration.json")
    train_img = os.path.join(cache_dir, "training_data/quadrant_enumeration/xrays")
    if os.path.exists(train_json):
        convert_coco_json_to_yolo_seg(train_json, train_img, yolo_dir, "train")
        
    # Val
    val_json = os.path.join(cache_dir, "validation_triple.json")
    val_img = os.path.join(cache_dir, "validation_data/quadrant_enumeration_disease/xrays")
    if os.path.exists(val_json):
        convert_coco_json_to_yolo_seg(val_json, val_img, yolo_dir, "val")
        
    # Create data.yaml
    yaml_path = os.path.join(yolo_dir, "data.yaml")
    
    # Windows 경로 이스케이프 방지
    yolo_dir_fwd = yolo_dir.replace("\\", "/")
    
    yaml_content = f"""path: {yolo_dir_fwd}
train: images/train
val: images/val

names:
  0: Tooth
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print("data.yaml created at:", yaml_path)
