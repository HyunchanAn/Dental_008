import os
import cv2
import glob
import random
import numpy as np
from tqdm import tqdm

def check_overlap(crop_box, boxes):
    """
    crop_box: [cx1, cy1, cx2, cy2]
    boxes: 리스트 of [bx1, by1, bx2, by2]
    """
    cx1, cy1, cx2, cy2 = crop_box
    for b in boxes:
        bx1, by1, bx2, by2 = b
        
        # 교집합 영역 계산
        ix1 = max(cx1, bx1)
        iy1 = max(cy1, by1)
        ix2 = min(cx2, bx2)
        iy2 = min(cy2, by2)
        
        if ix1 < ix2 and iy1 < iy2:
            return True # 단 1픽셀이라도 겹침
    return False

def generate_hard_negatives(yolo_dir, target_crops=500, crop_size=400):
    img_dir = os.path.join(yolo_dir, 'images', 'train')
    label_dir = os.path.join(yolo_dir, 'labels', 'train')
    
    img_paths = glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png'))
    # 기존 크롭 이미지 제외
    img_paths = [p for p in img_paths if 'hardneg_' not in os.path.basename(p)]
    
    generated = 0
    pbar = tqdm(total=target_crops, desc="Generating Hard Negatives")
    
    # 충분한 데이터 생성을 위해 여러 번 순회
    while generated < target_crops:
        random.shuffle(img_paths)
        for img_path in img_paths:
            if generated >= target_crops:
                break
                
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(label_dir, base_name + '.txt')
            
            if not os.path.exists(label_path):
                continue
                
            # 이미지 로드
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            
            # 너무 작은 이미지는 스킵
            if h <= crop_size or w <= crop_size:
                continue
                
            # 바운딩 박스 파싱
            boxes = []
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # 분할(Segmentation) 포맷인 경우: class x1 y1 x2 y2 ...
                        # YOLOv8-seg 포맷은 폴리곤이므로, x의 최소최대, y의 최소최대를 구해 박스로 변환
                        coords = list(map(float, parts[1:]))
                        xs = coords[0::2]
                        ys = coords[1::2]
                        
                        min_x, max_x = min(xs) * w, max(xs) * w
                        min_y, max_y = min(ys) * h, max(ys) * h
                        
                        boxes.append([min_x, min_y, max_x, max_y])
            
            # 무작위 크롭 시도 (한 이미지당 10번 시도하여 1개 성공하면 진행)
            for _ in range(10):
                rx = random.randint(0, w - crop_size)
                ry = random.randint(0, h - crop_size)
                crop_box = [rx, ry, rx + crop_size, ry + crop_size]
                
                if not check_overlap(crop_box, boxes):
                    # 겹치지 않으면 크롭 저장
                    crop_img = img[ry:ry+crop_size, rx:rx+crop_size]
                    
                    new_img_name = f"hardneg_{base_name}_{generated:04d}.jpg"
                    new_label_name = f"hardneg_{base_name}_{generated:04d}.txt"
                    
                    cv2.imwrite(os.path.join(img_dir, new_img_name), crop_img)
                    
                    # 빈 라벨 파일 생성
                    with open(os.path.join(label_dir, new_label_name), 'w') as f:
                        pass # 아무 내용도 쓰지 않음 (객체 0개)
                        
                    generated += 1
                    pbar.update(1)
                    break # 한 이미지에서 하나 건지면 다음 이미지로
                    
    pbar.close()
    print(f"\n{generated}개의 Hard Negative 크롭 데이터 생성이 완료되었습니다.")

if __name__ == '__main__':
    yolo_dir = os.path.expanduser('~/.cache/dentex_dataset/yolo_format')
    generate_hard_negatives(yolo_dir, target_crops=500, crop_size=320)
