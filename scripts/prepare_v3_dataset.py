import os
import shutil
import glob
from tqdm import tqdm
import yaml

def prepare_v3_dataset():
    src_dir = os.path.expanduser('~/.cache/dentex_dataset/yolo_format')
    dst_dir = os.path.expanduser('~/.cache/dentex_dataset/yolo_format_v3')
    edentulous_dir = r"\\rtx4060laptop-hc\Users\chema\Github\Dental_008\edentulous_images"
    
    # 1. 이전 v3 폴더 초기화
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
        
    os.makedirs(os.path.join(dst_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'labels', 'val'), exist_ok=True)
    
    # 2. 원본(크롭 없는) 데이터만 복사
    print("Copying original dataset...")
    for split in ['train', 'val']:
        src_imgs = glob.glob(os.path.join(src_dir, 'images', split, '*.*'))
        for src_img in tqdm(src_imgs, desc=f"Copying {split} images"):
            filename = os.path.basename(src_img)
            if 'hardneg_' in filename:
                continue # 크롭 데이터 제외
                
            dst_img = os.path.join(dst_dir, 'images', split, filename)
            shutil.copy2(src_img, dst_img)
            
            # 라벨 복사
            basename = os.path.splitext(filename)[0]
            src_label = os.path.join(src_dir, 'labels', split, basename + '.txt')
            dst_label = os.path.join(dst_dir, 'labels', split, basename + '.txt')
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
                
    # 3. 75장 무치악 데이터 병합 (train 폴더로)
    print("Merging edentulous images...")
    edentulous_imgs = glob.glob(os.path.join(edentulous_dir, '*.jpg'))
    for img_path in tqdm(edentulous_imgs, desc="Merging Edentulous"):
        filename = os.path.basename(img_path)
        dst_img = os.path.join(dst_dir, 'images', 'train', filename)
        shutil.copy2(img_path, dst_img)
        
        # 빈 라벨 파일 생성
        basename = os.path.splitext(filename)[0]
        dst_label = os.path.join(dst_dir, 'labels', 'train', basename + '.txt')
        with open(dst_label, 'w') as f:
            pass # 객체 0개인 배경
            
    # 4. yaml 생성
    data_yaml = {
        'path': dst_dir,
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'Tooth'}
    }
    with open(os.path.join(dst_dir, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
        
    print(f"Data preparation complete! YAML saved to {os.path.join(dst_dir, 'data.yaml')}")

if __name__ == '__main__':
    prepare_v3_dataset()
