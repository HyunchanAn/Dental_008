import os
import hashlib
import glob
from PIL import Image

def get_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def clean_images(target_dir):
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.PNG', '.JPEG'}
    
    files = []
    for f in os.listdir(target_dir):
        ext = os.path.splitext(f)[1]
        if ext in valid_exts:
            files.append(os.path.join(target_dir, f))
            
    seen_hashes = {}
    duplicates = []
    
    for f in files:
        h = get_hash(f)
        if h in seen_hashes:
            duplicates.append(f)
        else:
            seen_hashes[h] = f
            
    for d in duplicates:
        os.remove(d)
        print(f"Removed duplicate: {os.path.basename(d)}")
        
    unique_files = list(seen_hashes.values())
    unique_files.sort()
    
    # 1차적으로 임시 이름으로 싹 변경 (이름 충돌 방지)
    temp_files = []
    for i, f in enumerate(unique_files):
        temp_name = os.path.join(target_dir, f"temp_{i:04d}.jpg")
        try:
            img = Image.open(f)
            img = img.convert('RGB')
            img.save(temp_name, 'JPEG')
            img.close()
            os.remove(f)
            temp_files.append(temp_name)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    # 2차적으로 최종 이름으로 변경
    count = 1
    for f in temp_files:
        new_name = os.path.join(target_dir, f"edentulous_{count:04d}.jpg")
        os.rename(f, new_name)
        count += 1
            
    print(f"\n총 {len(duplicates)}개의 중복 파일 삭제 완료.")
    print(f"총 {count-1}개의 이미지가 edentulous_XXXX.jpg 형태로 정규화되었습니다.")

if __name__ == "__main__":
    target = r"\\rtx4060laptop-hc\Users\chema\Github\Dental_008\edentulous_images"
    clean_images(target)
