import os
import glob
from PIL import Image
import imagehash

def clean_images_phash(target_dir):
    files = glob.glob(os.path.join(target_dir, '*.jpg'))
    
    seen_hashes = {}
    duplicates = []
    
    for f in files:
        try:
            img = Image.open(f)
            # pHash (Perceptual Hash) 계산
            h = imagehash.phash(img)
            
            is_dup = False
            # 해시 거리가 5 이하면 시각적으로 동일한(혹은 거의 동일한) 이미지로 간주
            for seen_h, seen_f in seen_hashes.items():
                if abs(h - seen_h) <= 5:
                    is_dup = True
                    duplicates.append(f)
                    break
            
            if not is_dup:
                seen_hashes[h] = f
                
            img.close()
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    for d in duplicates:
        os.remove(d)
        print(f"Removed visual duplicate: {os.path.basename(d)}")
        
    print(f"\n총 {len(duplicates)}개의 '시각적' 중복 파일 삭제 완료.")

if __name__ == "__main__":
    target = r"\\rtx4060laptop-hc\Users\chema\Github\Dental_008\edentulous_images"
    clean_images_phash(target)
