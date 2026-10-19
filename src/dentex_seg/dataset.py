import os
import zipfile
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download

def get_fdi_to_class_id():
    """FDI 번호를 1~52 사이의 고유 클래스 ID로 매핑합니다. 0은 배경입니다."""
    fdi_list = []
    # 영구치 (11~18, 21~28, 31~38, 41~48)
    for q in [1, 2, 3, 4]:
        for t in range(1, 9):
            fdi_list.append(q * 10 + t)
    # 유치 (51~55, 61~65, 71~75, 81~85)
    for q in [5, 6, 7, 8]:
        for t in range(1, 6):
            fdi_list.append(q * 10 + t)
            
    fdi_to_id = {fdi: i+1 for i, fdi in enumerate(fdi_list)}
    id_to_fdi = {i+1: fdi for i, fdi in enumerate(fdi_list)}
    return fdi_to_id, id_to_fdi

class DENTEXDataset(Dataset):
    def __init__(self, split='train', transforms=None, seed=42, use_leak_free_split=True):
        self.transforms = transforms
        self.split = split
        self.fdi_to_id, self.id_to_fdi = get_fdi_to_class_id()

        # 캐시 디렉토리 설정
        cache_dir = os.path.expanduser("~/.cache/dentex_dataset")
        os.makedirs(cache_dir, exist_ok=True)

        self.img_dir = os.path.join(cache_dir, "training_data/quadrant_enumeration/xrays")
        self.json_path = os.path.join(cache_dir, "training_data/quadrant_enumeration/train_quadrant_enumeration.json")

        if not os.path.exists(self.json_path):
            print("Hugging Face에서 DENTEX training_data.zip 다운로드 중 (10.9 GB)...")
            zip_path = hf_hub_download(repo_id="LUNA0206/DENTEX", repo_type="dataset", filename="DENTEX/training_data.zip")
            print("필요한 학습 데이터셋 압축 해제 중...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                members = [m for m in zip_ref.namelist() if m.startswith("training_data/quadrant_enumeration/")]
                zip_ref.extractall(path=cache_dir, members=members)
            print("압축 해제 완료.")

        # COCO 포맷 JSON 로드
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.coco_data = json.load(f)

        all_images = {img['id']: img for img in self.coco_data['images']}

        if use_leak_free_split and split in ('train', 'val'):
            import csv
            import random
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            dental_000_root = os.path.abspath(os.path.join(repo_root, "..", "Dental_000"))
            manifest_candidates = [
                os.path.join(dental_000_root, "reports_archive", "dental_008_dentex_manifest.csv"),
                os.path.join(dental_000_root, "reports", "dental_008_dentex_manifest.csv"),
            ]
            manifest_path = next((p for p in manifest_candidates if os.path.exists(p)), None)
            if manifest_path is not None:
                train_files, val_files = set(), set()
                with open(manifest_path, "r", encoding="utf-8") as mf:
                    for row in csv.DictReader(mf):
                        if row["split"].strip() == "train":
                            train_files.add(row["filename"].strip())
                        elif row["split"].strip() == "val":
                            val_files.add(row["filename"].strip())
            else:
                all_fnames = sorted(img["file_name"] for img in all_images.values())
                rng = random.Random(seed)
                rng.shuffle(all_fnames)
                n_train = int(round(len(all_fnames) * 0.8))
                train_files = set(all_fnames[:n_train])
                val_files = set(all_fnames[n_train:])

            assert len(train_files & val_files) == 0, "Data leakage detected: Train and Val sets overlap!"
            target_files = train_files if split == "train" else val_files
            self.images = {img_id: img for img_id, img in all_images.items() if img["file_name"] in target_files}
        else:
            self.images = all_images

        # 이미지 ID별 주석 그룹화
        self.img_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id in self.images:
                self.img_to_anns.setdefault(img_id, []).append(ann)

        self.img_ids = list(self.images.keys())
        print(f"[{split}] 로드 완료. 총 이미지 수: {len(self.img_ids)}")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        
        # 이미지 파일명 및 경로
        file_name = img_info['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        
        # 이미지 로드
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        
        annotations = self.img_to_anns.get(img_id, [])
        
        boxes = []
        labels = []
        masks = []
        
        for ann in annotations:
            # category_id_1 & category_id_2가 있는 DENTEX 포맷이거나 단일 category_id 포맷 지원
            cat_1 = ann.get('category_id_1')
            cat_2 = ann.get('category_id_2')
            cat_id = ann.get('category_id')
            
            if cat_1 is not None and cat_2 is not None:
                fdi = (cat_1 + 1) * 10 + (cat_2 + 1)
            elif cat_id is not None:
                # category_id 자체가 FDI 번호인 경우 (예: 11~48, 51~85)
                if cat_id in self.fdi_to_id:
                    fdi = cat_id
                # category_id가 1~52 사이의 class_id인 경우
                elif cat_id in self.id_to_fdi:
                    fdi = self.id_to_fdi[cat_id]
                # category_id가 0~51 사이의 0-indexed class_id인 경우
                elif (cat_id + 1) in self.id_to_fdi:
                    fdi = self.id_to_fdi[cat_id + 1]
                else:
                    continue
            else:
                continue
                
            if fdi not in self.fdi_to_id:
                continue
                
            class_id = 1 if self.split == 'train' else self.fdi_to_id[fdi]
            
            # 폴리곤 좌표에서 마스크 생성
            polygon = ann.get('segmentation', [])
            if len(polygon) == 0:
                continue
                
            mask = np.zeros((h, w), dtype=np.uint8)
            for poly in polygon:
                poly_pts = np.array(poly).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [poly_pts], 1)
                
            masks.append(mask)
            labels.append(class_id)
            
            # Bounding Box [x_min, y_min, x_max, y_max] 포맷
            if 'bbox' in ann:
                bbox = ann['bbox'] # [x, y, width, height]
                boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            else:
                pos = np.where(mask)
                if len(pos[0]) > 0:
                    xmin = np.min(pos[1])
                    xmax = np.max(pos[1])
                    ymin = np.min(pos[0])
                    ymax = np.max(pos[0])
                    boxes.append([xmin, ymin, xmax, ymax])
                else:
                    boxes.append([0, 0, w, h])
                    
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, h, w), dtype=torch.uint8)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        
        # 텐서 변환 및 [0, 1] 정규화
        image = torch.as_tensor(image/255.0, dtype=torch.float32).permute(2, 0, 1)
            
        return image, target
