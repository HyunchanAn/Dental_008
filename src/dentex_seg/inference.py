import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dentex_seg.model import get_instance_segmentation_model
from dentex_seg.dataset import get_fdi_to_class_id
import os

def visualize_inference(image_path, model_path, output_path):
    print(f"입력 이미지 [{image_path}]에 대한 모델 추론을 시작합니다...")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # 모델 로드 (53개 클래스)
    num_classes = 53
    model = get_instance_segmentation_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    _, id_to_fdi = get_fdi_to_class_id()

    # 이미지 불러오기 및 텐서 변환
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)
    image_tensor = torch.as_tensor(image_np/255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    print("모델에 이미지를 통과시킵니다...")
    with torch.no_grad():
        predictions = model(image_tensor)

    pred = predictions[0]
    masks = (pred['masks'] > 0.5).squeeze(1).cpu().numpy()
    boxes = pred['boxes'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()

    # 시각화 기본 설정
    fig, ax = plt.subplots(1, figsize=(16, 10))
    ax.imshow(image_np)

    count = 0
    for i in range(len(boxes)):
        if scores[i] < 0.5: # 신뢰도가 0.5 이상인 결과만 표시
            continue
            
        count += 1
        box = boxes[i]
        mask = masks[i]
        label = labels[i]
        fdi_number = id_to_fdi.get(label, "Unknown")

        # 인스턴스 마스크 겹쳐서 그리기 (랜덤 색상 적용)
        color = np.random.rand(3)
        masked_image = np.ma.masked_where(mask == 0, mask)
        ax.imshow(masked_image, cmap='hsv', alpha=0.45)

        # 바운딩 박스 그리기
        rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], 
                             fill=False, edgecolor=color, linewidth=2.5)
        ax.add_patch(rect)

        # FDI 치식 번호 텍스트 삽입
        ax.text(box[0], box[1] - 8, f"FDI: {fdi_number} ({scores[i]:.2f})", 
                bbox=dict(facecolor='white', alpha=0.85, edgecolor=color, linewidth=1),
                fontsize=11, color='black', weight='bold')

    plt.axis('off')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"성공적으로 {count}개의 객체(영구치/유치/치배)를 식별했습니다.")
    print(f"추론 결과가 다음 경로에 저장되었습니다: {output_path}")

if __name__ == "__main__":
    # 실행 예시 안내
    print("이 코드는 임포트해서 사용하거나 주석을 해제하여 단독으로 실행할 수 있습니다.")
    print("-------------------------------------------------------------------------")
    # 아래 코드를 주석 해제하여 사용할 수 있습니다.
    # image_path = "sample_xray.jpg"
    # model_path = "weights/mask_rcnn_dentex_epoch_5.pth"
    # output_path = "output/segmentation_result.jpg"
    # visualize_inference(image_path, model_path, output_path)
