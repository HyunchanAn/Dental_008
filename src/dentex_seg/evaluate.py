import torch
import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dentex_seg.dataset import DENTEXDataset
from dentex_seg.model import get_instance_segmentation_model

def compute_iou(box1, box2):
    # box: [xmin, ymin, xmax, ymax]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = inter_area / float(box1_area + box2_area - inter_area + 1e-6)
    return iou

def compute_mask_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / float(union + 1e-6)

def determine_dentition_type(labels, id_to_fdi):
    has_permanent = False
    has_deciduous = False
    for lbl in labels:
        fdi = id_to_fdi.get(lbl.item(), -1)
        if 11 <= fdi <= 48:
            has_permanent = True
        elif 51 <= fdi <= 85:
            has_deciduous = True
            
    if has_permanent and has_deciduous:
        return "mixed"
    elif has_permanent and not has_deciduous:
        return "permanent"
    elif has_deciduous and not has_permanent:
        return "deciduous"
    return "unknown"

def visualize_and_save(image, boxes, masks, labels, id_to_fdi, save_path):
    # image: [C, H, W] tensor
    img_np = image.permute(1, 2, 0).cpu().numpy().copy()
    
    plt.figure(figsize=(12, 6))
    plt.imshow(img_np)
    ax = plt.gca()
    
    for i in range(len(boxes)):
        box = boxes[i].cpu().numpy()
        mask = masks[i].cpu().numpy()
        label = labels[i].item()
        fdi = id_to_fdi.get(label, -1)
        
        if fdi == -1:
            continue
            
        # Color: Permanent (11~48) -> Blue, Deciduous (51~85) -> Cyan
        if 11 <= fdi <= 48:
            color = np.array([0.0, 0.0, 1.0]) # Blue
            hex_color = 'blue'
        elif 51 <= fdi <= 85:
            color = np.array([0.0, 1.0, 1.0]) # Cyan (청록색)
            hex_color = 'cyan'
        else:
            color = np.array([1.0, 0.0, 0.0]) # Red for unknown
            hex_color = 'red'
        
        # Draw Label (centered roughly at the top of the box)
        ax.text(box[0] + (box[2]-box[0])/2, box[1]-5, str(fdi), color='white', fontsize=10, weight='bold', ha='center', bbox=dict(facecolor=hex_color, alpha=0.5, edgecolor='none', pad=1))
        
        # Draw Mask only
        colored_mask = np.zeros_like(img_np)
        for c in range(3):
            colored_mask[:, :, c] = color[c]
        
        # Overlay mask with alpha 0.4
        mask_bool = mask > 0.5
        img_np[mask_bool] = img_np[mask_bool] * 0.6 + colored_mask[mask_bool] * 0.4

    plt.imshow(img_np)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def evaluate():
    print("DENTEX E2E 성능 평가 및 시각화 리포트 생성 시작...")
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset_val = DENTEXDataset(split='val')
    
    num_classes = 53
    model = get_instance_segmentation_model(num_classes)
    
    # Load weights
    weight_path = "E:/Github/Dental_008/weights/mask_rcnn_dentex_epoch_30.pth"
    if not os.path.exists(weight_path):
        weight_path = "E:/Github/Dental_008/weights/pretrained/mask_rcnn_dentex_epoch_30.pth"
    
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()

    total_time = 0
    total_box_iou = 0
    total_mask_iou = 0
    num_matches = 0

    found_permanent = False
    found_mixed = False
    found_deciduous = False
    
    os.makedirs('E:/Github/Dental_008/reports_archive/images', exist_ok=True)

    with torch.no_grad():
        # Evaluate all images to calculate robust metrics and find cases
        num_eval = len(dataset_val)
        for idx in range(num_eval):
            img, target = dataset_val[idx]
            img_tensor = img.to(device).unsqueeze(0)
            
            start_time = time.time()
            output = model(img_tensor)[0]
            end_time = time.time()
            
            total_time += (end_time - start_time)
            
            gt_boxes = target['boxes']
            gt_masks = target['masks']
            gt_labels = target['labels']
            
            pred_boxes = output['boxes']
            pred_masks = output['masks'][:, 0]
            pred_labels = output['labels']
            pred_scores = output['scores']
            
            # Filter low confidence
            keep = pred_scores > 0.5
            pred_boxes = pred_boxes[keep]
            pred_masks = pred_masks[keep]
            pred_labels = pred_labels[keep]
            
            # Calculate IoU for matched labels
            for i, p_label in enumerate(pred_labels):
                for j, g_label in enumerate(gt_labels):
                    if p_label == g_label:
                        iou = compute_iou(pred_boxes[i].cpu().numpy(), gt_boxes[j].cpu().numpy())
                        if iou > 0.3:
                            m_iou = compute_mask_iou(pred_masks[i].cpu().numpy() > 0.5, gt_masks[j].cpu().numpy() > 0.5)
                            total_box_iou += iou
                            total_mask_iou += m_iou
                            num_matches += 1
            
            # Case detection for visualization
            dent_type = determine_dentition_type(gt_labels, dataset_val.id_to_fdi)
            
            if dent_type == "permanent" and not found_permanent:
                visualize_and_save(img, pred_boxes, pred_masks, pred_labels, dataset_val.id_to_fdi, "E:/Github/Dental_008/reports_archive/images/eval_permanent.jpg")
                found_permanent = True
            elif dent_type == "mixed" and not found_mixed:
                visualize_and_save(img, pred_boxes, pred_masks, pred_labels, dataset_val.id_to_fdi, "E:/Github/Dental_008/reports_archive/images/eval_mixed.jpg")
                found_mixed = True
            elif dent_type == "deciduous" and not found_deciduous:
                visualize_and_save(img, pred_boxes, pred_masks, pred_labels, dataset_val.id_to_fdi, "E:/Github/Dental_008/reports_archive/images/eval_deciduous.jpg")
                found_deciduous = True

    avg_time = total_time / num_eval
    fps = 1.0 / avg_time
    avg_box_iou = total_box_iou / num_matches if num_matches > 0 else 0
    avg_mask_iou = total_mask_iou / num_matches if num_matches > 0 else 0

    print("========================================")
    print("E2E Benchmark Results:")
    print(f"Evaluated Images: {num_eval}")
    print(f"Average Inference Time: {avg_time:.4f} s/image ({fps:.2f} FPS)")
    print(f"Average Bounding Box IoU: {avg_box_iou:.4f}")
    print(f"Average Mask IoU: {avg_mask_iou:.4f}")
    print("========================================")

if __name__ == "__main__":
    evaluate()
