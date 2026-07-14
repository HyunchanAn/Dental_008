import numpy as np
import torch

def assign_fdi_labels(boxes, scores, image_width, image_height):
    """
    1-Stage 모델이 탐지한 치아 Bounding Box 좌표를 기반으로 
    위치와 간격을 계산하여 FDI 번호를 매핑하는 알고리즘.
    boxes: Tensor [N, 4] (x_min, y_min, x_max, y_max)
    scores: Tensor [N]
    """
    if len(boxes) == 0:
        return torch.zeros(0, dtype=torch.int64, device=boxes.device)
        
    boxes_np = boxes.cpu().numpy()
    centers_x = (boxes_np[:, 0] + boxes_np[:, 2]) / 2.0
    centers_y = (boxes_np[:, 1] + boxes_np[:, 3]) / 2.0
    widths = boxes_np[:, 2] - boxes_np[:, 0]
    
    # 1. 상악(Upper) / 하악(Lower) 분리 (Median Y 사용)
    median_y = np.median(centers_y)
    upper_mask = centers_y < median_y
    lower_mask = centers_y >= median_y
    
    labels = np.zeros(len(boxes), dtype=np.int64)
    
    # 2. 상악/하악별로 번호 부여 함수
    def process_arch(mask, is_upper):
        indices = np.where(mask)[0]
        if len(indices) == 0: return
        
        arch_centers_x = centers_x[indices]
        arch_widths = widths[indices]
        
        # X좌표 기준 오름차순 정렬 (이미지 좌측 -> 우측)
        sort_idx = np.argsort(arch_centers_x)
        sorted_indices = indices[sort_idx]
        sorted_x = arch_centers_x[sort_idx]
        sorted_w = arch_widths[sort_idx]
        
        # 중심선 찾기: 이미지 중앙(image_width / 2)과 가장 가까운 x좌표
        center_line = image_width / 2.0
        
        # 이미지 좌측 = 환자 우측(Q1/Q4), 이미지 우측 = 환자 좌측(Q2/Q3)
        left_group_idx = np.where(sorted_x <= center_line)[0]
        right_group_idx = np.where(sorted_x > center_line)[0]
        
        avg_tooth_w = np.median(sorted_w) if len(sorted_w) > 0 else 50.0
        
        # 왼쪽 그룹 (중앙에서 바깥쪽으로 번호 부여)
        if len(left_group_idx) > 0:
            left_group = sorted_indices[left_group_idx]
            left_x = sorted_x[left_group_idx]
            
            # 중앙에서 가까운 순서로 순회 (역순)
            current_fdi = 11 if is_upper else 41
            prev_x = center_line
            for i in reversed(range(len(left_x))):
                idx = left_group[i]
                curr_x = left_x[i]
                # 간격 확인 (결손 치아 파악)
                gap = abs(prev_x - curr_x)
                
                # 중앙에서 첫번째 치아가 아니면서 간격이 1.5배 이상이면 결손으로 간주
                if gap > avg_tooth_w * 1.5 and current_fdi != (11 if is_upper else 41):
                    # 비어있는 치아 개수만큼 스킵 (단순화: 1개만 비었다고 가정하거나, gap 길이에 비례)
                    skip_count = int(np.round(gap / avg_tooth_w)) - 1
                    skip_count = max(1, skip_count)
                    current_fdi += skip_count
                
                # 유효한 FDI 번호까지만 기록 (사랑니까지)
                if (is_upper and current_fdi > 18) or (not is_upper and current_fdi > 48):
                    break
                    
                labels[idx] = current_fdi
                current_fdi += 1
                prev_x = curr_x
                    
        # 오른쪽 그룹 (중앙에서 바깥쪽으로 번호 부여)
        if len(right_group_idx) > 0:
            right_group = sorted_indices[right_group_idx]
            right_x = sorted_x[right_group_idx]
            
            # 중앙에서 가까운 순서로 순회 (정순)
            current_fdi = 21 if is_upper else 31
            prev_x = center_line
            for i in range(len(right_x)):
                idx = right_group[i]
                curr_x = right_x[i]
                gap = abs(curr_x - prev_x)
                
                if gap > avg_tooth_w * 1.5 and current_fdi != (21 if is_upper else 31):
                    skip_count = int(np.round(gap / avg_tooth_w)) - 1
                    skip_count = max(1, skip_count)
                    current_fdi += skip_count
                    
                if (is_upper and current_fdi > 28) or (not is_upper and current_fdi > 38):
                    break
                    
                labels[idx] = current_fdi
                current_fdi += 1
                prev_x = curr_x

    process_arch(upper_mask, is_upper=True)
    process_arch(lower_mask, is_upper=False)
    
    return torch.tensor(labels, dtype=torch.int64, device=boxes.device)
