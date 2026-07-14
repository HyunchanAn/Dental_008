import numpy as np

def split_arches(boxes):
    if len(boxes) == 0: return [], []
    centers_y = (boxes[:, 1] + boxes[:, 3]) / 2.0
    
    sorted_y = np.sort(centers_y)
    if len(sorted_y) > 1:
        gaps = np.diff(sorted_y)
        max_gap_idx = np.argmax(gaps)
        split_y = sorted_y[max_gap_idx] + gaps[max_gap_idx] / 2.0
    else:
        split_y = sorted_y[0] + 1.0
        
    upper_indices = np.where(centers_y < split_y)[0].tolist()
    lower_indices = np.where(centers_y >= split_y)[0].tolist()
    return upper_indices, lower_indices

def get_ideal_type(fdi):
    digit = fdi % 10
    if digit in [1, 2, 3]: return 0 # Anterior
    elif digit in [4, 5]: return 1 # Premolar
    elif digit in [6, 7, 8]: return 2 # Molar
    return 0

def get_box_type(w, h):
    aspect_ratio = w / h
    if aspect_ratio > 0.60:
        return 2 # Molar
    elif aspect_ratio < 0.38:
        return 0 # Anterior
    else:
        return 1 # Premolar

def match_cost(box_type, fdi):
    ideal_type = get_ideal_type(fdi)
    if box_type == ideal_type:
        return 0.0
    if abs(box_type - ideal_type) == 1:
        return 3.0
    return 6.0

def dp_align_arch(boxes_np, ideal_sequence):
    n = len(boxes_np)
    m = len(ideal_sequence)
    if n == 0: return []
    
    dp = np.full((n + 1, m + 1), float('inf'))
    dp[0, 0] = 0.0
    
    parent = np.zeros((n + 1, m + 1, 2), dtype=int)
    
    DISCARD_COST = 10.0
    
    for i in range(1, n + 1):
        box = boxes_np[i - 1]
        w = box[2] - box[0]
        h = box[3] - box[1]
        b_type = get_box_type(w, h)
        
        for j in range(1, m + 1):
            fdi = ideal_sequence[j - 1]
            
            # Option 1: Match box i to ideal j
            cost_match = dp[i - 1, j - 1] + match_cost(b_type, fdi)
            
            # Option 2: Box i is a false positive (discard box)
            cost_discard_box = dp[i - 1, j] + DISCARD_COST
            
            # Option 3: Ideal j is a missing tooth (skip ideal)
            if fdi % 10 == 8:
                # Wisdom teeth are frequently missing, so skipping them is very cheap
                cost_skip_ideal = dp[i, j - 1] + 0.1
            else:
                if i > 1:
                    prev_box = boxes_np[i - 2]
                    gap_x = box[0] - prev_box[2]
                    avg_w = (w + (prev_box[2] - prev_box[0])) / 2.0
                    if gap_x > avg_w * 0.6:
                        cost_skip_ideal = dp[i, j - 1] + 0.5 # Missing tooth expected due to gap
                    else:
                        cost_skip_ideal = dp[i, j - 1] + 2.0 # Penalty for skipping when no gap exists
                else:
                    cost_skip_ideal = dp[i, j - 1] + 1.0
                
            costs = [cost_match, cost_discard_box, cost_skip_ideal]
            min_idx = np.argmin(costs)
            dp[i, j] = costs[min_idx]
            
            if min_idx == 0:
                parent[i, j] = [i - 1, j - 1]
            elif min_idx == 1:
                parent[i, j] = [i - 1, j]
            else:
                parent[i, j] = [i, j - 1]
                
    curr_i, curr_j = n, m
    best_cost = float('inf')
    best_j = m
    for j in range(n, m + 1):
        if dp[n, j] < best_cost:
            best_cost = dp[n, j]
            best_j = j
    curr_j = best_j
    
    assignments = []
    while curr_i > 0 and curr_j > 0:
        prev_i, prev_j = parent[curr_i, curr_j]
        if prev_i == curr_i - 1 and prev_j == curr_j - 1:
            assignments.append((curr_i - 1, ideal_sequence[curr_j - 1]))
            curr_i, curr_j = prev_i, prev_j
        elif prev_i == curr_i - 1 and prev_j == curr_j:
            # Box discarded (False positive)
            assignments.append((curr_i - 1, 0))
            curr_i, curr_j = prev_i, prev_j
        else:
            # Ideal skipped (Missing tooth)
            curr_i, curr_j = prev_i, prev_j
            
    while curr_i > 0:
        assignments.append((curr_i - 1, 0))
        curr_i -= 1
        
    assignments.reverse()
    return assignments

def correct_fdi_numbers(boxes, yolo_labels):
    # This function now ignores yolo_labels entirely and does pure geometric DP
    is_tensor = False
    if hasattr(boxes, 'cpu'):
        is_tensor = True
        boxes_np = boxes.cpu().numpy()
        yolo_labels_np = yolo_labels.cpu().numpy()
    else:
        boxes_np = np.array(boxes)
        yolo_labels_np = np.array(yolo_labels)
        
    n = len(boxes_np)
    if n == 0: return yolo_labels
    corrected_labels = np.copy(yolo_labels_np)
    
    upper_idx, lower_idx = split_arches(boxes_np)
    
    # Process Upper Arch
    if len(upper_idx) > 0:
        upper_boxes = boxes_np[upper_idx]
        sort_order = np.argsort(upper_boxes[:, 0])
        sorted_boxes = upper_boxes[sort_order]
        
        ideal_upper = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
        assign_upper = dp_align_arch(sorted_boxes, ideal_upper)
        
        for local_i, fdi in assign_upper:
            global_i = upper_idx[sort_order[local_i]]
            corrected_labels[global_i] = fdi
            
    # Process Lower Arch
    if len(lower_idx) > 0:
        lower_boxes = boxes_np[lower_idx]
        sort_order = np.argsort(lower_boxes[:, 0])
        sorted_boxes = lower_boxes[sort_order]
        
        # Lower sequence sorted by X (left to right)
        ideal_lower = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]
        assign_lower = dp_align_arch(sorted_boxes, ideal_lower)
        
        for local_i, fdi in assign_lower:
            global_i = lower_idx[sort_order[local_i]]
            corrected_labels[global_i] = fdi
            
    if is_tensor:
        import torch
        return torch.tensor(corrected_labels, device=boxes.device, dtype=yolo_labels.dtype)
    return corrected_labels
