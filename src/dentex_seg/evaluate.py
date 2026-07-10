import numpy as np
import matplotlib.pyplot as plt

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
            
        if 11 <= fdi <= 48:
            color = np.array([0.0, 0.0, 1.0]) # Blue
            hex_color = 'blue'
        elif 51 <= fdi <= 85:
            color = np.array([0.0, 1.0, 1.0]) # Cyan
            hex_color = 'cyan'
        else:
            color = np.array([1.0, 0.0, 0.0]) # Red
            hex_color = 'red'
        
        ax.text(box[0] + (box[2]-box[0])/2, box[1]-5, str(fdi), color='white', fontsize=10, weight='bold', ha='center', bbox=dict(facecolor=hex_color, alpha=0.5, edgecolor='none', pad=1))
        
        colored_mask = np.zeros_like(img_np)
        for c in range(3):
            colored_mask[:, :, c] = color[c]
        
        mask_bool = mask > 0.5
        img_np[mask_bool] = img_np[mask_bool] * 0.6 + colored_mask[mask_bool] * 0.4

    plt.imshow(img_np)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
