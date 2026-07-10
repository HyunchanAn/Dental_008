import torch
import os
from dentex_seg.dataset import DENTEXDataset
from dentex_seg.model import get_instance_segmentation_model
from dentex_seg.evaluate import determine_dentition_type, visualize_and_save

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dataset = DENTEXDataset(split='train')

num_classes = 53
model = get_instance_segmentation_model(num_classes)
weight_path = "C:/Users/chema/Github/Dental_008/weights/pretrained/weights/mask_rcnn_dentex_epoch_30.pth"
if not os.path.exists(weight_path):
    weight_path = "C:/Users/chema/Github/Dental_008/weights/pretrained/weights/mask_rcnn_dentex_epoch_5.pth"
model.load_state_dict(torch.load(weight_path, map_location=device))
model.to(device)
model.eval()

found_mixed = False
found_deciduous = False

with torch.no_grad():
    for idx in range(len(dataset)):
        if found_mixed and found_deciduous:
            break
            
        img, target = dataset[idx]
        gt_labels = target['labels']
        dent_type = determine_dentition_type(gt_labels, dataset.id_to_fdi)
        
        if dent_type == "mixed" and not found_mixed:
            img_tensor = img.to(device).unsqueeze(0)
            output = model(img_tensor)[0]
            
            pred_boxes = output['boxes']
            pred_masks = output['masks'][:, 0]
            pred_labels = output['labels']
            pred_scores = output['scores']
            
            keep = pred_scores > 0.5
            visualize_and_save(img, pred_boxes[keep], pred_masks[keep], pred_labels[keep], dataset.id_to_fdi, "C:/Users/chema/Github/Dental_Panoramic_Reader/reports_archive/images/eval_mixed.jpg")
            found_mixed = True
            print("Found and saved mixed dentition case.")
            
        elif dent_type == "deciduous" and not found_deciduous:
            img_tensor = img.to(device).unsqueeze(0)
            output = model(img_tensor)[0]
            
            pred_boxes = output['boxes']
            pred_masks = output['masks'][:, 0]
            pred_labels = output['labels']
            pred_scores = output['scores']
            
            keep = pred_scores > 0.5
            visualize_and_save(img, pred_boxes[keep], pred_masks[keep], pred_labels[keep], dataset.id_to_fdi, "C:/Users/chema/Github/Dental_Panoramic_Reader/reports_archive/images/eval_deciduous.jpg")
            found_deciduous = True
            print("Found and saved deciduous dentition case.")

print("Visualization generation complete.")
