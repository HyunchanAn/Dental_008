import os
import json

def check_gt():
    json_path = os.path.expanduser('~/.cache/dentex_dataset/validation_data.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    img_to_anns = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        img_to_anns[img_id] = img_to_anns.get(img_id, 0) + 1
        
    ann_counts = list(img_to_anns.values())
    if not ann_counts:
        print("No annotations found.")
        return
        
    avg = sum(ann_counts) / len(data['images'])
    print(f"Total images in val JSON: {len(data['images'])}")
    print(f"Total annotations in val JSON: {len(data['annotations'])}")
    print(f"Average annotations per image: {avg:.2f}")
    
    # Check max and min
    print(f"Max annotations in an image: {max(ann_counts)}")
    print(f"Min annotations in an image: {min(ann_counts)}")
    
    # Are there images with 0 annotations?
    zero_count = len(data['images']) - len(img_to_anns)
    print(f"Images with 0 annotations: {zero_count}")

if __name__ == "__main__":
    check_gt()
