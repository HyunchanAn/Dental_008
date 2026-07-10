import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class DentalAgeDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        
        try:
            # Convert to RGB as some images might be grayscale
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Fallback for corrupted images
            img = Image.new('RGB', (224, 224))
            
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(label, dtype=torch.float32)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Gather file paths
    adult_imgs = glob.glob('data/Kaggle_Childrens/Adult tooth segmentation dataset/**/*.png', recursive=True) + \
                 glob.glob('data/Kaggle_Childrens/Adult tooth segmentation dataset/**/*.jpg', recursive=True)
    child_imgs = glob.glob('data/Kaggle_Childrens/Childrens dental caries segmentation dataset/**/*.png', recursive=True) + \
                 glob.glob('data/Kaggle_Childrens/Childrens dental caries segmentation dataset/**/*.jpg', recursive=True)

    print(f"Found {len(adult_imgs)} Adult images (Class 0) and {len(child_imgs)} Child images (Class 1).")

    all_paths = adult_imgs + child_imgs
    # Class 0: Adult, Class 1: Child (Deciduous)
    all_labels = [0] * len(adult_imgs) + [1] * len(child_imgs)

    # 2. Train / Val Split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )

    # 3. Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 4. Datasets and Loaders
    train_dataset = DentalAgeDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = DentalAgeDataset(val_paths, val_labels, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # 5. Model Definition
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Modify fc layer for binary classification (1 output node)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model = model.to(device)

    # 6. Loss and Optimizer (using pos_weight due to high imbalance 4012:386 = ~10.4)
    pos_weight = torch.tensor([len(adult_imgs) / max(1, len(child_imgs))]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    num_epochs = 10
    best_val_acc = 0.0

    print("Starting Training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total

        print(f"Epoch {epoch+1}: Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            os.makedirs('weights', exist_ok=True)
            torch.save(model.state_dict(), 'weights/classifier_best.pth')
            print("--> Saved best model")

    print(f"Training Complete. Best Val Acc: {best_val_acc:.4f}")

if __name__ == '__main__':
    main()
