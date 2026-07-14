import torch
from torch.utils.data import DataLoader
from dentex_seg.dataset import DENTEXDataset
from dentex_seg.model import get_instance_segmentation_model
import os

def collate_fn(batch):
    return tuple(zip(*batch))

def train_model():
    print("========================================")
    print("DENTEX 혼합치열기 인스턴스 세그멘테이션 학습 시작")
    print("========================================")
    
    # 디바이스 설정
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"학습 실행 환경: {device}")

    # 데이터셋 불러오기 (Hugging Face 직접 다운로드 처리됨)
    dataset_train = DENTEXDataset(split='train')
    data_loader = DataLoader(dataset_train, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=8, pin_memory=True)

    # 모델 준비 (배경 1개 + 치아 1개 = 총 2개 클래스)
    num_classes = 2
    model = get_instance_segmentation_model(num_classes)
    model.to(device)

    # 파라미터 및 옵티마이저 설정
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.02, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
    
    # AMP 활성화 (RTX 5080 텐서 코어 활용)
    scaler = torch.cuda.amp.GradScaler()

    num_epochs = 30
    os.makedirs('weights', exist_ok=True)

    print("모든 설정이 완료되었습니다. 본격적인 에포크 반복을 시작합니다.")
    
    # 학습 루프
    for epoch in range(num_epochs):
        model.train()
        print(f"\n--- 에포크 {epoch+1}/{num_epochs} 진행 중 ---")
        epoch_loss = 0
        
        for i, (images, targets) in enumerate(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 손실 계산 및 역전파 (AMP 적용)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += losses.item()
            
            # 콘솔 로깅 출력 (10번째 배치마다)
            if (i+1) % 10 == 0:
                print(f"  > [에포크 {epoch+1}] 배치 {i+1} 완료 - 현재 총 손실(Loss): {losses.item():.4f}")

        # 학습률 스케줄러 갱신
        lr_scheduler.step()
        avg_loss = epoch_loss / len(data_loader)
        print(f"[에포크 {epoch+1} 완료] 평균 손실: {avg_loss:.4f}")
        
        # 가중치 저장
        save_path = f"weights/mask_rcnn_dentex_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"모델 가중치 저장 완료: {save_path}")

    print("\n모든 학습이 성공적으로 종료되었습니다.")

if __name__ == "__main__":
    train_model()
