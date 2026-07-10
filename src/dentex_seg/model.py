import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_instance_segmentation_model(num_classes):
    """
    Mask R-CNN (ResNet-50-FPN V2) 모델을 초기화하고, 
    유치와 치배를 포함한 지정된 수의 클래스를 예측하도록 
    마스크 헤드와 바운딩 박스 분류기를 교체합니다.
    """
    print("고해상도 백본 기반의 Mask R-CNN (ResNet-50-FPN V2) 모델을 로드합니다...")
    # 사전 학습된 기본 가중치 로드
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT")

    # 바운딩 박스 분류기(Classifier)의 입력 피처 수 가져오기 및 교체
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 마스크 예측기(Mask Predictor)의 입력 피처 채널 가져오기 및 교체
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    print(f"모델 출력이 총 {num_classes}개 클래스(배경 포함)에 맞게 재구성되었습니다.")
    return model
