import pytest
import torch
from dentex_seg.model import get_instance_segmentation_model

def test_model_initialization():
    """Test model structure initialization without weights loading"""
    num_classes = 53
    model = get_instance_segmentation_model(num_classes)
    assert model is not None
    assert model.roi_heads.box_predictor.cls_score.out_features == num_classes
    assert model.roi_heads.mask_predictor.mask_fcn_logits.out_channels == num_classes
