import pytest
from dentex_seg.dataset import get_fdi_to_class_id

def test_fdi_mapping():
    """Test mapping between FDI tooth numbers and dataset class IDs"""
    fdi_to_id, id_to_fdi = get_fdi_to_class_id()
    
    # 32 permanent teeth + 20 deciduous teeth = 52 classes
    assert len(fdi_to_id) == 52
    assert len(id_to_fdi) == 52
    
    # Permanent teeth quadrant 1-4
    assert 11 in fdi_to_id
    assert 18 in fdi_to_id
    assert 48 in fdi_to_id
    
    # Deciduous teeth quadrant 5-8
    assert 51 in fdi_to_id
    assert 55 in fdi_to_id
    assert 85 in fdi_to_id
    
    # Check consistency
    for fdi, class_id in fdi_to_id.items():
        assert id_to_fdi[class_id] == fdi
