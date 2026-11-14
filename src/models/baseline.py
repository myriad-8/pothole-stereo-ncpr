"""MODEL 1 - BASELINE: stock SegFormer-B0."""
from src.models.common import SegModel, build_stock_segformer

def build_baseline(pretrained):
    """MODEL 1 - BASELINE: the stock SegFormer-B0 (RGB input, pretrained weights). Nothing is changed."""
    return SegModel(build_stock_segformer(pretrained))
