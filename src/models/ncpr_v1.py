"""MODEL 2 - NCPR v1: replaces the stride-2 patch-embedding convs of stages 2-4."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.common import SegModel, build_stock_segformer, get_patch_embeds

class NCPRDownsample(nn.Module):
    """NCPR v1: REPLACES a stride-2 patch-embedding conv by
    2x2 phases -> contrast |phase - mean| -> 1x1 routing conv -> softmax -> weighted sum (H/2, W/2) -> the same conv at stride 1.
    use_contrast=False feeds raw phases to the router instead."""
    def __init__(self, conv, use_contrast=True):
        super().__init__()
        C = conv.in_channels
        self.use_contrast = use_contrast
        self.router = nn.Conv2d(4 * C, 4, 1)
        nn.init.zeros_(self.router.weight); nn.init.zeros_(self.router.bias)    # starts as a plain average of the 4 phases
        self.conv = nn.Conv2d(C, conv.out_channels, conv.kernel_size, stride=1,
                              padding=conv.padding, bias=conv.bias is not None)
        self.conv.load_state_dict(conv.state_dict())

    def forward(self, x):
        H, W = x.shape[-2:]
        if H % 2 or W % 2:
            x = F.pad(x, (0, W % 2, 0, H % 2))
        ph = torch.stack([x[:, :, 0::2, 0::2], x[:, :, 0::2, 1::2],
                          x[:, :, 1::2, 0::2], x[:, :, 1::2, 1::2]], dim=1)          # B,4,C,h,w
        sig = (ph - ph.mean(1, keepdim=True)).abs() if self.use_contrast else ph
        w = torch.softmax(self.router(sig.flatten(1, 2)), dim=1)                      # B,4,h,w
        return self.conv((ph * w.unsqueeze(2)).sum(1))


def build_ncpr_v1(pretrained, use_contrast=True):
    """MODEL 2 - NCPR v1: stock SegFormer-B0 with the stage 2-4 patch-embedding convs replaced by NCPRDownsample."""
    m = SegModel(build_stock_segformer(pretrained))
    pe = get_patch_embeds(m.hf.segformer)
    for i in (1, 2, 3):          # stage 1 is a 7x7 stride-4 stem and is left unchanged
        pe[i].proj = NCPRDownsample(pe[i].proj, use_contrast)
    return m
