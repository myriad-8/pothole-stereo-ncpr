"""MODEL 3 - PROPOSED: stock SegFormer-B0 + residual NCPR (zero-initialised branch, new params at 10x lr)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.common import SegModel, build_stock_segformer, get_patch_embeds

class NCPRResidual(nn.Module):
    """Keeps the pretrained stride-2 conv untouched and ADDS a contrast-routed branch.
    The branch is zero-initialised, so at step 0 the network is identical to the baseline."""
    def __init__(self, conv, use_contrast=True):
        super().__init__()
        C = conv.in_channels
        self.orig, self.use_contrast = conv, use_contrast
        self.router = nn.Conv2d(4 * C, 4, 1)               # phase-routing scores from contrast
        self.mix = nn.Conv2d(C, conv.out_channels, 1)      # maps the routed features to the stage width
        for m in (self.router, self.mix):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        y = self.orig(x)                                    # pretrained path: (B, Cout, H/2, W/2)
        H, W = x.shape[-2:]
        if H % 2 or W % 2:
            x = F.pad(x, (0, W % 2, 0, H % 2))
        ph = torch.stack([x[:, :, 0::2, 0::2], x[:, :, 0::2, 1::2],
                          x[:, :, 1::2, 0::2], x[:, :, 1::2, 1::2]], dim=1)      # B,4,C,h,w
        sig = (ph - ph.mean(1, keepdim=True)).abs() if self.use_contrast else ph
        w = torch.softmax(self.router(sig.flatten(1, 2)), dim=1)                  # B,4,h,w
        return y + self.mix((ph * w.unsqueeze(2)).sum(1))


class ProposedSegFormer(SegModel):
    """MODEL 3 - PROPOSED: stock SegFormer-B0 + residual NCPR in stages 2-4."""
    def __init__(self, pretrained, use_contrast=True):
        super().__init__(build_stock_segformer(pretrained))
        pe = get_patch_embeds(self.hf.segformer)
        for i in (1, 2, 3):
            pe[i].proj = NCPRResidual(pe[i].proj, use_contrast)

    def param_groups(self, lr, new_mult=10.0):
        """New NCPR parameters (zero-initialised) train at 10x lr; pretrained weights keep lr."""
        new = [p for n, p in self.named_parameters() if ".router." in n or ".mix." in n]
        ids = {id(p) for p in new}
        old = [p for p in self.parameters() if id(p) not in ids]
        return [{"params": old, "lr": lr}, {"params": new, "lr": lr * new_mult}]


def build_proposed(pretrained, use_contrast=True):
    return ProposedSegFormer(pretrained, use_contrast)
