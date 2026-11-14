"""Shared model pieces: logits wrapper, patch-embedding helper, stock SegFormer loader."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

def get_patch_embeds(sf):
    """Return the 4 patch-embedding modules of a SegformerModel (works across transformers versions)."""
    for holder in (sf, getattr(sf, "encoder", None)):
        if holder is None:
            continue
        if hasattr(holder, "stages"):
            return [s.patch_embeddings for s in holder.stages]
        if hasattr(holder, "patch_embeddings"):
            return list(holder.patch_embeddings)
    raise RuntimeError("Unknown SegFormer layout: " + str([n for n, _ in sf.named_children()]))


class SegModel(nn.Module):
    """Thin wrapper so every model returns plain logits: model(x) -> (B, 2, H/4, W/4). No custom logic."""
    def __init__(self, hf):
        super().__init__()
        self.hf = hf
    def forward(self, x):
        return self.hf(pixel_values=x).logits


def build_stock_segformer(pretrained, num_labels=2):
    """Stock Hugging Face SegFormer with pretrained weights and a two-class head (background / pothole)."""
    return SegformerForSemanticSegmentation.from_pretrained(
        pretrained, num_labels=num_labels,
        id2label={0: "background", 1: "pothole"}, label2id={"background": 0, "pothole": 1},
        ignore_mismatched_sizes=True)


count_params = lambda m: sum(p.numel() for p in m.parameters())
