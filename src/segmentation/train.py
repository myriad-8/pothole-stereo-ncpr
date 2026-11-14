"""Losses, probability collection and the training loop."""
import torch
import torch.nn.functional as F
from src.datasets.pothole600 import IGNORE_INDEX
from src.evaluation.metrics import seg_metrics

def seg_loss(logits, m, kind="ce", class_weights=None):
    """kind: 'ce' | 'ce_weighted' | 'ce_dice'. Void pixels (IGNORE_INDEX) never enter the loss."""
    w = torch.tensor(class_weights, device=logits.device, dtype=torch.float32) if (kind == "ce_weighted" and class_weights) else None
    loss = F.cross_entropy(logits, m, weight=w, ignore_index=IGNORE_INDEX)
    if kind == "ce_dice":
        p = logits.softmax(1)[:, 1]; valid = (m != IGNORE_INDEX).float(); g = (m == 1).float()
        inter = (p * g * valid).sum()
        loss = loss + 1 - (2 * inter + 1) / ((p * valid).sum() + (g * valid).sum() + 1)
    return loss


@torch.no_grad()
def collect_probs(model, loader, device, loss_kind="ce", class_weights=None):
    """Pothole probability maps (N,H,W float16), masks (N,H,W uint8, 255=void), names, mean loss."""
    model.eval(); P, M, names, loss_sum, n = [], [], [], 0.0, 0
    for b in loader:
        x, m = b["rgb"].to(device), b["mask"].to(device)
        lg = F.interpolate(model(x), size=m.shape[-2:], mode="bilinear", align_corners=False)
        loss_sum += seg_loss(lg, m, loss_kind, class_weights).item() * x.size(0); n += x.size(0)
        P.append(lg.softmax(1)[:, 1].half().cpu()); M.append(m.to(torch.uint8).cpu()); names += list(b["name"])
    return torch.cat(P).numpy(), torch.cat(M).numpy(), names, loss_sum / n


def train_model(model, train_loader, val_loader, cfg, device, ckpt_path, tag=""):
    """Fine-tune; keep the best-validation-IoU checkpoint; return model with those weights."""
    opt = torch.optim.AdamW(model.param_groups(cfg["lr"]) if hasattr(model, "param_groups") else model.parameters(),
                            lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    best, best_ep, hist = -1.0, 0, []
    for ep in range(cfg["epochs"]):
        model.train(); tl = n = 0
        for b in train_loader:
            x, m = b["rgb"].to(device), b["mask"].to(device)
            opt.zero_grad(set_to_none=True)
            lg = F.interpolate(model(x), size=m.shape[-2:], mode="bilinear", align_corners=False)
            loss = seg_loss(lg, m, cfg["loss"], cfg.get("class_weights")); loss.backward(); opt.step()
            tl += loss.item() * x.size(0); n += x.size(0)
        P, M, _, vl = collect_probs(model, val_loader, device, cfg["loss"], cfg.get("class_weights"))
        v, _ = seg_metrics(P, M)
        hist.append(dict(epoch=ep + 1, train_loss=tl / n, val_loss=vl, val_iou=v["iou"], val_dice=v["dice"]))
        print(f"{tag} ep{ep+1:02d} | train loss {tl/n:.4f} | val loss {vl:.4f} | val IoU {v['iou']:.4f}")
        if v["iou"] > best:
            best, best_ep = v["iou"], ep + 1; torch.save(model.state_dict(), ckpt_path)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model, hist, best_ep
