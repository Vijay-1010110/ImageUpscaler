import torch
import os
import re
from collections import OrderedDict


def save_checkpoint(model, optimizer, epoch, path):
    """Save checkpoint — handles DataParallel by stripping 'module.' prefix."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Strip DataParallel wrapper if present
    state_dict = model.state_dict()
    clean_state = OrderedDict()
    for k, v in state_dict.items():
        clean_state[k.replace("module.", "")] = v

    torch.save({
        "epoch": epoch,
        "model": clean_state,
        "optimizer": optimizer.state_dict()
    }, path)


def load_latest_checkpoint(model, optimizer, checkpoint_dir):
    """Load latest checkpoint — works for both single-GPU and DataParallel models."""
    if not os.path.exists(checkpoint_dir):
        return 0

    files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if not files:
        return 0

    def epoch_num(fname):
        m = re.search(r'(\d+)', fname)
        return int(m.group(1)) if m else -1
    files.sort(key=epoch_num)
    latest = files[-1]
    path = os.path.join(checkpoint_dir, latest)

    checkpoint = torch.load(path, weights_only=False, map_location="cpu")

    # Handle loading into DataParallel or regular model
    state_dict = checkpoint["model"]
    model_is_parallel = hasattr(model, "module")

    clean_state = OrderedDict()
    for k, v in state_dict.items():
        key = k.replace("module.", "")
        if model_is_parallel:
            key = "module." + key
        clean_state[key] = v

    model.load_state_dict(clean_state)
    optimizer.load_state_dict(checkpoint["optimizer"])

    print(f"Resumed from {latest}")
    return checkpoint["epoch"] + 1
