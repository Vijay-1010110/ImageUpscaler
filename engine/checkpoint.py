import torch
import os


def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }, path)

import re
def load_latest_checkpoint(model, optimizer, checkpoint_dir):
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

    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    print(f"Resumed from {latest}")
    return checkpoint["epoch"] + 1
