from torch.utils.tensorboard import SummaryWriter
import os
import datetime


def create_writer():
    log_dir = os.path.join(
        "runs",
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    return SummaryWriter(log_dir=log_dir)
