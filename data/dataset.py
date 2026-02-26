import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class SRDataset(Dataset):
    def __init__(self, hr_folder, patch_size=96, scale=2):
        self.files = [os.path.join(hr_folder, f)
                      for f in os.listdir(hr_folder)]
        self.patch_size = patch_size
        self.scale = scale

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")

        w, h = img.size

        if w < self.patch_size or h < self.patch_size:
            raise ValueError("Patch size larger than image size")

        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size)

        hr = TF.crop(img, y, x, self.patch_size, self.patch_size)

        lr = hr.resize(
            (self.patch_size // self.scale,
             self.patch_size // self.scale),
            Image.BICUBIC
        )

        return TF.to_tensor(lr), TF.to_tensor(hr)
