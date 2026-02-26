import argparse
import torch
import yaml
from PIL import Image
import torchvision.transforms.functional as TF

from models.model_factory import build_model
from utils.device import get_device


def upscale(input_path, output_path, checkpoint_path, config_path="configs/config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = get_device()

    model = build_model(config).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    img = Image.open(input_path).convert("RGB")
    lr_tensor = TF.to_tensor(img).unsqueeze(0).to(device)

    with torch.no_grad():
        sr_tensor = model(lr_tensor)

    sr_tensor = sr_tensor.squeeze(0).clamp(0, 1).cpu()
    sr_image = TF.to_pil_image(sr_tensor)
    sr_image.save(output_path)

    print(f"Upscaled: {input_path} -> {output_path}")
    print(f"Output size: {sr_image.size[0]}x{sr_image.size[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upscale an image using a trained SR model")
    parser.add_argument("--input", required=True, help="Path to input low-resolution image")
    parser.add_argument("--output", required=True, help="Path to save upscaled image")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")

    args = parser.parse_args()

    upscale(args.input, args.output, args.checkpoint, args.config)
