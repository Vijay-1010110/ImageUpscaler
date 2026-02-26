import os
import io
import uuid
import yaml
import torch
from flask import Flask, request, jsonify, send_from_directory, send_file
from PIL import Image
import torchvision.transforms.functional as TF

from models.model_factory import build_model
from utils.device import get_device

app = Flask(__name__, static_folder="web", static_url_path="")

# Load model once at startup
with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

device = get_device()
model = build_model(config).to(device)

# Load latest checkpoint
checkpoint_dir = "checkpoints"
import re
files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
if files:
    def epoch_num(fname):
        m = re.search(r'(\d+)', fname)
        return int(m.group(1)) if m else -1
    files.sort(key=epoch_num)
    latest = files[0]
    path = os.path.join(checkpoint_dir, latest)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    print(f"Loaded checkpoint: {latest}")
else:
    print("WARNING: No checkpoint found. Model has random weights.")

model.eval()

# Temp folder for results
os.makedirs("web/uploads", exist_ok=True)


@app.route("/")
def index():
    return send_from_directory("web", "index.html")


@app.route("/upscale", methods=["POST"])
def upscale():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB")

    # Save original for comparison
    uid = str(uuid.uuid4())[:8]
    original_path = f"web/uploads/{uid}_original.png"
    img.save(original_path)

    # Run inference
    lr_tensor = TF.to_tensor(img).unsqueeze(0).to(device)

    with torch.no_grad():
        sr_tensor = model(lr_tensor)

    sr_tensor = sr_tensor.squeeze(0).clamp(0, 1).cpu()
    sr_image = TF.to_pil_image(sr_tensor)

    upscaled_path = f"web/uploads/{uid}_upscaled.png"
    sr_image.save(upscaled_path)

    return jsonify({
        "original": f"/uploads/{uid}_original.png",
        "upscaled": f"/uploads/{uid}_upscaled.png",
        "original_size": f"{img.size[0]}x{img.size[1]}",
        "upscaled_size": f"{sr_image.size[0]}x{sr_image.size[1]}",
        "scale": config["scale"]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
