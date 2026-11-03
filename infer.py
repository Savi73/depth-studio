import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

_processor: Any = None
_model: Any = None


def load_model() -> tuple[Any, Any]:
    global _processor, _model

    if _processor is None or _model is None:
        print(f"Loading {MODEL_ID}...")
        _processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        _model = AutoModelForDepthEstimation.from_pretrained(
            MODEL_ID, device_map="auto"
        )
    return _processor, _model


def estimate_depth(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    processor, model = load_model()
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        predicted_depth = model(**inputs).predicted_depth

    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(image.height, image.width),
        mode="bicubic",
        align_corners=False,
    )
    depth = prediction.squeeze().cpu().numpy()
    normalized_depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    grayscale = (normalized_depth * 255).astype(np.uint8)
    colored = cv2.applyColorMap(grayscale, cv2.COLORMAP_INFERNO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    return Image.fromarray(grayscale), Image.fromarray(colored)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate depth maps from an image.")
    parser.add_argument(
        "image",
        nargs="?",
        default="http://images.cocodataset.org/val2017/000000039769.jpg",
        help="Local image path or image URL",
    )
    args = parser.parse_args()

    if args.image.startswith(("http://", "https://")):
        response = requests.get(args.image, timeout=30)
        response.raise_for_status()
        from io import BytesIO

        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(Path(args.image))

    grayscale, colored = estimate_depth(image)
    grayscale.save("depth_map_grayscale.png")
    colored.save("depth_map_colored.png")
    print("Done! Saved 'depth_map_grayscale.png' and 'depth_map_colored.png'.")


if __name__ == "__main__":
    main()
