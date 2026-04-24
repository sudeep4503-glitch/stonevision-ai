import os
import torch
import streamlit as st
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO

# PyTorch 2.6 fix: force weights_only=False for all model loads
torch.serialization.add_safe_globals([])
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

class KidneyStoneDetectionModel:
    def __init__(self, model_path="ks_detection.pt"):
        st.write("Loading kidney stone model...")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} not found in {os.getcwd()}")

        self.model = YOLO(model_path)
        st.success(f"✅ Model {model_path} loaded")

    def predict(self, image):
        # Windows multiprocessing fix
        os.environ['OMP_NUM_THREADS'] = '1'

        # Convert PIL to numpy
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        elif not isinstance(image, np.ndarray):
            raise TypeError(f"Unsupported image type: {type(image)}")

        st.write(f"DEBUG: raw shape={image.shape}, dtype={image.dtype}")

        # Fix 1: Remove batch dimension (1, H, W, C) -> (H, W, C)
        if image.ndim == 4 and image.shape[0] == 1:
            image = np.squeeze(image, axis=0)

        if image.size == 0 or 0 in image.shape:
            raise ValueError("Image is empty or has zero dimension")

        # Fix 2: Force uint8. X-rays are often 16-bit.
        if image.dtype!= np.uint8:
            if image.max() > 255:
                image = (image / image.max() * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # Fix 3: Force 3 channels RGB
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[-1] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Fix 4: Make memory contiguous for OpenCV
        image = np.ascontiguousarray(image)

        st.write(f"DEBUG: final shape={image.shape}, dtype={image.dtype}")

        if image.ndim!= 3 or image.shape[2]!= 3:
            raise ValueError(f"Image still wrong shape after fix: {image.shape}")

        # Run inference
        results = self.model.predict(source=image, verbose=False)
        result = results[0]
        output_image = result.plot()

        label = "No Stone Detected"
        confidence = 0.0

        if result.boxes is not None and len(result.boxes) > 0:
            max_conf = float(result.boxes.conf.cpu().numpy().max())
            if max_conf > 0.6:
                label = "Kidney Stone Detected"
                confidence = max_conf

        return label, confidence, output_image