import os
import torch
import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

# PyTorch 2.6 fix: force weights_only=False for all model loads
torch.serialization.add_safe_globals([()])
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
        
    def predict(self, image):
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        results = self.model(image)
        return results
