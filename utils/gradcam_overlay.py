import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def explain(inference_image, model, device):
    target_layers = [layer for _, layer in model.named_modules() if isinstance(layer, torch.nn.Conv2d)]
    with torch.no_grad():
        inference_image = inference_image.to(device)
        logits = model(inference_image)
        predicted_label = torch.argmax(logits, dim=1).item()
    targets = [ClassifierOutputTarget(predicted_label)]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    result = cam(input_tensor=inference_image, targets=targets)
    salience_map = result[0, :]
    return salience_map

def apply_cam_overlay(inference_image, visualize_image, model, device):
    salience_map = explain(inference_image, model, device)
    salience_map_resized = cv2.resize(salience_map, (visualize_image.shape[1], visualize_image.shape[0]))

    if visualize_image.dtype != np.float32:
        visualize_image = visualize_image.astype(np.float32) / 255.0  

    visualization = show_cam_on_image(visualize_image, salience_map_resized, use_rgb=True)

    return visualization
