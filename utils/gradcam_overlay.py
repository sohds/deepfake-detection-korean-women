import torch
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def explain(inference_image, model, device):
    target_layers = [layer for name, layer in model.named_modules() if isinstance(layer, torch.nn.Conv2d)]
    with torch.no_grad():
        inference_image = inference_image.to(device)
        logits = model(inference_image)
        predicted_label = torch.argmax(logits, dim=1).item()
    targets = [ClassifierOutputTarget(predicted_label)]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    result = cam(input_tensor=inference_image, targets=targets)
    salience_map = result[0, :]
    return salience_map

