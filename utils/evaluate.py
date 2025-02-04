import cv2
import numpy as np

def calculate_presence_binary(activation_map, mask, threshold=0.8):
    activation_map_resized = cv2.resize(activation_map, (mask.shape[1], mask.shape[0]))
    presence_binary = int(np.max(activation_map_resized * (mask[:, :, 0] > 0)) >= threshold)
    return presence_binary

def calculate_area_in_mask(activation_map, mask, threshold=0.8):
    activation_map_resized = cv2.resize(activation_map, (mask.shape[1], mask.shape[0]))
    activation_map_thresholded = (activation_map_resized >= threshold).astype(np.float32)
    mask_2d = mask[:, :, 0]
    masked_area = activation_map_thresholded * mask_2d
    return np.sum(masked_area), np.sum(mask_2d)