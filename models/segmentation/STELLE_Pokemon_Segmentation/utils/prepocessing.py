import torch
import numpy as np

def preprocess_image(np_image):
    if np_image.ndim == 3 and np_image.shape[0] <= 4:
        tensor = torch.from_numpy(np_image).float()
    elif np_image.ndim == 3 and np_image.shape[2] <= 4:
        tensor = torch.from_numpy(np_image).permute(2, 0, 1).float()
    else:
        raise ValueError("Formato de imagen no válido. Usa [C,H,W] o [H,W,C]")

    if tensor.max() > 1:
        tensor = tensor / 255.0
    return tensor
