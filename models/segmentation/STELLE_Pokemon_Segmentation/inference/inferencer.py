import os
import torch
import onnxruntime as ort
import numpy as np
from models.segmentation.STELLE_Pokemon_Segmentation.model.STELLE_model import STELLE_Seg
from models.segmentation.STELLE_Pokemon_Segmentation.utils.prepocessing import preprocess_image
from models.segmentation.STELLE_Pokemon_Segmentation.utils.visualization import apply_overlay

class STELLEInferencer:
    def __init__(self, torch_model_path="models/segmentation/STELLE_Pokemon_Segmentation/weights/STELLE_Seg.pth", onnx_model_path="models/segmentation/STELLE_Pokemon_Segmentation/weights/STELLE_Seg.onnx"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_onnx = self.device == 'cpu'
        print(f"[Vision Model] Usando el dispositivo {self.device}")
        if self.use_onnx:
            if not os.path.isfile(onnx_model_path):
                raise FileNotFoundError("[Vision Model] Modelo ONNX no encontrado.")
            self.session = ort.InferenceSession(onnx_model_path)
        else:
            self.model = STELLE_Seg().to(self.device)
            self.model.load_state_dict(torch.load(torch_model_path, map_location=self.device))
            self.model.eval()

    def predict(self, np_image):
        tensor = preprocess_image(np_image)
        tensor = tensor.unsqueeze(0)

        if self.use_onnx:
            ort_inputs = {self.session.get_inputs()[0].name: tensor.numpy()}
            ort_outs = self.session.run(None, ort_inputs)
            output = torch.from_numpy(ort_outs[0])
        else:
            with torch.no_grad():
                output = self.model(tensor.to(self.device)).cpu()

        pred = torch.argmax(output, dim=1).squeeze(0).numpy().astype(np.uint8)
        return pred
    
    def predict_with_overlay(self,np_image):
        tensor = preprocess_image(np_image)
        tensor = tensor.unsqueeze(0)

        if self.use_onnx:
            ort_inputs = {self.session.get_inputs()[0].name: tensor.numpy()}
            ort_outs = self.session.run(None, ort_inputs)
            output = torch.from_numpy(ort_outs[0])
        else:
            with torch.no_grad():
                output = self.model(tensor.to(self.device)).cpu()

        pred = torch.argmax(output, dim=1).squeeze(0).numpy().astype(np.uint8)
        overlay = apply_overlay(np_image, pred)
        return pred, overlay

