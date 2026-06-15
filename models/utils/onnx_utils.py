import torch
from pathlib import Path
from typing import Union, Tuple
import openvino as ov
import numpy as np

from ..src import YOLO, Detector

class OpenVINODetector:
    def __init__(self, model_path: str, device: str = "CPU"):
        self.core = ov.Core()
        self.compiled_model = self.core.compile_model(model_path, device_name=device)
        self.input_layer = self.compiled_model.input(0)

    def __call__(self, image_batch):
        image_batch_np = image_batch.numpy().astype(np.float32)
        logits = self.compiled_model({self.input_layer: image_batch_np})
        return torch.from_numpy(logits["x"].astype(np.float32))

def is_cuda_available():
    return torch.cuda.is_available()

def compile_model(model_path: Path | str,
                  model: Union[YOLO, Detector],
                  image_size: Tuple[int, int],
                  batch_size: int = 1,):

    if model_path is None:
        return None, None
    onnx_path = Path(model_path.replace('.pth', '.onnx')).resolve()
    H, W = image_size
    dummy = torch.randn(batch_size, 3, H, W)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["inputs"],
        output_names=["x"],
        opset_version=18,
        do_constant_folding=True,
    )
    compiled_model = OpenVINODetector(onnx_path)
    print(f"Compied model: {onnx_path} for CPU optimization")
    return compiled_model, True
