from .fpn_trainer import FPNTrainer
from .yolo_trainer import YOLOTrainer

def get_trainer(cfg):
    if cfg.model.name == 'fpn':
        return FPNTrainer
    elif cfg.model.name == 'yolo':
        return YOLOTrainer
    else:
        raise ValueError(f"Unknown model_type: {cfg.model.name}")
