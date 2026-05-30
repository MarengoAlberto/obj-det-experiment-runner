from . import Detector, DataEncoder, YOLO, YOLODataEncoder

def get_model(cfg):
    if cfg.model.name == 'fpn':
        model = Detector(backbone_name=cfg.model.backbone_name,
                              num_classes=len(cfg.dataset.names),
                              fpn_channels=cfg.model.fpn_channels,
                              num_anchors=cfg.model.num_anchors, )
        data_encoder = DataEncoder(input_size=cfg.model.image_size[:2], classes=cfg.dataset.names)
    elif cfg.model.name == 'yolo':
        box_format = cfg.dataset.metadata.box_format
        model = YOLO(backbone_name=cfg.model.backbone_name,
                     train_backbone=cfg.model.train_backbone,
                     returned_layers=cfg.model.returned_layers,
                     num_classes=cfg.dataset.nc,
                     fpn_channels=cfg.model.fpn_channels,
                     attention_type=cfg.model.attention_type,)
        data_encoder = YOLODataEncoder(input_size=cfg.model.image_size[:2],
                                       classes=cfg.dataset.names,
                                       strides=model.backbone.strides,
                                       top_k_per_level=cfg.model.top_k_per_level,
                                       center_radius=cfg.model.center_radius,
                                       allow_multi_level=cfg.model.allow_multi_level,
                                       debug=cfg.experiment.train.debug,
                                       box_format=box_format,
                                       )
    else:
        raise ValueError(f"Unknown model_type: {cfg.model.name}")
    return model, data_encoder