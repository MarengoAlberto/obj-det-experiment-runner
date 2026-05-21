import torch
from torch import nn

from .backbone import DetBackbone
from .fpn_pan import FPNPAN
from .attention import CBAM

class YOLOHeads(nn.Module):
    def __init__(self, fpn_channels=64, num_classes=2, n_scales=3, **kwargs):
        super().__init__(**kwargs)
        self.fpn_channels = fpn_channels
        self.num_classes = num_classes
        self.n_scales = n_scales
        self.output_head_nodes = 4+1+self.num_classes

        self.head = nn.ModuleList([self._make_head(self.fpn_channels, self.output_head_nodes) for _ in range(self.n_scales)])

    @staticmethod
    def _make_head(fpn_channels, final_op_channels):
        layers = []

        for _ in range(4):
            layers.append(nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(fpn_channels, final_op_channels, kernel_size=3, stride=1, padding=1))

        return nn.Sequential(*layers)

    def forward(self, feature_maps):

        preds = []

        for idx, feature_map in enumerate(feature_maps):
            # print(f"feature map {idx} shape: {feature_map.shape}")
            pred = self.head[idx](feature_map)
            pred = pred.permute(0, 2, 3, 1).reshape(pred.shape[0], -1, self.output_head_nodes)
            preds.append(pred)

        preds = torch.cat(preds, dim=1)

        return preds

class YOLO(nn.Module):
    def __init__(self,
                 backbone_name: str = "resnet18",
                 train_backbone: bool = True,
                 returned_layers: list[int] = [4],
                 num_classes: int = 2,
                 fpn_channels: int = 64,
                 **kwargs) -> None:

        super().__init__()

        self.use_attention = kwargs.get("use_attention", False)

        self.backbone = DetBackbone(
            name=backbone_name,
            train_backbone=train_backbone,
            returned_layers=returned_layers,)

        self.neck = FPNPAN(channels_out=fpn_channels,
                           backbone_out_channels=self.backbone.num_channels)

        self.head = YOLOHeads(fpn_channels=fpn_channels,
                              num_classes=num_classes,
                              n_scales=len(returned_layers))

        if self.use_attention:
            self.backbone_attention = nn.ModuleList(
                [CBAM(channels=self.backbone.num_channels[idx]) for idx in range(len(returned_layers))]
            )
            self.attention = CBAM(channels=fpn_channels,)

    def forward(self, inputs):
        x = self.backbone(inputs)
        if self.use_attention:
            x = [self.backbone_attention[idx](x[layer]) for idx,layer in enumerate(x)]
            x = self.neck(x)
            x = [self.attention(feature_map) for feature_map in x]
        else:
            x = [x[layer] for layer in x]
            x = self.neck(x)
        x = self.head(x)
        return x
