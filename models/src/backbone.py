import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter


class DetBackboneBase(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        return_layers: dict,
        strides: list[int],
        num_channels: list[int],
    ):
        super().__init__()

        for name, parameter in backbone.named_parameters():
            if (
                not train_backbone
                or (
                    "layer2" not in name
                    and "layer3" not in name
                    and "layer4" not in name
                )
            ):
                print(f"Freezing {name}")
                parameter.requires_grad_(False)

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.strides = strides
        self.num_channels = num_channels

    def forward(self, input_tensor):
        return self.body(input_tensor)


class DetBackbone(DetBackboneBase):
    """General ResNet backbone."""

    def __init__(
        self,
        name: str = "resnet50",
        train_backbone: bool = True,
        returned_layers: list[int] | None = None,
        pretrained: bool = True,
    ):
        if returned_layers is None:
            returned_layers = [4]

        assert name in {
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
        }, f"Unsupported backbone: {name}"

        backbone = getattr(torchvision.models, name)(pretrained=pretrained)

        return_layers = {
            f"layer{layer_idx}": str(i)
            for i, layer_idx in enumerate(returned_layers)
        }

        all_strides = {
            1: 4,
            2: 8,
            3: 16,
            4: 32,
        }

        if name in {"resnet18", "resnet34"}:
            all_num_channels = {
                1: 64,
                2: 128,
                3: 256,
                4: 512,
            }
        else:
            all_num_channels = {
                1: 256,
                2: 512,
                3: 1024,
                4: 2048,
            }

        strides = [all_strides[i] for i in returned_layers]
        num_channels = [all_num_channels[i] for i in returned_layers]

        super().__init__(
            backbone=backbone,
            train_backbone=train_backbone,
            return_layers=return_layers,
            strides=strides,
            num_channels=num_channels,
        )

class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            # return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [4, 8, 16, 32]
            self.num_channels = [256, 512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, input_tensor):
        out = self.body(input_tensor)
        return out

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool):

        backbone = getattr(torchvision.models, name)(pretrained=True)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
