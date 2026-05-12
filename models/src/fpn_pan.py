from torch import nn
import torch.nn.functional as F

from models.src.fpn import Lateral_Connection

class FPNPAN(nn.Module):

    def __init__(self, channels_out=64, backbone_out_channels = [256, 512, 1024, 2048], **kwargs):

        super().__init__(**kwargs)

        self.fpn_layers_lat_conns = []
        self.fpn_layers_3x3 = []
        self.pan_layers_lat_conns = []
        self.pan_layers_3x3 = []

        start = len(backbone_out_channels)-1
        for i in range(start, -1, -1):
            if i != start:
                self.fpn_layers_lat_conns.append(Lateral_Connection(backbone_out_channels[i], channels_out))
                self.fpn_layers_3x3.append(nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1))
                self.pan_layers_lat_conns.append(Lateral_Connection(channels_out, channels_out, kernel_size=3, padding=1))
            else:
                self.fpn_layer_1x1 = nn.Conv2d(backbone_out_channels[i], channels_out, kernel_size=1, stride=1, padding=0)
                self.pan_last_layer_3x3 = nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1)

            self.pan_layers_3x3.append(nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1))

        self.fpn_layers_lat_conns = nn.ModuleList(self.fpn_layers_lat_conns)
        self.fpn_layers_3x3 = nn.ModuleList(self.fpn_layers_3x3)
        self.pan_layers_lat_conns = nn.ModuleList(self.pan_layers_lat_conns)
        self.pan_layers_3x3 = nn.ModuleList(self.pan_layers_3x3)


    def forward(self, inputs):

        # FPN
        fpn_outputs = []
        latest = self.fpn_layer_1x1(inputs[-1])
        fpn_outputs.append(latest)

        start = len(inputs) - 2
        for idx in range(start, -1, -1):
            latest = self.fpn_layers_lat_conns[start-idx]((latest, inputs[idx]))
            additional_3x3 = F.relu(self.fpn_layers_3x3[start-idx](latest), inplace=True)
            fpn_outputs.append(additional_3x3)

        # PAN
        outputs = []
        pan = F.relu(self.pan_last_layer_3x3(fpn_outputs[-1]), inplace=True)
        outputs.append(pan)
        for idx in range(len(fpn_outputs)-1, 0, -1):
            pan = self.pan_layers_lat_conns[idx-1]((fpn_outputs[idx], fpn_outputs[idx-1]))
            latest = F.relu(self.pan_layers_3x3[idx](pan), inplace=True)
            outputs.append(latest)

        return outputs
