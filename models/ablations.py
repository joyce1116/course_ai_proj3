import torch
from torch import nn
from torch.nn import functional as F

from models.cnn import CifarCNN


EXPERIMENTS = {
    "E1": (False, False),
    "E2": (True, False),
    "E3": (False, True),
    "E4": (True, True),
    "N1": (False, False),
    "N2": (False, False),
    "N3": (False, False),
}


def projection(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class MultiScaleFusion(nn.Module):
    def __init__(self, middle_channels, head_channels, deep_projection):
        super().__init__()
        self.middle_projection = projection(middle_channels, head_channels)
        self.fuse = projection(2 * head_channels, head_channels)
        self.middle_projection.apply(CifarCNN._initialize)
        self.fuse.apply(CifarCNN._initialize)
        self.deep_projection = deep_projection

    def forward(self, middle, deep):
        middle = self.middle_projection(middle)
        deep = self.deep_projection(deep)
        deep = F.interpolate(deep, size=middle.shape[-2:], mode="bilinear",
                             align_corners=False)
        return self.fuse(torch.cat((middle, deep), dim=1))


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.apply(CifarCNN._initialize)

    def forward(self, features):
        average = features.mean(dim=1, keepdim=True)
        maximum = features.max(dim=1, keepdim=True).values
        weights = torch.sigmoid(self.conv(torch.cat((average, maximum), dim=1)))
        return features * weights


class SqueezeExcitation(nn.Module):
    def __init__(self, channels=128, hidden_channels=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mapping = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels),
            nn.Sigmoid(),
        )
        self.apply(CifarCNN._initialize)

    def forward(self, features):
        weights = self.mapping(self.pool(features).flatten(1))
        return features * weights[:, :, None, None]


class AblationCNN(CifarCNN):
    def __init__(self, experiment, **options):
        super().__init__(**options)
        multiscale, attention = EXPERIMENTS[experiment]
        original_head = list(self.head.children())
        mapping = nn.Sequential(*original_head[:3])
        self.fusion = (MultiScaleFusion(self.stages[1][0].out_channels,
                                       original_head[0].out_channels, mapping)
                       if multiscale else None)
        if not multiscale:
            self.channel_mapping = mapping
        if experiment == "N3":
            self.pre_pool = SqueezeExcitation(original_head[0].out_channels, 8)
        else:
            self.pre_pool = SpatialAttention() if attention else nn.Identity()
        self.head = nn.Sequential(*original_head[3:])

    @property
    def cam_layer(self):
        return self.pre_pool

    def forward(self, inputs):
        middle = None
        for index, stage in enumerate(self.stages):
            inputs = stage(inputs)
            if index == 1:
                middle = inputs
        features = (self.fusion(middle, inputs) if self.fusion is not None
                    else self.channel_mapping(inputs))
        return self.head(self.pre_pool(features))


def build_model(config):
    experiment = config.get("ablation", {}).get("experiment")
    if experiment is None or experiment in ("E1", "N1", "N2"):
        return CifarCNN(**config["model"])
    if experiment not in EXPERIMENTS:
        raise ValueError("Unknown ablation: " + str(experiment))
    return AblationCNN(experiment, **config["model"])
