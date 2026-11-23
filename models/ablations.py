import torch
from torch import nn
from torch.nn import functional as F

from models.cnn import CifarCNN
from recipes import ADDITIONAL_EXPERIMENTS, COMPLETE_MODEL, experiment_name, uses_recipe_model


STRUCTURAL_COMPARISONS = {
    "multiscale_fusion": (True, False),
    "spatial_attention": (False, True),
    "multiscale_fusion_spatial_attention": (True, True),
    "head_se": (False, False),
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
        multiscale, attention = STRUCTURAL_COMPARISONS[experiment]
        original_head = list(self.head.children())
        mapping = nn.Sequential(*original_head[:3])
        self.fusion = (MultiScaleFusion(self.stages[1][0].out_channels,
                                       original_head[0].out_channels, mapping)
                       if multiscale else None)
        if not multiscale:
            self.channel_mapping = mapping
        if experiment == "head_se":
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


class GlobalMaximum(nn.Module):
    def forward(self, features):
        return features.amax(dim=(-2, -1), keepdim=True)


class AverageMaximum(nn.Module):
    def forward(self, features):
        return (F.adaptive_avg_pool2d(features, 1)
                + features.amax(dim=(-2, -1), keepdim=True)) * 0.5


class ResidualStage(nn.Module):
    def __init__(self, stage, in_channels, out_channels, pooled):
        super().__init__()
        layers = list(stage.children())
        self.pool = layers.pop() if pooled else nn.Identity()
        self.branch = nn.Sequential(*layers[:-1])
        self.shortcut = (nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                       nn.BatchNorm2d(out_channels))
                         if in_channels != out_channels else nn.Identity())
        self.shortcut.apply(CifarCNN._initialize)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, features):
        return self.pool(self.activation(self.branch(features) + self.shortcut(features)))


class RecipeCNN(CifarCNN):
    def __init__(self, structure, **options):
        super().__init__(**options)
        channels = options.get("channels", (64, 128, 256, 384))
        if structure.get("residual"):
            in_channels = 3
            for index, out_channels in enumerate(channels):
                self.stages[index] = ResidualStage(self.stages[index], in_channels,
                                                    out_channels, index < 3)
                in_channels = out_channels
        if structure.get("downsampling") == "avg":
            for stage in self.stages[:3]:
                stage[-1] = nn.AvgPool2d(2, stride=2)
        if structure.get("se") == "stage2":
            self.stages[1].add_module("se", SqueezeExcitation(channels[1], 8))
        head = list(self.head.children())
        if structure.get("global_pool") == "max":
            head[3] = GlobalMaximum()
        elif structure.get("global_pool") == "avgmax":
            head[3] = AverageMaximum()
        attention = []
        if structure.get("se") == "head":
            attention.append(SqueezeExcitation(head[0].out_channels, 8))
        if structure.get("spatial_attention"):
            attention.append(SpatialAttention())
        self.head = nn.Sequential(*head[:3], *attention, *head[3:])


def build_model(config):
    experiment = experiment_name(config)
    if uses_recipe_model(config):
        return RecipeCNN(config["structure"], **config["model"])
    if experiment in STRUCTURAL_COMPARISONS:
        return AblationCNN(experiment, **config["model"])
    if experiment == COMPLETE_MODEL or experiment in ADDITIONAL_EXPERIMENTS:
        return CifarCNN(**config["model"])
    raise ValueError("Unknown experiment: " + str(experiment))
