from torch import nn


class CifarCNN(nn.Module):
    def __init__(self, channels=(64, 128, 256, 384), depths=(2, 2, 3, 3),
                 head_channels=128, dropout=0.3, num_classes=10):
        super().__init__()
        if len(channels) != 4 or len(depths) != 4:
            raise ValueError("CifarCNN requires exactly four stages")
        self.stages = nn.ModuleList()
        in_channels = 3
        for stage, (out_channels, depth) in enumerate(zip(channels, depths)):
            layers = []
            for _ in range(depth):
                layers.extend([
                    nn.Conv2d(in_channels, out_channels, 3, stride=1,
                              padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ])
                in_channels = out_channels
            if stage < 3:
                layers.append(nn.MaxPool2d(2))
            self.stages.append(nn.Sequential(*layers))
        self.head = nn.Sequential(
            nn.Conv2d(channels[-1], head_channels, 1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(head_channels, num_classes),
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        # [B,3,32,32] -> [B,64,16,16] -> [B,128,8,8]
        # -> [B,256,4,4] -> [B,384,4,4]
        for stage in self.stages:
            x = stage(x)
        # -> [B,128,4,4] -> [B,128,1,1] -> [B,128] -> [B,10]
        return self.head(x)
