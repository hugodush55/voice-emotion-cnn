"""The two CNNs. Both take a raw waveform batch (B, T) and return logits (B, 6):
the log-mel front end and SpecAugment are part of the model, so a checkpoint
contains everything needed for inference (including normalisation stats)."""
import torch
import torch.nn as nn
import torchaudio
import torchvision

from src.audio import LogMel


class SpecAugment(nn.Module):
    """Random frequency/time masks, active only in train mode."""

    def __init__(self, freq_mask: int = 10, time_mask: int = 40, n_masks: int = 2):
        super().__init__()
        self.masks = nn.Sequential(*[m for _ in range(n_masks) for m in (
            torchaudio.transforms.FrequencyMasking(freq_mask, iid_masks=True),
            torchaudio.transforms.TimeMasking(time_mask, iid_masks=True),
        )])

    def forward(self, x):
        return self.masks(x) if self.training else x


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class ScratchCNN(nn.Module):
    """4 VGG-style blocks (32-64-128-256 channels), global pooling, linear head."""

    def __init__(self, n_classes: int = 6, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(conv_block(1, 32), conv_block(32, 64),
                                      conv_block(64, 128), conv_block(128, 256))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * 256, n_classes))

    def forward(self, x):
        x = self.features(x)
        # mean over frequency, then mean + max over time
        x = x.mean(dim=2)
        x = torch.cat([x.mean(dim=2), x.amax(dim=2)], dim=1)
        return self.head(x)


class ResNet18Transfer(nn.Module):
    """ImageNet-pretrained ResNet18. The 1-channel spectrogram is repeated on
    the 3 input channels; the 1000-class layer is replaced by a 6-class head."""

    def __init__(self, n_classes: int = 6, dropout: float = 0.3, pretrained: bool = True,
                 freeze_backbone: bool = False):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = torchvision.models.resnet18(weights=weights)
        self.backbone.fc = nn.Identity()
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, n_classes))

    def forward(self, x):
        return self.head(self.backbone(x.expand(-1, 3, -1, -1)))


class SERModel(nn.Module):
    def __init__(self, arch: str, n_classes: int = 6, **kwargs):
        super().__init__()
        self.arch = arch
        self.frontend = LogMel()
        self.augment = SpecAugment()
        self.net = {"scratch": ScratchCNN, "resnet18": ResNet18Transfer}[arch](n_classes, **kwargs)

    def spectrogram(self, wav):
        with torch.autocast(device_type=wav.device.type, enabled=False):
            return self.frontend(wav.float())

    def forward(self, wav):
        return self.net(self.augment(self.spectrogram(wav)))

    def param_groups(self, lr: float, backbone_lr_mult: float = 0.1):
        """Pretrained weights get a smaller learning rate than the new head."""
        if self.arch == "resnet18":
            backbone = [p for p in self.net.backbone.parameters() if p.requires_grad]
            return [{"params": backbone, "lr": lr * backbone_lr_mult},
                    {"params": self.net.head.parameters(), "lr": lr}]
        return [{"params": self.net.parameters(), "lr": lr}]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
