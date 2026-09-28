"""The CNNs. All take a raw waveform batch (B, T) and return logits (B, 6):
the log-mel front end and SpecAugment are part of the model, so a checkpoint
contains everything needed for inference (including normalisation stats)."""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchvision

from src.audio import LogMel

PANNS_WEIGHTS = Path(__file__).resolve().parents[1] / "checkpoints" / "pretrained" / "Cnn14_16k.pth"


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
    """4 VGG-style blocks (32-64-128-256 channels by default), global pooling, linear head."""

    def __init__(self, n_classes: int = 6, dropout: float = 0.3, width: int = 32):
        super().__init__()
        w = [width, 2 * width, 4 * width, 8 * width]
        self.features = nn.Sequential(conv_block(1, w[0]), conv_block(w[0], w[1]),
                                      conv_block(w[1], w[2]), conv_block(w[2], w[3]))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * w[3], n_classes))

    def forward(self, x):
        x = self.features(x)
        # mean over frequency, then mean + max over time
        x = x.mean(dim=2)
        x = torch.cat([x.mean(dim=2), x.amax(dim=2)], dim=1)
        return self.head(x)


class DilatedCNN(nn.Module):
    """Time-dilated CNN (design from the course slides). A 3x3 stem, then six
    3x3 convolutions dilated along time only (rates 1-2-4-8-16-32). Frequency is
    halved after every layer (64 -> 1 mel rows) but time is never pooled, so each
    of the 301 frames keeps 10 ms resolution while its receptive field grows to
    3 + 2 * (1 + 2 + ... + 32) = 129 frames, about 1.3 s: a whole phrase."""

    def __init__(self, n_classes: int = 6, dropout: float = 0.3, width: int = 32):
        super().__init__()
        chans = [width, width, 2 * width, 2 * width, 4 * width, 4 * width, 8 * width]
        layers = [nn.Conv2d(1, chans[0], 3, padding=1, bias=False), nn.BatchNorm2d(chans[0]),
                  nn.ReLU(inplace=True), nn.MaxPool2d((2, 1))]
        for i, d in enumerate([1, 2, 4, 8, 16, 32]):
            layers += [nn.Conv2d(chans[i], chans[i + 1], 3, padding=(1, d), dilation=(1, d), bias=False),
                       nn.BatchNorm2d(chans[i + 1]), nn.ReLU(inplace=True)]
            if i < 5:
                layers.append(nn.MaxPool2d((2, 1)))  # frequency only
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * chans[-1], n_classes))

    def forward(self, x):
        x = self.features(x).mean(dim=2)  # (B, C, T)
        x = torch.cat([x.mean(dim=2), x.amax(dim=2)], dim=1)
        return self.head(x)


class ResNet18Transfer(nn.Module):
    """ImageNet-pretrained ResNet18. The 1-channel spectrogram is repeated on
    the 3 input channels; the 1000-class layer is replaced by a 6-class head.
    `upsample` enlarges the 64x301 spectrogram so ResNet's 32x downsampling
    leaves a bigger final feature map."""

    def __init__(self, n_classes: int = 6, dropout: float = 0.3, pretrained: bool = True,
                 freeze_backbone: bool = False, upsample: int = 1):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = torchvision.models.resnet18(weights=weights)
        self.backbone.fc = nn.Identity()
        self.upsample = upsample
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, n_classes))

    def forward(self, x):
        if self.upsample > 1:
            x = F.interpolate(x, scale_factor=self.upsample, mode="bilinear", align_corners=False)
        return self.head(self.backbone(x.expand(-1, 3, -1, -1)))


class PannsConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.bn1, self.bn2 = nn.BatchNorm2d(c_out), nn.BatchNorm2d(c_out)

    def forward(self, x, pool):
        x = F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))
        return F.avg_pool2d(x, pool)


class Cnn14Transfer(nn.Module):
    """CNN14 from PANNs (Kong et al. 2020), pretrained on AudioSet (2M clips,
    527 sound classes) at 16 kHz. Layer names match the original checkpoint
    so its weights load directly; the AudioSet output layer is replaced by a
    6-class head."""

    def __init__(self, n_classes: int = 6, dropout: float = 0.5, pretrained: bool = True,
                 freeze_backbone: bool = False):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(64)
        chans = [1, 64, 128, 256, 512, 1024, 2048]
        for i in range(6):
            setattr(self, f"conv_block{i + 1}", PannsConvBlock(chans[i], chans[i + 1]))
        self.fc1 = nn.Linear(2048, 2048)
        self.dropout = dropout
        if pretrained:
            state = torch.load(PANNS_WEIGHTS, map_location="cpu", weights_only=False)["model"]
            state = {k: v for k, v in state.items() if not k.startswith(("spectrogram_", "logmel_", "fc_audioset"))}
            self.load_state_dict(state)
        if freeze_backbone:
            for p in self.parameters():
                p.requires_grad = False
        self.head = nn.Linear(2048, n_classes)

    def backbone_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("head") and p.requires_grad]

    def forward(self, x):
        # (B, 1, mel, T) -> PANNs layout (B, 1, T, mel); bn0 normalises each mel bin
        x = self.bn0(x.transpose(2, 3).transpose(1, 3)).transpose(1, 3)
        for i in range(1, 7):
            x = getattr(self, f"conv_block{i}")(x, (2, 2) if i < 6 else (1, 1))
            x = F.dropout(x, 0.2, self.training)
        x = x.mean(dim=3)
        x = x.amax(dim=2) + x.mean(dim=2)
        x = F.dropout(x, self.dropout, self.training)
        x = F.relu(self.fc1(x))
        return self.head(F.dropout(x, self.dropout, self.training))


ARCHS = {"scratch": ScratchCNN, "dilated": DilatedCNN, "resnet18": ResNet18Transfer, "cnn14": Cnn14Transfer}


class SERModel(nn.Module):
    def __init__(self, arch: str, n_classes: int = 6, **kwargs):
        super().__init__()
        self.arch = arch
        self.frontend = LogMel(panns=arch == "cnn14")
        self.augment = SpecAugment()
        self.net = ARCHS[arch](n_classes, **kwargs)

    def spectrogram(self, wav):
        with torch.autocast(device_type=wav.device.type, enabled=False):
            return self.frontend(wav.float())

    def forward(self, wav):
        return self.net(self.augment(self.spectrogram(wav)))

    def param_groups(self, lr: float, backbone_lr_mult: float = 0.1):
        """Pretrained weights get a smaller learning rate than the new head."""
        if self.arch == "resnet18":
            backbone = [p for p in self.net.backbone.parameters() if p.requires_grad]
        elif self.arch == "cnn14":
            backbone = self.net.backbone_parameters()
        else:
            return [{"params": self.net.parameters(), "lr": lr}]
        groups = [{"params": self.net.head.parameters(), "lr": lr}]
        return groups + ([{"params": backbone, "lr": lr * backbone_lr_mult}] if backbone else [])


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
