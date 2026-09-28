"""Grad-CAM (Selvaraju et al. 2017) on the log-mel input: which time-frequency
regions pushed the network towards its predicted emotion.

Works for the models with a `features` convolutional trunk (scratch, dilated).
The map is as coarse as the last feature map: 4 x 18 cells for the scratch CNN,
1 x 301 (time only) for the dilated CNN.
"""
import numpy as np
import torch
import torch.nn.functional as F


def _members(model):
    """The models to explain: the model itself, or the members of an Ensemble."""
    return [m for m in getattr(model, "members", [model]) if hasattr(m.net, "features")]


def supports_gradcam(model) -> bool:
    return bool(_members(model))


def grad_cam(model, wav: torch.Tensor, class_idx: int | None = None):
    """wav: (1, T) fixed-length window. Returns (cam in [0, 1] with the shape of
    the spectrogram (n_mels, frames), spectrogram as a numpy array, class index).
    For an Ensemble, the maps of the members are averaged."""
    if class_idx is None:
        with torch.no_grad():
            class_idx = int(model.eval()(wav).argmax(1))
    cams = [_single_grad_cam(m, wav, class_idx) for m in _members(model)]
    cam = np.mean(cams, axis=0)
    spec = model.frontend.log_mel(wav)[0, 0].detach().numpy()
    return cam / (cam.max() + 1e-8), spec, class_idx


def _single_grad_cam(model, wav: torch.Tensor, class_idx: int) -> np.ndarray:
    model.eval()
    store = {}

    def hook(_, __, out):
        out.retain_grad()
        store["act"] = out

    handle = model.net.features.register_forward_hook(hook)
    try:
        with torch.enable_grad():
            spec = model.spectrogram(wav)
            logits = model.net(spec)
            model.zero_grad(set_to_none=True)
            logits[0, class_idx].backward()
    finally:
        handle.remove()
    act, grad = store["act"], store["act"].grad
    weights = grad.mean(dim=(2, 3), keepdim=True)            # importance of each channel
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))  # (1, 1, F', T')
    cam = F.interpolate(cam, size=spec.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    return (cam / (cam.max() + 1e-8)).detach().numpy()
