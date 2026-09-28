import pytest
import torch

from src.audio import CLIP_SAMPLES
from src.gradcam import grad_cam
from src.models import SERModel, count_params

ARCHS = [("scratch", {}), ("dilated", {"width": 48}), ("resnet18", {"pretrained": False}),
         ("cnn14", {"pretrained": False})]


@pytest.mark.parametrize("arch,kwargs", ARCHS)
def test_waveform_in_logits_out(arch, kwargs):
    model = SERModel(arch, **kwargs).eval()
    with torch.no_grad():
        assert model(torch.randn(2, CLIP_SAMPLES)).shape == (2, 6)


def test_scratch_cnn_matches_the_course_design():
    assert count_params(SERModel("scratch")) == 1_175_718  # "1.2 M parameters"


def test_dilated_receptive_field_is_about_1_3_s():
    torch.manual_seed(0)
    model = SERModel("dilated").eval()
    x = torch.randn(1, 1, 64, 301, requires_grad=True)
    model.net.features(x)[0, :, 0, 150].sum().backward()
    frames = (x.grad.abs().sum(dim=(0, 1, 2)) > 0).nonzero().flatten()
    assert frames.max() - frames.min() + 1 == 129  # 10 ms frames -> 1.29 s


def test_checkpoint_roundtrip(tmp_path):
    model = SERModel("scratch").eval()
    model.frontend.mean.fill_(1.5)  # normalisation stats travel with the checkpoint
    torch.save({"state_dict": model.state_dict()}, tmp_path / "m.pt")
    loaded = SERModel("scratch").eval()
    loaded.load_state_dict(torch.load(tmp_path / "m.pt")["state_dict"])
    x = torch.randn(1, CLIP_SAMPLES)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x))


@pytest.mark.parametrize("arch", ["scratch", "dilated"])
def test_gradcam_map(arch):
    model = SERModel(arch).eval()
    cam, spec, cls = grad_cam(model, torch.randn(1, CLIP_SAMPLES))
    assert cam.shape == spec.shape == (64, 301)
    assert 0 <= cam.min() and cam.max() <= 1 and 0 <= cls < 6
