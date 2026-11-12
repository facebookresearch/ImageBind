import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import torch

ROOT = Path(__file__).resolve().parents[1]


def _fake_fbank(waveform, num_mel_bins=128, **kwargs):
    n_frames = 0 if waveform.shape[-1] == 0 else max(1, waveform.shape[-1] // 16)
    return torch.ones(n_frames, num_mel_bins)


def _load_data_module():
    if "imagebind_data_under_test" in sys.modules:
        return sys.modules["imagebind_data_under_test"]

    def put_module(name, module=None):
        module = module or types.ModuleType(name)
        sys.modules[name] = module
        return module

    torchaudio = put_module("torchaudio")
    compliance = put_module("torchaudio.compliance")
    kaldi = put_module("torchaudio.compliance.kaldi")
    torchaudio.compliance = compliance
    compliance.kaldi = kaldi
    kaldi.fbank = _fake_fbank
    put_module("torchaudio.functional")

    for name in [
        "pytorchvideo",
        "pytorchvideo.transforms",
        "pytorchvideo.data",
        "pytorchvideo.data.clip_sampling",
        "pytorchvideo.data.encoded_video",
        "torchvision",
        "torchvision.transforms",
        "PIL",
        "PIL.Image",
        "pkg_resources",
        "imagebind",
        "imagebind.models",
        "imagebind.models.multimodal_preprocessors",
    ]:
        sys.modules.setdefault(name, MagicMock())

    spec = importlib.util.spec_from_file_location(
        "imagebind_data_under_test", ROOT / "imagebind" / "data.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["imagebind_data_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_waveform2melspec_does_not_mutate_input():
    data = _load_data_module()
    waveform = torch.tensor([[1.0, 3.0, 5.0, 7.0]])
    original = waveform.clone()
    data.waveform2melspec(waveform[:, :2], sample_rate=16000, num_mel_bins=8, target_length=4)
    assert torch.equal(waveform, original)


def test_waveform2melspec_empty_clip_is_padded_not_divzero():
    data = _load_data_module()
    empty = torch.zeros(1, 0)
    fbank = data.waveform2melspec(empty, sample_rate=16000, num_mel_bins=8, target_length=4)
    assert torch.isfinite(fbank).all()
    assert fbank.shape[-1] == 4
