"""Model-level tests: shapes, determinism, ONNX export parity."""
import numpy as np
import pytest
import torch

from islspeech import FRAME_FEAT, SEQ_LEN
from islspeech.models import FrameMLPClassifier, GRUAttentionClassifier


def test_gru_output_shapes():
    torch.manual_seed(0)
    m = GRUAttentionClassifier(num_classes=10)
    x = torch.randn(4, SEQ_LEN, FRAME_FEAT)
    mask = torch.ones(4, SEQ_LEN)
    logits, attn = m(x, mask)
    assert logits.shape == (4, 10)
    assert attn.shape == (4, SEQ_LEN)
    assert torch.allclose(attn.sum(dim=1), torch.ones(4), atol=1e-5)


def test_gru_mask_respected():
    torch.manual_seed(1)
    m = GRUAttentionClassifier(num_classes=5)
    m.eval()
    x = torch.randn(2, SEQ_LEN, FRAME_FEAT)
    mask = torch.ones(2, SEQ_LEN)
    mask[0, 30:] = 0
    with torch.no_grad():
        _, attn = m(x, mask)
    assert attn[0, 30:].max() < 1e-6  # masked frames get zero attention


def test_frame_mlp_shapes():
    torch.manual_seed(2)
    m = FrameMLPClassifier(num_classes=7)
    x = torch.randn(3, SEQ_LEN, FRAME_FEAT)
    logits, attn = m(x, None)
    assert logits.shape == (3, 7)
    assert attn is None


def test_deterministic_forward():
    torch.manual_seed(3)
    m = GRUAttentionClassifier(num_classes=4)
    m.eval()
    x = torch.randn(2, SEQ_LEN, FRAME_FEAT)
    mask = torch.ones(2, SEQ_LEN)
    with torch.no_grad():
        a, _ = m(x, mask)
        b, _ = m(x, mask)
    assert torch.allclose(a, b)


def test_onnx_export_parity(tmp_path):
    torch.manual_seed(4)
    m = GRUAttentionClassifier(num_classes=6)
    m.eval()
    onnx_path = str(tmp_path / "m.onnx")
    dummy = torch.zeros(1, SEQ_LEN, FRAME_FEAT)
    mask = torch.ones(1, SEQ_LEN)

    class W(torch.nn.Module):
        def __init__(self, mm):
            super().__init__()
            self.mm = mm

        def forward(self, x, mk):
            logits, attn = self.mm(x, mk)
            return torch.softmax(logits, -1), attn

    torch.onnx.export(W(m), (dummy, mask), onnx_path,
                      input_names=["frames", "mask"], output_names=["probs", "attention"],
                      opset_version=17, do_constant_folding=True)
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)
    x = np.random.default_rng(5).normal(size=(1, SEQ_LEN, FRAME_FEAT)).astype(np.float32)
    mk = np.ones((1, SEQ_LEN), dtype=np.float32)
    probs_ort, _ = sess.run(None, {"frames": x, "mask": mk})
    with torch.no_grad():
        logits, _ = m(torch.from_numpy(x), torch.from_numpy(mk))
        probs_t = torch.softmax(logits, -1).numpy()
    assert float(np.max(np.abs(probs_ort - probs_t))) < 1e-4
