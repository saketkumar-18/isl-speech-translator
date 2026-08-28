"""Unit tests for the feature spec (Python side)."""
import numpy as np
import pytest

from islspeech import FRAME_FEAT, HAND_FEAT, SEQ_LEN
from islspeech.features import (frame_vector, normalize_hand, order_hands,
                                resample_sequence, sequence_from_hands)


def make_hand(seed=0, offset=(0.0, 0.0, 0.0), scale=1.0):
    rng = np.random.default_rng(seed)
    lm = rng.normal(0.5, 0.1, size=(21, 3)) * scale + np.array(offset)
    return lm


def test_normalize_hand_translation_invariant():
    lm = make_hand(seed=1)
    a = normalize_hand(lm)
    b = normalize_hand(lm + np.array([0.37, -0.91, 0.22]))
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_normalize_hand_scale_invariant():
    lm = make_hand(seed=2)
    a = normalize_hand(lm)
    b = normalize_hand((lm - lm[0]) * 2.7 + lm[0])
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_normalize_hand_unit_bound():
    lm = make_hand(seed=3)
    a = normalize_hand(lm)
    assert np.max(np.linalg.norm(a, axis=1)) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(a[0], 0.0, atol=1e-12)


def test_normalize_hand_degenerate():
    lm = np.zeros((21, 3))
    a = normalize_hand(lm)
    np.testing.assert_allclose(a, 0.0)


def test_order_hands_single_left():
    lm = make_hand(seed=4)
    slots = order_hands([{"landmarks": lm, "handedness": "Left"}])
    assert slots[0][0] == 0


def test_order_hands_single_right():
    lm = make_hand(seed=5)
    slots = order_hands([{"landmarks": lm, "handedness": "Right"}])
    assert slots[0][0] == 1


def test_order_hands_labels_win():
    l_left = make_hand(seed=6, offset=(0.8, 0, 0))   # left label but larger x
    l_right = make_hand(seed=7, offset=(0.1, 0, 0))
    slots = order_hands([
        {"landmarks": l_left, "handedness": "Left"},
        {"landmarks": l_right, "handedness": "Right"},
    ])
    m = {s: h for s, h in slots}
    assert set(m.keys()) == {0, 1}
    np.testing.assert_allclose(m[0]["landmarks"], l_left)


def test_order_hands_collision_geometric():
    a = make_hand(seed=8, offset=(0.9, 0, 0))
    b = make_hand(seed=9, offset=(0.1, 0, 0))
    slots = order_hands([
        {"landmarks": a, "handedness": "Right"},
        {"landmarks": b, "handedness": "Right"},
    ])
    m = {s: h for s, h in slots}
    np.testing.assert_allclose(m[0]["landmarks"], b)  # smaller wrist x -> slot 0


def test_frame_vector_layout():
    lm = make_hand(seed=10)
    v = frame_vector([{"landmarks": lm, "handedness": "Left"}])
    assert v.shape == (FRAME_FEAT,)
    assert v[2 * HAND_FEAT] == 1.0 and v[2 * HAND_FEAT + 1] == 0.0
    assert np.any(v[:HAND_FEAT] != 0)
    assert np.all(v[HAND_FEAT:2 * HAND_FEAT] == 0)


def test_frame_vector_empty():
    v = frame_vector([])
    assert v.shape == (FRAME_FEAT,)
    assert np.all(v == 0)


def test_resample_deterministic_endpoints():
    frames = [np.full(FRAME_FEAT, float(i), dtype=np.float32) for i in range(10)]
    out = resample_sequence(frames, SEQ_LEN)
    assert out.shape == (SEQ_LEN, FRAME_FEAT)
    np.testing.assert_allclose(out[0], frames[0])
    np.testing.assert_allclose(out[-1], frames[-1])
    out2 = resample_sequence(frames, SEQ_LEN)
    np.testing.assert_array_equal(out, out2)


def test_resample_single_and_empty():
    f = np.ones(FRAME_FEAT, dtype=np.float32)
    out = resample_sequence([f], SEQ_LEN)
    np.testing.assert_allclose(out, 1.0)
    out0 = resample_sequence([], SEQ_LEN)
    np.testing.assert_allclose(out0, 0.0)


def test_sequence_from_hands_shape():
    per_frame = [[{"landmarks": make_hand(seed=i), "handedness": "Left"}] for i in range(30)]
    seq = sequence_from_hands(per_frame)
    assert seq.shape == (SEQ_LEN, FRAME_FEAT)
    assert seq.dtype == np.float32
