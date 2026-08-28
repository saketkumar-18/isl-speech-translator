"""isl-speech-translator: ISL video -> text/speech via hand-pose sequences."""

__version__ = "1.0.0"

FEATURE_SPEC_VERSION = "v1"
NUM_LANDMARKS = 21
COORDS = 3
HAND_SLOTS = 2
HAND_FEAT = NUM_LANDMARKS * COORDS  # 63
FRAME_FEAT = HAND_SLOTS * HAND_FEAT + HAND_SLOTS  # 128: [L(63), R(63), pres_L, pres_R]
SEQ_LEN = 48  # frames per sequence
