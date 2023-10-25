import numpy as np

NUM_STIM = 79
MAX_FRAMES = 217

RIDGECV_ALPHA_CV = np.sort(
    np.concatenate(
        [
            np.geomspace(1e-9, 1e5, num=15, endpoint=True),
            5 * np.geomspace(1e-4, 1e4, num=9, endpoint=True),
        ]
    )
)

PHYSION_SCENARIOS = [
    "Dominoes",
    "Support",
    "Collide",
    "Contain",
    "Drop",
    "Link",
    "Roll",
    "Drape",
]
PHYSION_TYPES = ["dynamics_training", "readout_training", "testing"]
PHYSION_DEFAULT_SUBSAMPLE_FACTOR = 9

PF_ENCODERS = [
    "VGG16",
    "ResNet50",
    "DEIT",
    "DINO",
    "DINOv2",
    "CLIP",
    "VIP",
    "VC1",
    "R3M",
]
