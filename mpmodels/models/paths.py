import os

from mpmodels.core.default_dirs import MODEL_CKPT_DIR
from mpmodels.core.constants import PF_ENCODERS
from mpmodels.models.layers import LAYERS
from mpmodels.models.transforms import TRANSFORMS

PATHS = {}

PATHS["fitvid_ctxt7_physion_aug_frames_64x64"] = os.path.join(
    MODEL_CKPT_DIR, "FitVid_physion_64x64.pt"
)
PATHS["svg_physion_128x128"] = os.path.join(MODEL_CKPT_DIR, "SVG_physion_128x128.pt")

PATHS["large_CSWM_physion"] = os.path.join(MODEL_CKPT_DIR, "CSWM_large_physion.pt")

for encoder in PF_ENCODERS:
    # these models have no trainable parameters
    PATHS[f"pf{encoder}_ID"] = None

for encoder in ["R3M", "VC1"]:
    for dynamics in ["LSTM", "CTRNN"]:
        for dataset in ["physion", "k700"]:
            encoder_name = encoder
            if encoder == "VC1":
                encoder_name = "VC-1"
            PATHS[f"pf{encoder}_{dynamics}_{dataset}"] = os.path.join(
                MODEL_CKPT_DIR, f"{encoder_name}+{dynamics}_{dataset}.pt"
            )

for model in PATHS.keys():
    assert model in LAYERS.keys(), f"{model} not in layers.py"
    assert model in TRANSFORMS.keys(), f"{model} not in transforms.py"
