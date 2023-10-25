from collections import defaultdict
from mpmodels.core.constants import PF_ENCODERS

LAYERS = dict()
AE_LAYERS = ["encoder", "dynamics", "decoder"]
LAYERS["fitvid_physion_64x64"] = AE_LAYERS
LAYERS["fitvid_physion_aug_64x64_2"] = AE_LAYERS
LAYERS["fitvid_k700_64x64"] = AE_LAYERS
LAYERS["fitvid_k700_aug_64x64"] = AE_LAYERS
LAYERS["fitvid_k700_128x128"] = AE_LAYERS
LAYERS["fitvid_k700_224x224"] = AE_LAYERS
LAYERS["fitvid_bridge_64x64"] = AE_LAYERS
LAYERS["fitvid_bridge_aug_64x64"] = AE_LAYERS
LAYERS["fitvid_ctxt7_physion_64x64"] = AE_LAYERS
LAYERS["fitvid_ctxt7_physion_aug_frames_64x64"] = AE_LAYERS

LAYERS["svg_physion_64x64"] = AE_LAYERS
LAYERS["svg_physion_128x128"] = AE_LAYERS

ENCODER_DYNAMICS_LAYERS = ["encoder", "dynamics"]
for encoder in PF_ENCODERS:
    LAYERS[f"pf{encoder}_LSTM_k700"] = ENCODER_DYNAMICS_LAYERS
    LAYERS[f"pf{encoder}_CTRNN_k700"] = ENCODER_DYNAMICS_LAYERS
    LAYERS[f"pf{encoder}_LSTM_physion"] = ENCODER_DYNAMICS_LAYERS
    LAYERS[f"pf{encoder}_CTRNN_physion"] = ENCODER_DYNAMICS_LAYERS
    LAYERS[f"pf{encoder}_ID"] = ENCODER_DYNAMICS_LAYERS

LAYERS["small_CSWM_physion"] = ENCODER_DYNAMICS_LAYERS
LAYERS["medium_CSWM_physion"] = ENCODER_DYNAMICS_LAYERS
LAYERS["large_CSWM_physion"] = ENCODER_DYNAMICS_LAYERS
