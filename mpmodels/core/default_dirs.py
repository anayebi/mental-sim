import os
import socket

ROOT_DIR = os.path.expanduser("~/")
BASE_DIR = os.path.join(ROOT_DIR, "mental-sim/")

NEURAL_RESP_DIR = ""
NEURAL_STIM_JPEG_DIR = ""
BEHAVIORAL_FIT_RESULTS_DIR = ""
OCP_PHYSION_RESULTS_DIR = os.path.join(
    BEHAVIORAL_FIT_RESULTS_DIR, "ocp_physion_results/"
)
OCP_PHYSION_REGRESSION_RESULTS_DIR = os.path.join(
    OCP_PHYSION_RESULTS_DIR, "regression/"
)
OCP_PHYSION_HUMAN_RESULTS_DIR = os.path.join(OCP_PHYSION_RESULTS_DIR, "humans/")

MODEL_CKPT_DIR = os.path.join(BASE_DIR, "trained_models/")
MODEL_FEATURES_SAVE_DIR = ""

# kinetics 700
KINETICS_BASE_DIR = os.path.join(ROOT_DIR, "kinetics_dataset/")
K700_2020_VIDEO_BASE_DIR = os.path.join(KINETICS_BASE_DIR, "k700-2020/")
# shortest side 480
K700_2020_JPEG_480SS_BASE_DIR = os.path.join(KINETICS_BASE_DIR, "k700-2020_rgb480ss/")
K700_2020_TAR_480SS_BASE_DIR = os.path.join(KINETICS_BASE_DIR, "k700-2020_tar480ss/")
K700_SCRIPT_BASE_DIR = ""
# physion
PHYSION_BASE_DIR = os.path.join(ROOT_DIR, "physion/")
OCP_PHYSION_HUMAN_DATA = os.path.join(ROOT_DIR, "OCP_physion_human_data/csv/")
