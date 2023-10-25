import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from ptutils.core.default_constants import IMAGENET_MEAN, IMAGENET_STD
from ptutils.model_training.trainer_transforms import TRAINER_TRANSFORMS
from mpmodels.model_training.video_dataset import ImglistToTensor
from mpmodels.core.constants import PF_ENCODERS
from mpmodels.models.custom_transforms import (
    GroupRandomResizedCrop,
    GroupRandAugment,
    GroupResize,
)
from vc_models.models.vit import model_utils

_, _, VC1_VAL_TRANSFORMS, _ = model_utils.load_model(model_utils.VC1_LARGE_NAME)


def apply_totensor(base_transforms, jpeg_input=True, prepend=True):
    if jpeg_input:
        add_transforms = [ImglistToTensor()]
    else:
        add_transforms = [transforms.ToTensor()]

    if prepend:
        final_transforms = add_transforms + base_transforms
    else:
        final_transforms = base_transforms + add_transforms
    return final_transforms


def resize_only(size, jpeg_input=True, prepend=True):
    # returns a list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
    base_transforms = [
        transforms.Resize(size),
    ]
    return apply_totensor(
        base_transforms=base_transforms, jpeg_input=jpeg_input, prepend=prepend
    )


def fitvid_randcropaug_resize(
    size, min_crop_ratio=0.8, num_ops=1, magnitude=5, jpeg_input=True,
):
    base_transforms = [
        transforms.RandomResizedCrop(
            size=size, scale=(min_crop_ratio, 1.0), ratio=(1.0, 1.0)
        ),
        transforms.ConvertImageDtype(
            torch.uint8
        ),  # RandAugment expects uint8s, outputs float32
        transforms.RandAugment(num_ops=num_ops, magnitude=magnitude),
    ]
    return apply_totensor(base_transforms=base_transforms, jpeg_input=jpeg_input)


def fitvid_randcropaug_resize_2(
    size, min_crop_ratio=0.8, num_ops=1, magnitude=5,
):
    # applies PIL RRCrop (as opposed to Pytorch's, to avoid any aliasing) rather than ToTensor() first
    base_transforms = [
        transforms.RandomResizedCrop(
            size=size, scale=(min_crop_ratio, 1.0), ratio=(1.0, 1.0)
        ),
        transforms.RandAugment(num_ops=num_ops, magnitude=magnitude),
        transforms.ToTensor(),
    ]
    return base_transforms


def fitvid_randcropaug_resize_2_frames(
    size, min_crop_ratio=0.8, num_ops=1, magnitude=5,
):
    # applies SAME random crops across frames
    base_transforms = [
        GroupRandomResizedCrop(
            size=size, scale=(min_crop_ratio, 1.0), ratio=(1.0, 1.0)
        ),
        GroupRandAugment(num_ops=num_ops, magnitude=magnitude),
        ImglistToTensor(),
    ]
    return base_transforms


# from: https://colab.research.google.com/github/facebookresearch/deit/blob/colab/notebooks/deit_inference.ipynb#scrollTo=sVwQc7BcxFgi
DEIT_VAL_TRANSFORMS = [
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
]

# from: https://github.com/openai/CLIP/blob/c5478aac7b9e007a2659d36b57ebe148849e542a/clip/clip.py#L75-L86
def _convert_image_to_rgb(image):
    return image.convert("RGB")


CLIP_VAL_TRANSFORMS = [
    transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    _convert_image_to_rgb,
    transforms.ToTensor(),
    transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    ),
]

# from: https://github.com/facebookresearch/r3m/blob/921599e7325dc0f10b6d8ef6e76ba74df6ea4ce0/r3m/example.py#L23-L34
R3M_VAL_TRANSFORMS = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
]

IMNET_VAL_TRANSFORMS = TRAINER_TRANSFORMS["SupervisedImageNetTrainer"]["val"]

RESIZE_64x64_VAL_TRANSFORMS = [
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
]

RESIZE_128x128_VAL_TRANSFORMS = [
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
]

RESIZE_224x224_VAL_TRANSFORMS = [
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
]

RESIZE_64x64_VAL_TRANSFORMS_FRAMES = [
    GroupResize((64, 64)),
    ImglistToTensor(),
]

TRANSFORMS = {
    "fitvid_physion_64x64": {"val": RESIZE_64x64_VAL_TRANSFORMS,},
    "resize_only_64x64": {
        "train": resize_only(size=(64, 64)),
        "val": resize_only(size=(64, 64)),
    },
    "resize_only_128x128": {
        "train": resize_only(size=(128, 128)),
        "val": resize_only(size=(128, 128)),
    },
    "resize_only_224x224": {
        "train": resize_only(size=(224, 224)),
        "val": resize_only(size=(224, 224)),
    },
    "fitvid_aug_64x64": {
        "train": fitvid_randcropaug_resize(size=(64, 64)),
        "val": resize_only(size=(64, 64)),
    },
    "fitvid_nonjpeg_aug_64x64": {
        "train": fitvid_randcropaug_resize(size=(64, 64), jpeg_input=False),
        "val": resize_only(size=(64, 64), jpeg_input=False),
    },
    # we use val transforms here since encoders are pretrained on a different task
    "DEIT_pretrained": {"train": DEIT_VAL_TRANSFORMS, "val": DEIT_VAL_TRANSFORMS,},
    # from: https://github.com/facebookresearch/dino/blob/cb711401860da580817918b9167ed73e3eef3dcf/eval_linear.py#L65-L70
    "DINO_pretrained": {"train": DEIT_VAL_TRANSFORMS, "val": DEIT_VAL_TRANSFORMS,},
    # from: https://github.com/facebookresearch/dinov2/blob/fc49f49d734c767272a4ea0e18ff2ab8e60fc92d/dinov2/data/transforms.py#L78-L92
    "DINOv2_pretrained": {"train": DEIT_VAL_TRANSFORMS, "val": DEIT_VAL_TRANSFORMS,},
    "VGG16_pretrained": {"train": IMNET_VAL_TRANSFORMS, "val": IMNET_VAL_TRANSFORMS,},
    "ResNet50_pretrained": {
        "train": IMNET_VAL_TRANSFORMS,
        "val": IMNET_VAL_TRANSFORMS,
    },
    "CLIP_pretrained": {"train": CLIP_VAL_TRANSFORMS, "val": CLIP_VAL_TRANSFORMS,},
    "R3M_pretrained": {"train": R3M_VAL_TRANSFORMS, "val": R3M_VAL_TRANSFORMS,},
    # from: https://github.com/facebookresearch/vip/blob/0b6c9dfe46d1925be967c517ff5c873633d683be/vip/examples/encoder_example.py#L25-L27
    "VIP_pretrained": {"train": R3M_VAL_TRANSFORMS, "val": R3M_VAL_TRANSFORMS,},
    "VC1_pretrained": {"train": VC1_VAL_TRANSFORMS, "val": VC1_VAL_TRANSFORMS,},
}

TRANSFORMS["fitvid_k700_64x64"] = TRANSFORMS["resize_only_64x64"]
TRANSFORMS["fitvid_k700_aug_64x64"] = TRANSFORMS["fitvid_aug_64x64"]
TRANSFORMS["fitvid_k700_128x128"] = TRANSFORMS["resize_only_128x128"]
TRANSFORMS["fitvid_k700_224x224"] = TRANSFORMS["resize_only_224x224"]

TRANSFORMS["fitvid_bridge_64x64"] = TRANSFORMS["resize_only_64x64"]
TRANSFORMS["fitvid_bridge_aug_64x64"] = TRANSFORMS["fitvid_aug_64x64"]

TRANSFORMS["fitvid_physion_roll_aug_64x64"] = TRANSFORMS["fitvid_nonjpeg_aug_64x64"]
TRANSFORMS["fitvid_physion_aug_64x64"] = TRANSFORMS["fitvid_nonjpeg_aug_64x64"]
TRANSFORMS["fitvid_physion_aug_64x64_2"] = dict()
TRANSFORMS["fitvid_physion_aug_64x64_2"]["train"] = fitvid_randcropaug_resize_2(
    size=(64, 64)
)
TRANSFORMS["fitvid_physion_aug_64x64_2"]["val"] = RESIZE_64x64_VAL_TRANSFORMS
# this one is ill defined because the augmentations are not the same across frames
TRANSFORMS["fitvid_ctxt7_physion_aug_64x64"] = TRANSFORMS["fitvid_physion_aug_64x64_2"]
# same random transform across frames
TRANSFORMS["fitvid_ctxt7_physion_aug_frames_64x64"] = dict()
TRANSFORMS["fitvid_ctxt7_physion_aug_frames_64x64"][
    "train"
] = fitvid_randcropaug_resize_2_frames(size=(64, 64))
TRANSFORMS["fitvid_ctxt7_physion_aug_frames_64x64"][
    "val"
] = RESIZE_64x64_VAL_TRANSFORMS_FRAMES
# training this version to fairly compare to SVG
TRANSFORMS["fitvid_ctxt7_physion_64x64"] = dict()
TRANSFORMS["fitvid_ctxt7_physion_64x64"]["train"] = RESIZE_64x64_VAL_TRANSFORMS
TRANSFORMS["fitvid_ctxt7_physion_64x64"]["val"] = RESIZE_64x64_VAL_TRANSFORMS

# matching fitvid transforms for fair comparison across architectures
# *not* doing augmentations here since this model is low capacity
TRANSFORMS["svg_physion_64x64"] = dict()
TRANSFORMS["svg_physion_64x64"]["train"] = RESIZE_64x64_VAL_TRANSFORMS
TRANSFORMS["svg_physion_64x64"]["val"] = RESIZE_64x64_VAL_TRANSFORMS
TRANSFORMS["svg_physion_128x128"] = dict()
TRANSFORMS["svg_physion_128x128"]["train"] = RESIZE_128x128_VAL_TRANSFORMS
TRANSFORMS["svg_physion_128x128"]["val"] = RESIZE_128x128_VAL_TRANSFORMS

# matching physion repo: https://github.com/neuroailab/physion/blob/master/physion/data/pydata.py#L72
# and here: https://github.com/neuroailab/physion/blob/master/configs/CSWM/physion.yaml#L38
TRANSFORMS["small_CSWM_physion"] = dict()
TRANSFORMS["small_CSWM_physion"]["train"] = RESIZE_224x224_VAL_TRANSFORMS
TRANSFORMS["small_CSWM_physion"]["val"] = RESIZE_224x224_VAL_TRANSFORMS
TRANSFORMS["medium_CSWM_physion"] = dict()
TRANSFORMS["medium_CSWM_physion"]["train"] = RESIZE_224x224_VAL_TRANSFORMS
TRANSFORMS["medium_CSWM_physion"]["val"] = RESIZE_224x224_VAL_TRANSFORMS
TRANSFORMS["large_CSWM_physion"] = dict()
TRANSFORMS["large_CSWM_physion"]["train"] = RESIZE_224x224_VAL_TRANSFORMS
TRANSFORMS["large_CSWM_physion"]["val"] = RESIZE_224x224_VAL_TRANSFORMS

for encoder in PF_ENCODERS:
    TRANSFORMS[f"pf{encoder}_ID"] = TRANSFORMS[f"{encoder}_pretrained"]
    TRANSFORMS[f"pf{encoder}_LSTM_physion"] = TRANSFORMS[f"{encoder}_pretrained"]
    TRANSFORMS[f"pf{encoder}_CTRNN_physion"] = TRANSFORMS[f"{encoder}_pretrained"]
    TRANSFORMS[f"pf{encoder}_LSTM_k700"] = TRANSFORMS[f"{encoder}_pretrained"]
    TRANSFORMS[f"pf{encoder}_CTRNN_k700"] = TRANSFORMS[f"{encoder}_pretrained"]
