import torch
from torchvision import transforms
import os
import numpy as np
import glob
from mpmodels.core.default_dirs import NEURAL_STIM_JPEG_DIR
from mpmodels.core.constants import MAX_FRAMES
from mpmodels.model_training.video_dataset import VideoFrameDataset
from ptutils.model_training.trainer_transforms import compose_ifnot
from brainmodel_utils.models.dataloader_utils import get_generic_dataloader


def idx_from_cond(cond):
    if cond == "occ":
        cond_idx = 1
    else:
        assert cond == "vis"
        cond_idx = 0
    return cond_idx


def get_frames(
    i,
    cond="occ",
    transform=[transforms.ToTensor()],
    subsample_factor=None,
    n_past_idxs=None,
    transform_per_frame=True,
    **kwargs,
):
    cond_idx = idx_from_cond(cond)
    curr_name = glob.glob(
        os.path.join(NEURAL_STIM_JPEG_DIR, f"{cond}/*_occ_{cond_idx}_idx{i}")
    )
    assert len(curr_name) == 1
    curr_name = curr_name[0]
    # get the last part of the specificfile_path to point dataloader to
    specificfile_path = "/".join(curr_name.split("/")[-2:])
    video_set = VideoFrameDataset(
        root_path=NEURAL_STIM_JPEG_DIR,
        annotationfile_path=os.path.join(NEURAL_STIM_JPEG_DIR, f"{cond}_processed.txt"),
        specificfile_path=specificfile_path,
        num_segments=None,
        frames_per_segment=None,
        imagefile_template="frame_{0:03d}.jpg",
        transform=compose_ifnot(transform),
        transform_per_frame=transform_per_frame,
        test_mode=True,
    )
    dataloader = get_generic_dataloader(dataset=video_set, **kwargs)
    assert len(dataloader) == 1
    for v in dataloader:
        assert len(v) == 2
        assert len(v[1]) == 0
        frames = v[0]
    if torch.cuda.is_available():
        frames = frames.cuda()
    # B x F x C x H x W
    assert frames.ndim == 5
    # batch size of 1
    assert frames.shape[0] == 1
    num_frames = frames.shape[1]
    assert num_frames <= MAX_FRAMES
    if subsample_factor is not None:
        frames = frames[:, 0:num_frames:subsample_factor]
    if n_past_idxs is not None:
        # n_past_idxs already takes subsample factor into account
        assert isinstance(n_past_idxs, list)
        assert len(n_past_idxs) > 0
        # use model's native n_past
        curr_n_past = n_past_idxs[-1]
        # get the mean difference, floor it to maximize rollout from the model prior to feature interpolation
        spacing = np.floor(np.mean(np.diff(n_past_idxs))).astype(np.int64)
        curr_num_frames = frames.shape[1]
        future_idxs = np.arange(curr_n_past + spacing, curr_num_frames, spacing).astype(
            np.int64
        )
        assert len(future_idxs) > 0
        # make sure no frame is repeated
        assert len(future_idxs) == len(np.unique(future_idxs))
        # empty intersection
        assert set(list(future_idxs)) & set(list(n_past_idxs)) == set([])
        sample_idxs = list(n_past_idxs) + list(future_idxs)
        frames = frames[:, sample_idxs]

    return frames, num_frames
