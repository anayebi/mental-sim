import os
import copy
import numpy as np
import pickle
import torch
from torchvision import transforms

import mpmodels
from mpmodels.core.utils import interp_2d
from mpmodels.core.constants import NUM_STIM, MAX_FRAMES
from mpmodels.core.default_dirs import NEURAL_RESP_DIR, MODEL_FEATURES_SAVE_DIR
from mpmodels.models.transforms import TRANSFORMS
from mpmodels.models.layers import LAYERS
from mpmodels.models.paths import PATHS
from mpmodels.phys.stim_utils import get_frames
from ptutils.core.utils import set_seed
from brainmodel_utils.models.utils import (
    get_base_model_name,
    get_model_func_from_name,
    get_model_transforms_from_name,
    get_model_path_from_name,
    get_model_layers_from_name,
)
from brainmodel_utils.models.feature_extractor import ModelFeaturesPipeline


def get_model_loader_kwargs(model_name):
    model_loader_kwargs = dict()
    if model_name == "fitvid_physion_64x64":
        # I didn't train this model, so ckpt doesn't have state_dict_key
        model_loader_kwargs["state_dict_key"] = None
    return model_loader_kwargs


def get_model_kwargs():
    # only applies to the models that have this flag; otherwise ignored
    model_kwargs = dict()
    model_kwargs["full_rollout"] = True
    return model_kwargs


class SingleVideoModelFeatures(ModelFeaturesPipeline):
    def __init__(
        self,
        model_name,
        stim_idx,
        n_past,
        cond="occ",
        subsample_factor=None,
        fixed_n_past=False,
        vectorize=True,
        dataloader_name="get_passthrough_dataloader",
        model_transforms_key="val",
        transform_per_frame=True,
        **kwargs,
    ):
        self.stim_idx = stim_idx
        self.cond = cond
        self.vectorize = vectorize
        self.subsample_factor = subsample_factor
        self.fixed_n_past = fixed_n_past
        self.model_transforms_key = model_transforms_key
        self.transform_per_frame = transform_per_frame
        # we will fill these in automatically
        assert "model_path" not in kwargs.keys()
        assert "dataloader_transforms" not in kwargs.keys()
        model_loader_kwargs = get_model_loader_kwargs(model_name)
        model_kwargs = get_model_kwargs()

        curr_n_past = n_past
        if self.subsample_factor is not None:
            assert self.subsample_factor < curr_n_past
            # get index of curr_n_past in the subsampled movie (corresponds to last index of this array, hence the -1)
            # (we add +1 in arange to include it if it evenly divides subsample_factor)
            curr_n_past = len(np.arange(0, curr_n_past + 1, self.subsample_factor)) - 1

        if not self.fixed_n_past:
            model_kwargs["n_past"] = curr_n_past

        # this loads the model
        super(SingleVideoModelFeatures, self).__init__(
            model_name=model_name,
            model_kwargs=model_kwargs,
            model_loader_kwargs=model_loader_kwargs,
            model_path=get_model_path_from_name(
                model_paths_dict=PATHS, model_name=model_name
            ),
            dataloader_name=dataloader_name,
            dataloader_transforms=None,
            feature_extractor_kwargs={"vectorize": self.vectorize, "temporal": True},
            **kwargs,
        )

        n_past_idxs = None
        if self.fixed_n_past:
            # get indices of curr_n_past
            n_past_idxs = list(
                np.linspace(0, curr_n_past, self.model.n_past, endpoint=True).astype(
                    np.int64
                )
            )
            # make sure no frame is repeated
            assert len(n_past_idxs) == len(np.unique(n_past_idxs))

        # base this on the loaded self.model
        self.frames, self.num_frames = get_frames(
            self.stim_idx,
            cond=self.cond,
            transform=get_model_transforms_from_name(
                model_transforms_dict=TRANSFORMS,
                model_name=model_name,
                model_transforms_key=self.model_transforms_key,
            ),
            subsample_factor=self.subsample_factor,
            n_past_idxs=n_past_idxs,
            transform_per_frame=self.transform_per_frame,
        )

    def _get_model_func_from_name(self, model_name, model_kwargs):
        return get_model_func_from_name(
            model_func_dict=mpmodels.models.__dict__,
            model_name=model_name,
            model_kwargs=model_kwargs,
        )

    def _get_model_layers_list(self, model_name, model_kwargs):
        return get_model_layers_from_name(
            model_layers_dict=LAYERS, model_name=model_name
        )

    def _postproc_features(self, features):
        interp_feats = features
        if self.vectorize and (
            (self.subsample_factor is not None) or (self.fixed_n_past)
        ):
            # B x T x D
            assert len(features.shape) == 3
            interp_feats = []
            for l_b_idx in range(features.shape[0]):
                # T x D, interpolate across time
                curr_interp_feats = interp_2d(
                    features=features[l_b_idx],
                    num_interp=self.num_frames,
                    num_original=features[l_b_idx].shape[0],
                    features_axis=1,
                )
                interp_feats.append(curr_interp_feats)
            interp_feats = np.stack(interp_feats, axis=0)
        return interp_feats


def aggregate_model_features(
    model_name, eval_mode="occ_frame_start", cond="occ", **kwargs
):
    neural_dat = np.load(
        os.path.join(
            NEURAL_RESP_DIR,
            f"neural_responses_reliable_cond{cond}_frameinterpolated.npz",
        ),
        allow_pickle=True,
    )["arr_0"][()]
    occ_frame_start_idxs = neural_dat["occ_frame_start_idxs"]
    vis_frame_end_idxs = neural_dat["vis_frame_end_idxs"]

    layer_features_full = dict()
    for i in range(NUM_STIM):
        if eval_mode == "occ_frame_start":
            # up to but just up to excluding the first frame after it is occluded
            # (corresponds to occ_frame_start_idxs[i], which is the first frame after occluder),
            # so the model knows the ball disappears rather than collides or bounces
            n_past = occ_frame_start_idxs[i]
            print(f"{eval_mode} n past {n_past}")
        else:
            assert eval_mode == "vis_frame_end"
            # up to and including the last frame when the ball is fully visible
            n_past = vis_frame_end_idxs[i] + 1
            print(f"{eval_mode} n past {n_past}")

        mf = SingleVideoModelFeatures(
            model_name=model_name, stim_idx=i, n_past=n_past, cond=cond, **kwargs
        )
        layer_feats = mf.get_model_features(mf.frames)
        if i == 0:
            # initialize
            for k, v in layer_feats.items():
                layer_features_full[k] = (
                    np.zeros((NUM_STIM, MAX_FRAMES,) + v.shape[2:]) + np.nan
                )

        for k, v in layer_feats.items():
            assert v.shape[0] == 1
            assert len(v.shape) >= 3
            assert v.shape[1] == mf.num_frames
            layer_features_full[k][i, : mf.num_frames] = v[0]

    return layer_features_full


def construct_filename(
    model_name,
    cond="occ",
    eval_mode="occ_frame_start",
    subsample_factor=None,
    fixed_n_past=False,
):
    # Set up filename for the model features
    save_dir = os.path.join(MODEL_FEATURES_SAVE_DIR, f"{cond}/{model_name}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fname = ""
    if eval_mode != "occ_frame_start":
        fname += f"_eval{eval_mode}"
    if subsample_factor is not None:
        fname += f"_sf{subsample_factor}"
    if fixed_n_past:
        fname += f"_fixednpast"
    fname += ".npz"
    fname = os.path.join(save_dir, fname)
    return fname


def load_and_save_features_for_dataset(args):
    # Load features
    features = aggregate_model_features(
        model_name=args.model_name,
        cond=args.cond,
        eval_mode=args.eval_mode,
        subsample_factor=args.subsample_factor,
        fixed_n_past=args.fixed_n_past,
        transform_per_frame=True if not args.group_transform else False,
    )

    # Save features
    fname = construct_filename(
        model_name=args.model_name,
        cond=args.cond,
        eval_mode=args.eval_mode,
        subsample_factor=args.subsample_factor,
        fixed_n_past=args.fixed_n_past,
    )
    try:
        np.savez(fname, features)
    except OverflowError:
        # for large features, protocol 4 is needed whereas numpy uses protocol 3
        pickle.dump(features, open(fname, "wb"), protocol=4)


def load_and_save_features_for_model(args):
    set_seed(int(args.seed))
    load_and_save_features_for_dataset(args)


def load_and_save_input_features(args):
    # Set up filename for the input features
    save_dir = os.path.join(MODEL_FEATURES_SAVE_DIR, f"{args.cond}/")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fname = os.path.join(save_dir, "inputs.npz")

    assert args.fixed_n_past is False  # won't affect stimulus
    for i in range(NUM_STIM):
        frames, curr_num_frames = get_frames(
            i,
            cond=args.cond,
            subsample_factor=args.subsample_factor,
            transform_per_frame=True if not args.group_transform else False,
        )
        frames = frames.cpu().numpy()
        frames = np.reshape(frames, (frames.shape[0], frames.shape[1], -1))
        if i == 0:
            input_features_full = (
                np.zeros((NUM_STIM, MAX_FRAMES,) + frames.shape[2:]) + np.nan
            )
        assert frames.shape[0] == 1
        input_features_full[i, :curr_num_frames, :] = frames[0]

    np.savez(fname, input_features_full)


def main(args, models):
    for model_name in models:
        args.model_name = model_name

        print(f"Saving features for {args.model_name}...")
        if args.model_name == "inputs":
            load_and_save_input_features(args)
        else:
            load_and_save_features_for_model(args)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", type=str, default=None, help="What gpu to use (if any)."
    )
    parser.add_argument("--models", type=str, default=None, required=True)
    parser.add_argument("--cond", type=str, default="occ")
    parser.add_argument("--eval-mode", type=str, default="occ_frame_start")
    parser.add_argument("--subsample-factor", type=int, default=None)
    parser.add_argument("--fixed-n-past", type=bool, default=False)
    parser.add_argument("--group-transform", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # If GPUs available, select which to train on
    if args.gpu is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    models = args.models.split(",")
    print(f"Getting features for {models}.")
    main(args=args, models=models)
