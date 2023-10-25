import os
import io
import glob
import h5py
import json
from PIL import Image
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
from mpmodels.core.constants import PHYSION_SCENARIOS, PHYSION_DEFAULT_SUBSAMPLE_FACTOR


class PhysionDatasetBase(data.Dataset):
    def __init__(
        self,
        root_path,
        seq_len,
        scenarios=PHYSION_SCENARIOS,
        transform=None,
        transform_per_frame=True,
        random_seq=True,
        subsample_factor=PHYSION_DEFAULT_SUBSAMPLE_FACTOR,
        prefixes=None,
        seed=0,
    ):
        if prefixes is not None:
            if not isinstance(prefixes, list):
                prefixes = [prefixes]
        if not isinstance(scenarios, list):
            scenarios = [scenarios]
        self.scenarios = scenarios
        self.transform = transform
        self.transform_per_frame = transform_per_frame
        self.seq_len = seq_len
        # whether sequence should be sampled randomly from whole video or taken from the beginning
        self.random_seq = random_seq
        self.subsample_factor = subsample_factor
        self.rng = np.random.RandomState(seed=seed)

        self.hdf5_files = []
        data_paths = []
        for s in self.scenarios:
            if prefixes is not None:
                for p in prefixes:
                    curr_path = os.path.join(root_path, f"{s}/{s}/{p}/*.hdf5")
                    data_paths.append(curr_path)
            else:
                curr_path = os.path.join(root_path, f"{s}/{s}/*.hdf5")
                data_paths.append(curr_path)
        for path in data_paths:
            assert "*.hdf5" in path
            files = sorted(glob.glob(path))
            files = [fn for fn in files if "tfrecords" not in fn]
            assert len(files) > 0
            self.hdf5_files.extend(files)
            print("Processed {} with {} files".format(path, len(files)))
        self.N = len(self.hdf5_files)
        assert self.N > 0
        print("Dataset len: {}".format(self.N))

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.get_seq(index)

    def get_seq(self, index):
        with h5py.File(
            self.hdf5_files[index], "r"
        ) as f:  # load i-th hdf5 file from list
            frames = list(f["frames"])
            target_contacted_zone = False
            for frame in reversed(frames):
                lbl = f["frames"][frame]["labels"]["target_contacting_zone"][()]
                if lbl:  # as long as one frame touching, label is True
                    target_contacted_zone = True
                    break

            assert (
                len(frames) // self.subsample_factor >= self.seq_len
            ), "Images must be at least len {}, but are {}".format(
                self.seq_len, len(frames) // self.subsample_factor
            )
            if self.random_seq:  # randomly sample sequence of seq_len
                start_idx = self.rng.randint(
                    len(frames) - (self.seq_len * self.subsample_factor) + 1
                )
            else:  # get first seq_len # of frames
                start_idx = 0
            end_idx = start_idx + (self.seq_len * self.subsample_factor)
            images = []
            for frame in frames[start_idx : end_idx : self.subsample_factor]:
                img = f["frames"][frame]["images"]["_img"][()]
                img = Image.open(io.BytesIO(img))  # (256, 256, 3)
                if self.transform_per_frame and (self.transform is not None):
                    img = self.transform(img)
                images.append(img)

            if self.transform_per_frame and (self.transform is not None):
                assert isinstance(images, list)
                # in this case ImglistToTensor is *not* used,
                # and ToTensor() is used, then automatically stack images
                images = torch.stack(images, dim=0)
            elif self.transform is not None:
                images = self.transform(images)

            labels = (
                torch.ones((self.seq_len, 1))
                if target_contacted_zone
                else torch.zeros((self.seq_len, 1))
            )  # Get single label over whole sequence
            stimulus_name = f["static"]["stimulus_name"][()].decode("utf-8")

        sample = {
            "images": images,
            "binary_labels": labels,
            "stimulus_name": stimulus_name,
            "filepath": self.hdf5_files[index],
        }
        return sample


class PhysionDataset(PhysionDatasetBase):
    def __getitem__(self, index):
        sample = self.get_seq(index)
        images = sample["images"]  # (seq_len, 3, D', D')
        labels = sample["binary_labels"]
        return images, labels


class PhysionDatasetFrames(PhysionDatasetBase):
    def __getitem__(self, index):
        sample = self.get_seq(index)
        images = sample["images"]  # (seq_len, 3, D', D')
        return images
