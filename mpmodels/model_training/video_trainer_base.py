import os

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

import mpmodels
from mpmodels.model_training.video_dataset import VideoFrameDataset
from mpmodels.model_training.physion_dataset import PhysionDataset
from mpmodels.models.transforms import TRANSFORMS
from ptutils.model_training.train_utils import save_checkpoint
from ptutils.model_training.training_dataloader_utils import (
    _acquire_dataloader,
    wrap_dataloaders,
)
from ptutils.model_training.trainer import Trainer
from ptutils.model_training.trainer_transforms import compose_ifnot
from mpmodels.core.constants import PHYSION_SCENARIOS, PHYSION_DEFAULT_SUBSAMPLE_FACTOR


class BaseVideoTrainer(Trainer):
    def __init__(self, config):
        super(BaseVideoTrainer, self).__init__(config)

        # We need this condition just in case we loaded results previously using
        # load_checkpoint(). See __init__ in trainer.py and load_checkpoint() in
        # this file. If we loaded results previously, we don't want to overwrite
        # results with an empty dictionary.
        if not hasattr(self, "results"):
            self.results = dict()
            self.results["losses"] = {"train": [], "val": []}

        # We need this condition since the best (lowest) loss up to the current epoch
        # could have been loaded from a previous checkpoint, as above.
        if not hasattr(self, "best_loss"):
            self.best_loss = np.inf

        self.is_best = False

    # a lot of times we do have unused parameters (e.g. pretrained frozen encoder)
    def initialize_model(self):
        assert hasattr(self, "device")
        assert hasattr(self, "use_tpu")
        self.check_key("model")

        model = self.get_model(self.config["model"],)

        model = model.to(self.device)

        # gpu training
        if not self.use_tpu:
            assert hasattr(self, "gpu_ids")
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=self.gpu_ids, find_unused_parameters=True
            )

        model_name = self.config["model"]

        return model, model_name

    def get_video_loaders(self, params, my_transforms, rank=0, world_size=1, tpu=False):
        # Assumes root_path organization is /PATH/TO/VIDEO/{train, val}/{label}/*.JPEG
        assert "root_path" in params.keys()
        assert "train_root_path" in params.keys()
        assert "val_root_path" in params.keys()
        if self.dataset_type == "VideoFrameDataset":
            assert "path_label_include" in params.keys()
            assert "num_segments" in params.keys()
            assert "frames_per_segment" in params.keys()
            assert "imagefile_template" in params.keys()
            assert "transform_per_frame" in params.keys()
        else:
            assert self.dataset_type == "PhysionDataset"
            assert "seq_len" in params.keys()
            assert "scenarios" in params.keys()
            assert "subsample_factor" in params.keys()
            assert "train_prefixes" in params.keys()
            assert "val_prefixes" in params.keys()
            assert "transform_per_frame" in params.keys()
        assert "train_batch_size" in params.keys()
        assert "val_batch_size" in params.keys()
        assert "num_workers" in params.keys()
        assert "train" in my_transforms.keys()
        assert "val" in my_transforms.keys()

        train_batch_size = params["train_batch_size"]
        val_batch_size = params["val_batch_size"]
        num_workers = params["num_workers"]
        drop_last = params.get("drop_last", False)
        train_transforms = my_transforms["train"]
        val_transforms = my_transforms["val"]

        if train_transforms is not None:
            print("Loading train loader")
            train_root_path = params["train_root_path"]
            if train_root_path is None:
                assert params["root_path"] is not None
                train_root_path = os.path.join(params["root_path"], "train")
            if self.dataset_type == "VideoFrameDataset":
                train_set = VideoFrameDataset(
                    root_path=train_root_path,
                    annotationfile_path=os.path.join(
                        train_root_path, "train_processed.txt"
                    ),
                    path_label_include=params["path_label_include"],
                    num_segments=params["num_segments"],
                    frames_per_segment=params["frames_per_segment"],
                    imagefile_template=params["imagefile_template"],
                    transform=train_transforms,
                    transform_per_frame=params["transform_per_frame"],
                    test_mode=False,
                )
            else:
                assert self.dataset_type == "PhysionDataset"
                train_set = PhysionDataset(
                    root_path=train_root_path,
                    seq_len=params["seq_len"],
                    scenarios=params["scenarios"],
                    subsample_factor=params["subsample_factor"],
                    transform=train_transforms,
                    transform_per_frame=params["transform_per_frame"],
                    random_seq=True,
                    prefixes=params["train_prefixes"],
                    seed=self.config["seed"],
                )
        print("Loading val loader")
        val_root_path = params["val_root_path"]
        if val_root_path is None:
            assert params["root_path"] is not None
            val_root_path = os.path.join(params["root_path"], "val")
        if self.dataset_type == "VideoFrameDataset":
            val_set = VideoFrameDataset(
                root_path=val_root_path,
                annotationfile_path=os.path.join(val_root_path, "val_processed.txt"),
                path_label_include=params["path_label_include"],
                num_segments=params["num_segments"],
                frames_per_segment=params["frames_per_segment"],
                imagefile_template=params["imagefile_template"],
                transform=val_transforms,
                transform_per_frame=params["transform_per_frame"],
                test_mode=True,
            )
        else:
            assert self.dataset_type == "PhysionDataset"
            val_set = PhysionDataset(
                root_path=val_root_path,
                seq_len=params["seq_len"],
                scenarios=params["scenarios"],
                subsample_factor=params["subsample_factor"],
                transform=val_transforms,
                transform_per_frame=params["transform_per_frame"],
                random_seq=False,
                prefixes=params["val_prefixes"],
                seed=self.config["seed"],
            )

        if train_transforms is not None:
            train_loader = _acquire_dataloader(
                dataset=train_set,
                train=True,
                batch_size=train_batch_size,
                num_workers=num_workers,
                rank=rank,
                world_size=world_size,
                drop_last=drop_last,
                tpu=tpu,
            )
        else:
            train_loader = None

        val_loader = _acquire_dataloader(
            dataset=val_set,
            train=False,
            batch_size=val_batch_size,
            num_workers=num_workers,
            rank=rank,
            world_size=world_size,
            drop_last=drop_last,
            tpu=tpu,
        )

        return train_loader, val_loader

    def get_model(self, model_name):
        """
        Inputs:
            model_name : (string) Name of deep net architecture.

        Outputs:
            model     : (torch.nn.DataParallel) model
        """
        model = mpmodels.models.__dict__[model_name]()
        return model

    def initialize_dataloader(self):
        assert hasattr(self, "config")
        assert hasattr(self, "use_tpu")
        assert hasattr(self, "model_name")

        self.check_key("optimizer_params")
        self.check_key("dataloader_params")

        assert "train_batch_size" in self.config["optimizer_params"].keys()
        assert "val_batch_size" in self.config["optimizer_params"].keys()

        self.dataset_type = self.config["dataloader_params"].get(
            "dataset_type", "VideoFrameDataset"
        )

        params = dict()
        # video dataset params
        params["root_path"] = self.config["dataloader_params"].get("root_path", None)
        params["train_root_path"] = self.config["dataloader_params"].get(
            "train_root_path", None
        )
        params["val_root_path"] = self.config["dataloader_params"].get(
            "val_root_path", None
        )
        if self.dataset_type == "VideoFrameDataset":
            params["path_label_include"] = self.config["dataloader_params"].get(
                "path_label_include", None
            )
            params["num_segments"] = self.config["dataloader_params"]["num_segments"]
            params["frames_per_segment"] = self.config["dataloader_params"][
                "frames_per_segment"
            ]
            params["imagefile_template"] = self.config["dataloader_params"][
                "imagefile_template"
            ]
            params["transform_per_frame"] = self.config["dataloader_params"].get(
                "transform_per_frame", False
            )
        else:
            assert self.dataset_type == "PhysionDataset"
            params["seq_len"] = self.config["dataloader_params"]["seq_len"]
            params["scenarios"] = self.config["dataloader_params"].get(
                "scenarios", PHYSION_SCENARIOS
            )
            params["subsample_factor"] = self.config["dataloader_params"].get(
                "subsample_factor", PHYSION_DEFAULT_SUBSAMPLE_FACTOR
            )
            params["train_prefixes"] = self.config["dataloader_params"].get(
                "train_prefixes", None
            )
            params["val_prefixes"] = self.config["dataloader_params"].get(
                "val_prefixes", None
            )
            # preserving defaults
            params["transform_per_frame"] = self.config["dataloader_params"].get(
                "transform_per_frame", True
            )
        params["num_workers"] = self.config["dataloader_params"]["dataloader_workers"]
        # batch size
        params["train_batch_size"] = self.config["optimizer_params"]["train_batch_size"]
        params["val_batch_size"] = self.config["optimizer_params"]["val_batch_size"]

        my_transforms = dict()
        my_transforms["train"] = compose_ifnot(TRANSFORMS[self.model_name]["train"])
        my_transforms["val"] = compose_ifnot(TRANSFORMS[self.model_name]["val"])

        train_loader, val_loader = wrap_dataloaders(
            dataloader_func=self.get_video_loaders,
            params=params,
            my_transforms=my_transforms,
            device=self.device,
            rank=self.rank,
            world_size=self.world_size,
        )
        return train_loader, val_loader

    def save_checkpoint(self):
        assert hasattr(self, "current_epoch")
        assert hasattr(self, "model")
        assert hasattr(self, "optimizer")
        assert hasattr(self, "results")
        assert hasattr(self, "save_dir")
        assert hasattr(self, "use_tpu")
        assert hasattr(self, "config")
        assert "save_freq" in self.config.keys()

        curr_state = {
            "epoch": self.current_epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "results": self.results,
            "curr_best_loss": self.best_loss,
        }

        # Every "save_freq" epochs, save into new checkpoint file and relevant keys
        # to database. Overwrite existing checkpoint otherwise.
        save_epoch = None
        if (self.current_epoch % self.config["save_freq"] == 0) or (
            self.current_epoch + 1 == self.config["num_epochs"]
        ):
            save_epoch = self.current_epoch
            self._save_to_db(
                curr_state=curr_state, save_keys=["epoch", "results", "curr_best_loss"]
            )

        save_checkpoint(
            state=curr_state,
            save_dir=self.save_dir,
            is_best=self.is_best,
            save_epoch=save_epoch,
            rank=self.rank,
            tpu=self.use_tpu,
        )

    def load_checkpoint(self):
        assert hasattr(self, "model")
        assert hasattr(self, "optimizer")
        assert hasattr(self, "config")
        assert hasattr(self, "use_tpu")
        self.check_key("resume_checkpoint")

        checkpoint_path = self.config["resume_checkpoint"]
        if os.path.isfile(checkpoint_path):
            if self.use_tpu:
                # on TPU, the Trainer locally saves each core's copy of the same saved parameters
                # to a different local file name, and there is a single TPU device we are loading to
                cpt = torch.load(checkpoint_path)
            else:
                # According to: https://stackoverflow.com/questions/61642619/pytorch-distributed-data-parallel-confusion
                # When saving the parameters (or any tensor for that matter)
                # PyTorch includes the device where it was stored. On gpu,
                # this is always gpu 0. Therefore, we ensure we map the SAME parameters to each
                # other gpu (not just gpu 0), otherwise it will load the same model
                # multiple times on 1 gpu.
                assert len(self.gpu_ids) == 1  # one subprocess per gpu
                cpt = torch.load(
                    checkpoint_path, map_location="cuda:{}".format(self.gpu_ids[0])
                )
            self.print_fn(f"Loaded checkpoint at '{checkpoint_path}'")
        else:
            raise ValueError(f"No checkpoint at '{checkpoint_path}'")

        # Make sure keys are in the checkpoint
        assert "epoch" in cpt.keys()
        assert "results" in cpt.keys()
        assert "state_dict" in cpt.keys()
        assert "optimizer" in cpt.keys()
        assert "curr_best_loss" in cpt.keys()

        # Load current epoch, +1 since we stored the last completed epoch
        self.current_epoch = cpt["epoch"] + 1

        # Load results
        assert not hasattr(self, "results")
        self.results = cpt["results"]

        # Load model state dict
        self.model.load_state_dict(cpt["state_dict"])

        # Load optimizer state dict
        self.optimizer.load_state_dict(cpt["optimizer"])

        # Load the current best (lowest) loss
        assert not hasattr(self, "best_loss")
        self.best_loss = cpt["curr_best_loss"]
