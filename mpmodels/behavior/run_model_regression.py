import os
import numpy as np

import mpmodels
from mpmodels.core.constants import PHYSION_SCENARIOS
from mpmodels.core.default_dirs import (
    PHYSION_BASE_DIR,
    OCP_PHYSION_REGRESSION_RESULTS_DIR,
)
from mpmodels.core.feature_extractor import get_model_loader_kwargs, get_model_kwargs
from mpmodels.model_training.physion_dataset import (
    PhysionDatasetBase,
    PhysionDatasetFrames,
)
from mpmodels.models.transforms import TRANSFORMS
from mpmodels.models.layers import LAYERS
from mpmodels.models.paths import PATHS

from brainmodel_utils.models.utils import (
    get_base_model_name,
    get_model_func_from_name,
    get_model_transforms_from_name,
    get_model_path_from_name,
    get_model_layers_from_name,
)
from brainmodel_utils.models.dataloader_utils import get_generic_dataloader
from brainmodel_utils.models.feature_extractor import ModelFeaturesPipeline

from ptutils.model_training.trainer_transforms import compose_ifnot

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold


class OCPScenarioFeatures(ModelFeaturesPipeline):
    def __init__(
        self,
        model_name,
        mode,
        scenarios=PHYSION_SCENARIOS,
        subsample_factor=6,  # every model I trained uses this
        seq_len=25,  # every model I trained uses this
        transform_per_frame=True,
        batch_size=256,
        vectorize=True,
        dataloader_name="get_generic_dataloader",
        model_transforms_key="val",
        **kwargs,
    ):
        assert mode in ["train", "test"]
        self.scenarios = scenarios
        self.subsample_factor = subsample_factor
        self.seq_len = seq_len
        self.mode = mode
        self.transform_per_frame = transform_per_frame
        self.batch_size = batch_size
        self.vectorize = vectorize
        self.model_transforms_key = model_transforms_key
        # we will fill these in automatically
        assert "model_path" not in kwargs.keys()
        assert "dataloader_transforms" not in kwargs.keys()
        model_loader_kwargs = get_model_loader_kwargs(model_name)
        model_kwargs = get_model_kwargs()

        if mode == "train":
            root_path = os.path.join(PHYSION_BASE_DIR, "readout_training/")
            prefix = None
        else:
            root_path = os.path.join(PHYSION_BASE_DIR, "testing/")
            prefix = "hdf5s-redyellow"

        transform = compose_ifnot(
            get_model_transforms_from_name(
                model_transforms_dict=TRANSFORMS,
                model_name=model_name,
                model_transforms_key=self.model_transforms_key,
            )
        )

        self.stimuli_dataset_kwargs = {
            "root_path": root_path,
            "seq_len": self.seq_len,
            "subsample_factor": self.subsample_factor,
            "scenarios": self.scenarios,
            "random_seq": False,
            "prefixes": prefix,
            "transform": transform,
            "transform_per_frame": self.transform_per_frame,
        }

        self.stimuli = PhysionDatasetFrames(**self.stimuli_dataset_kwargs)

        # this loads the model
        super(OCPScenarioFeatures, self).__init__(
            model_name=model_name,
            model_kwargs=model_kwargs,
            model_loader_kwargs=model_loader_kwargs,
            model_path=get_model_path_from_name(
                model_paths_dict=PATHS, model_name=model_name
            ),
            dataloader_name=dataloader_name,
            dataloader_transforms=None,  # already passed into dataloader above
            dataloader_kwargs={"batch_size": self.batch_size},
            feature_extractor_kwargs={"vectorize": self.vectorize},
            **kwargs,
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


def get_stim_names_and_labels(dataset_kwargs, batch_size=256, **kwargs):
    dataloader = get_generic_dataloader(
        PhysionDatasetBase(**dataset_kwargs), batch_size=batch_size, **kwargs
    )

    stim_names = []
    filepaths = []
    labels = []
    for i, x in enumerate(dataloader):
        stim_names.extend(x["stimulus_name"])
        filepaths.extend(x["filepath"])
        curr_label = x["binary_labels"].cpu().numpy()
        # batch x time x dimensions (1)
        assert curr_label.ndim == 3
        # true or false if there was object contact at any timepoint
        labels.append(np.any(curr_label, axis=(1, 2)).astype(np.int32))
    labels = np.concatenate(labels, axis=0)
    assert labels.ndim == 1
    assert len(labels) == len(stim_names)
    ## ensure each name is unique -- actually this does not hold across all scenarios, so we enforce it later when doing the regression on a subselected portion of the data
    # assert len(stim_names) == len(np.unique(stim_names))
    assert len(filepaths) == len(stim_names)
    # ensure each filepath is unique
    assert len(filepaths) == len(np.unique(filepaths))
    return stim_names, labels, filepaths


def run_regression(
    model_name,
    layer_name="dynamics",
    param_grid={"clf__C": np.logspace(-8, 8, 17), "clf__penalty": ["l2"]},
    max_iter=20000,
    n_cv_splits=5,
    random_state=42,
    **kwargs,
):
    # Create the pipeline with a logistic regression model and a scaler
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=max_iter))]
    )
    stratified_kfold = StratifiedKFold(
        n_splits=n_cv_splits, shuffle=True, random_state=random_state
    )
    grid_search = GridSearchCV(
        estimator=pipeline, param_grid=param_grid, cv=stratified_kfold, verbose=3
    )

    # train Logistic Regression
    OCP_train = OCPScenarioFeatures(model_name=model_name, mode="train", **kwargs)
    train_features = OCP_train.get_model_features(OCP_train.stimuli)[layer_name]
    train_stim_names, train_labels, train_filepaths = get_stim_names_and_labels(
        OCP_train.stimuli_dataset_kwargs, batch_size=kwargs.get("batch_size", 256)
    )
    assert train_features.ndim == 2
    assert train_features.shape[0] == len(train_labels)
    grid_search.fit(train_features, train_labels)

    result = grid_search.best_params_
    result["train_accuracy"] = grid_search.score(train_features, train_labels)

    OCP_test = OCPScenarioFeatures(model_name=model_name, mode="test", **kwargs)
    test_features = OCP_test.get_model_features(OCP_test.stimuli)[layer_name]
    test_stim_names, test_labels, test_filepaths = get_stim_names_and_labels(
        OCP_test.stimuli_dataset_kwargs, batch_size=kwargs.get("batch_size", 256)
    )
    assert test_features.ndim == 2
    assert test_features.shape[0] == len(test_labels)
    result["test_probabilities"] = grid_search.predict_proba(test_features)
    result["test_accuracy"] = grid_search.score(test_features, test_labels)
    result["test_labels"] = test_labels
    result["test_stim_names"] = test_stim_names
    result["test_filepaths"] = test_filepaths
    result["classes"] = grid_search.classes_

    print("Test accuracy:", result["test_accuracy"])

    return result


def main(args, models):
    for model_name in models:
        args.model_name = model_name

        print(f"Running regression for {args.model_name}...")
        result = run_regression(
            model_name=args.model_name,
            layer_name=args.layer_name,
            transform_per_frame=True if not args.group_transform else False,
        )
        fname = os.path.join(
            OCP_PHYSION_REGRESSION_RESULTS_DIR,
            f"{args.model_name}_layer{args.layer_name}.npz",
        )
        np.savez(fname, result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", type=str, default=None, help="What gpu to use (if any)."
    )
    parser.add_argument("--models", type=str, default=None, required=True)
    parser.add_argument("--layer-name", type=str, default="dynamics")
    parser.add_argument("--group-transform", type=bool, default=False)
    args = parser.parse_args()

    # If GPUs available, select which to train on
    if args.gpu is not None:
        print(f"Using GPU {args.gpu}")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        print("Using CPU")

    models = args.models.split(",")
    print(f"Getting features for {models}.")
    main(args=args, models=models)
