import numpy as np
import copy
from collections import defaultdict
from mpmodels.core.constants import NUM_STIM, MAX_FRAMES


def dict_to_str(adict):
    """Converts a dictionary (e.g. hyperparameter configuration) into a string"""
    return "".join("{}{}".format(key, val) for key, val in sorted(adict.items()))


def check_np_equal(a, b):
    """Checks two numpy arrays are equal and works with nan values unlike np.array_equal.
    From: https://stackoverflow.com/questions/10710328/comparing-numpy-arrays-containing-nan"""
    return ((a == b) | (np.isnan(a) & np.isnan(b))).all()


def get_params_from_workernum(worker_num, param_lookup):
    return param_lookup[worker_num]


def dict_app(d, curr):
    assert set(list(d.keys())) == set(list(curr.keys()))
    for k, v in curr.items():
        d[k].append(v)


def dict_np(d):
    for k, v in d.items():
        assert isinstance(v, list)
        d[k] = np.array(v)


def interp_2d(features, num_interp, num_original, features_axis=0):
    assert len(features.shape) == 2
    assert features.shape[1 - features_axis] == num_original

    interp_resp = []
    for curr_feature_idx in range(features.shape[features_axis]):
        if features_axis == 0:
            curr_feats = features[curr_feature_idx, :]
        else:
            assert features_axis == 1
            curr_feats = features[:, curr_feature_idx]
        curr_interp_resp = np.interp(
            np.linspace(0, 1, num_interp), np.linspace(0, 1, num_original), curr_feats,
        )
        interp_resp.append(curr_interp_resp)
    interp_resp = np.stack(interp_resp, axis=features_axis)
    assert len(interp_resp.shape) == 2
    return interp_resp


def nan_filter(
    curr_resp,
    num_frames_arr,
    frame_start_idx=1,
    filter_frame_end_offset_idx=1,
    prep_2d=True,
    apply_filter=True,
):
    # helps to equalize timepoints for neural comparisons
    # we keep [curr_frame_start_idx ... num_frames - filter_frame_end_offset_idx - 1] inclusive,
    # as these are available for all model layers (encoder and future predictor).

    assert len(curr_resp.shape) == 3  # NUM_STIM x MAX_FRAMES x DIMS
    assert curr_resp.shape[0] == NUM_STIM
    assert curr_resp.shape[1] == MAX_FRAMES

    filtered_resp = copy.deepcopy(curr_resp)
    for i in range(NUM_STIM):
        if isinstance(frame_start_idx, int):
            curr_frame_start_idx = frame_start_idx
        else:
            curr_frame_start_idx = frame_start_idx[i]
            assert isinstance(curr_frame_start_idx, int) or isinstance(
                curr_frame_start_idx, np.int64
            )
        filtered_resp[i, :curr_frame_start_idx] = np.nan
        curr_num_frames = num_frames_arr[i]
        if filter_frame_end_offset_idx is not None:
            curr_end = curr_num_frames - filter_frame_end_offset_idx
            filtered_resp[i, curr_end:] = np.nan
            # check within the desired range that it is finite
            assert np.isfinite(filtered_resp[i, curr_frame_start_idx:curr_end]).all()
        else:
            assert np.isfinite(
                filtered_resp[i, curr_frame_start_idx:curr_num_frames]
            ).all()

    if prep_2d:
        filtered_resp = np.reshape(
            filtered_resp,
            (filtered_resp.shape[0] * filtered_resp.shape[1], filtered_resp.shape[2]),
        )
        # stimuli across all cells in a given animal/model that are finite
        ind = np.isfinite(filtered_resp).all(axis=-1)
        if apply_filter:
            filtered_resp = filtered_resp[ind]
            assert np.isfinite(filtered_resp).all()
        return filtered_resp, ind
    else:
        return filtered_resp


def concat_dict_sp(results_arr, partition_names=["train", "test"], agg_func=None):
    results_dict = {}
    for p in partition_names:
        results_dict[p] = defaultdict(list)

    for res_idx, res in enumerate(results_arr):  # of length num_train_test_splits
        for p in partition_names:
            for metric_name, metric_value in res[p].items():
                assert not isinstance(metric_value, dict)
                results_dict[p][metric_name].append(metric_value)

    for p, v1 in results_dict.items():
        for metric_name, metric_value in v1.items():
            # turn list into np array of train_test_splits x neurons
            metric_value_concat = np.stack(metric_value, axis=0)
            assert metric_value_concat.ndim == 2
            if agg_func is not None:
                metric_value_concat = agg_func(metric_value_concat, axis=0)
            results_dict[p][metric_name] = metric_value_concat

    return results_dict


def make_dict_list(d, num_times):
    return [d] * num_times
