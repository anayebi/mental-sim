# Models of Mental Simulation
This repo contains pretrained PyTorch models that are optimized to predict the future state of their environment.
As these models are stimulus-computable, they can be ``dropped in'' on any new video for further neural and behavioral comparisons of your design.

This repository is based on our paper:

**Aran Nayebi, Rishi Rajalingham, Mehrdad Jazayeri, Guangyu Robert Yang**

["Neural foundations of mental simulation: future prediction of latent representations on dynamic scenes"](https://arxiv.org/abs/2305.11772)

*37th Conference on Neural Information Processing Systems (NeurIPS 2023). Selected for spotlight.*

Here's a [video recording](https://youtu.be/9h_3bHVDMhA?t=1756) that explains our work a bit.

## Getting started
It is recommended that you install this repo within a virtual environment (Python 3.6 recommended), and run inferences there.
An example command for doing this with `anaconda` would be:
```
conda create -y -n your_env python=3.9.7 anaconda
```
To install this package and all of its dependecies, clone this repo on your machine and then install it via pip:
1. `git clone https://github.com/anayebi/mental-sim.git` to clone the repository.
2. `cd mental-sim/`
3. `conda activate your_env`
4. Run `pip install -e .` to install the current version.
The installation above will automatically download the necessary dependencies, which includes [`ptutils`](https://github.com/anayebi/ptutils) and [`brainmodel_utils`](https://github.com/anayebi/brainmodel_utils), which are my Python packages for training PyTorch models and extracting their features for neural and behavioral regression, respectively.

## Available Pre-trained Models
To get the saved checkpoints of the models, simply run this bash script:
```
./get_checkpoints.sh
```
This will automatically download them from [Hugging Face](https://huggingface.co/anayebi/mental-sim-models) to the current directory in the folder `./trained_models/`.
If you want a subset of the models, feel free to modify the `for` loop in the above bash script.

Models are named according to the convention of `[architecture]_[pretraining-dataset]_[image_size]`, all of which are described in [our paper](https://arxiv.org/abs/2305.11772).
**You can see [this notebook](https://github.com/anayebi/mental-sim/blob/main/Loading%20model%20weights.ipynb) for loading all of our pretrained models.**

Some models may be better suited than others based on your needs, but we generally recommend: 
- `VC-1+CTRNN/LSTM` models, where the dynamics module is pretrained on either Physion or the much larger Kinetics-700 dataset. This model class reasonably matches *both* Mental-Pong neural and OCP behavioral benchmarks we tested.
- `R3M+CTRNN/LSTM` models, where the dynamics module is pretrained on either Physion or the much larger Kinetics-700 dataset. This model class best matches the Mental-Pong neural benchmark we tested.

We also include our best Physion-pretrained FitVid, SVG, and temporally-augmented C-SWM models, for additional points of comparison involving end-to-end pixel-wise and object-slot future predictors.

Once you have loaded the PyTorch model, you can extract features according to your pipeline of choice.
**Note that all of the models expect 7 context frames before running the forward simulation, so be sure to provide that minimally as input!**
If you want a standard example of extracting model features and running behavioral regression, see [here](https://github.com/anayebi/mental-sim/blob/main/mpmodels/behavior/run_model_regression.py).
If you want examples of extracting model features per video (where the number of frames can be different per video, so they must be processed one at a time), see [here](https://github.com/anayebi/mental-sim/blob/main/mpmodels/core/feature_extractor.py#L42-L196).

## Training Code
Download your video pretraining dataset of choice and then run under `mpmodels/model_training/`:
```
python runner.py --config=[]
```
Specify the `gpu_id` in the config. The configs and hyperparameters we used are specified in the `mpmodels/model_training/configs` [directory](https://github.com/anayebi/mental-sim/tree/main/mpmodels/model_training/configs).
Model architectures are implemented in the `mpmodels/models/` [directory](https://github.com/anayebi/mental-sim/tree/main/mpmodels/models).

For example, to train our `VC-1+CTRNN` model on the Physion dataset, you can run this command:
```
CUDA_VISIBLE_DEVICES=0 python runner.py --config=configs/pretrained_frozen_encoder/pfVC1_CTRNN/physion.json
```
Note that you will have to modify the `save_prefix` key in the json file to the directory that you want to save your checkpoints, as well as the `train_root_path` and `val_root_path` directories to point to where the pretraining dataset is stored.

## Cite
If you used this codebase for your research, please consider citing our paper:
```
@inproceedings{nayebi2023neural,
  title={Neural Foundations of Mental Simulation: Future Prediction of Latent Representations on Dynamic Scenes},
  author={Nayebi, Aran and Rajalingham, Rishi and Jazayeri, Mehrdad and Yang, Guangyu Robert},
  booktitle={The 37th Conference on Neural Information Processing Systems (NeurIPS 2023)},
  url={https://arxiv.org/abs/2305.11772},
  year={2023}
}
```

## Contact
If you have any questions or encounter issues, either submit a Github issue here or email `anayebi@mit.edu`.
