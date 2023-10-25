from setuptools import setup, find_packages
import os

if os.path.exists("requirements.txt"):
    with open("requirements.txt", "r") as fb:
        requirements = fb.readlines()
else:
    requirements = [
        "torch>=2.0.0",
        "torchvision>=0.15.1",
        "numpy<=1.23",
        "scikit-learn>=0.24.2",
        "scipy>=1.7.1",
#        "h5py>=3.3.0",
        "pandas>=1.3.4",
#        "boto3>=1.22.10",
#        "botocore>=1.25.10",
        "opencv-python>=4.5.5",
        "black>=19.10b0",
#        "av>=9.2.0",
#        "xarray>=2022.3.0",
#        "ipdb>=0.13.9",
#        "vispy>=0.11.10",
#        "trimesh>=3.15.8",
#        "gdown>=4.4.0",
        "xformers>=0.0.18",
        "ptutils @ git+https://git@github.com/anayebi/ptutils.git@a0512026013725522ec7af2a68e72e3e28de5325",
        "brainmodel_utils @ git+https://git@github.com/anayebi/brainmodel_utils.git@231ebf2de33b3a6a6a279be540278fb7c892d99a",
        "clip @ git+https://github.com/openai/CLIP.git@a9b1bf5920416aaeaec965c25dd9e8f98c864f16",
        "r3m @ git+https://github.com/facebookresearch/r3m.git@b2334e726887fa0206962d7984c69c5fb09cceab",
        "vip @ git+https://github.com/facebookresearch/vip.git@781fa2f54a268b9f63d30256060ed15eef1b3539",
        "vc_models @ git+https://github.com/facebookresearch/eai-vc.git@f58f69279556388aec0f1232dbde69eacc87c0ea#subdirectory=vc_models",
        "timm",
    ]

print(find_packages())
setup(
    name="mpmodels",
    version="0.1",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.6",
    # metadata to display on PyPI
    description="Models of Mental Simulation",
    # could also include long_description, download_url, etc.
)
