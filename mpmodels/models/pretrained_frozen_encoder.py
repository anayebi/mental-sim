import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpmodels.models.custom_rnns import CTRNN as CTRNN_module

__all__ = [
    "pfDEIT_LSTM_physion",
    "pfVGG16_LSTM_physion",
    "pfCLIP_LSTM_physion",
    "pfDINO_LSTM_physion",
    "pfDINOv2_LSTM_physion",
    "pfResNet50_LSTM_physion",
    "pfR3M_LSTM_physion",
    "pfR3M_LSTM_k700",
    "pfVIP_LSTM_physion",
    "pfVIP_LSTM_k700",
    "pfVC1_LSTM_physion",
    "pfVC1_LSTM_k700",
    "pfDEIT_ID",
    "pfVGG16_ID",
    "pfCLIP_ID",
    "pfDINO_ID",
    "pfDINOv2_ID",
    "pfResNet50_ID",
    "pfR3M_ID",
    "pfVIP_ID",
    "pfVC1_ID",
    "pfDEIT_CTRNN_physion",
    "pfVGG16_CTRNN_physion",
    "pfCLIP_CTRNN_physion",
    "pfDINO_CTRNN_physion",
    "pfDINOv2_CTRNN_physion",
    "pfResNet50_CTRNN_physion",
    "pfR3M_CTRNN_physion",
    "pfR3M_CTRNN_k700",
    "pfVIP_CTRNN_physion",
    "pfVIP_CTRNN_k700",
    "pfVC1_CTRNN_physion",
    "pfVC1_CTRNN_k700",
]

# ---Encoders---
# Generates latent representation for an image
class DEIT_pretrained(nn.Module):
    def __init__(self):
        super(DEIT_pretrained, self).__init__()
        self.deit = torch.hub.load(
            "facebookresearch/deit:main", "deit_base_patch16_224", pretrained=True
        )
        # TODO: assumes layer norm is last layer and shape is int
        self.latent_dim = self.deit.norm.normalized_shape[0]
        self.deit.head = nn.Identity()  # hack to remove head

    def forward(self, x):
        return self.deit(x)


class VGG16_pretrained(nn.Module):
    def __init__(self):
        from torchvision.models import vgg16_bn

        super(VGG16_pretrained, self).__init__()
        self.vgg = vgg16_bn(pretrained=True)
        # get up to second fc w/o dropout
        self.vgg.classifier = nn.Sequential(
            *[self.vgg.classifier[i] for i in [0, 1, 3]]
        )
        self.latent_dim = list(self.vgg.modules())[-1].out_features

    def forward(self, x):
        return self.vgg(x)


class CLIP_pretrained(nn.Module):
    def __init__(self):
        import clip

        super().__init__()
        self.clip, _ = clip.load("ViT-B/32", jit=False)
        self.clip_vision = self.clip.encode_image
        self.latent_dim = self.clip.ln_final.normalized_shape[0]
        # 512
        assert self.latent_dim == 512

    def forward(self, x):
        return self.clip_vision(x).type(torch.float32)


class DINO_pretrained(nn.Module):
    def __init__(self, variant="dino_vits16"):
        super().__init__()
        self.dino = torch.hub.load("facebookresearch/dino:main", variant)
        self.latent_dim = self.dino.norm.normalized_shape[0]
        # 384 for vits16
        assert self.latent_dim == 384

    def forward(self, x):
        return self.dino(x)


class DINOv2_pretrained(nn.Module):
    def __init__(self, variant="dinov2_vitg14"):
        super().__init__()
        self.dinov2 = torch.hub.load("facebookresearch/dinov2", variant)
        self.latent_dim = self.dinov2.norm.normalized_shape[0]
        # 1536 for vitg14
        assert self.latent_dim == 1536

    def forward(self, x):
        return self.dinov2(x)


class ResNet50_pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torch.hub.load(
            "pytorch/vision:v0.8.2", "resnet50", pretrained=True
        )
        self.latent_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # remove final fc

    def forward(self, x):
        return self.resnet(x)


class R3M_pretrained(nn.Module):
    def __init__(self):
        from r3m import load_r3m

        super().__init__()
        self.r3m = load_r3m("resnet50")
        self.latent_dim = 2048  # resnet50 final fc in_features

    def forward(self, x):
        return self.r3m(x * 255.0)  # R3M expects image input to be [0-255]


class VIP_pretrained(nn.Module):
    def __init__(self):
        from vip import load_vip

        super().__init__()
        self.vip = load_vip()
        self.latent_dim = 1024

    def forward(self, x):
        return self.vip(x * 255.0)  # VIP expects image input to be [0-255]


class VC1_pretrained(nn.Module):
    def __init__(self):
        from vc_models.models.vit import model_utils

        super().__init__()
        # latent dim is 1024
        (
            self.vc1,
            self.latent_dim,
            model_transforms,
            self.model_info,
        ) = model_utils.load_model(model_utils.VC1_LARGE_NAME)

    def forward(self, x):
        return self.vc1(x)


# ---Dynamics---
# Given a sequence of latent representations, generates the next latent
class ID(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

    def forward(self, x):
        assert isinstance(x, list)
        return x[-1]  # just return last embedding


class MLP(nn.Module):
    def __init__(self, latent_dim, n_past):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_past = n_past
        self.regressor = nn.Sequential(
            nn.Linear(self.latent_dim * self.n_past, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.latent_dim),
        )

    def forward(self, x):
        feats = torch.cat(x, dim=-1)
        pred = self.regressor(feats)
        return pred


class LSTM(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.lstm = nn.LSTM(self.latent_dim, 1024)
        self.regressor = nn.Linear(1024, self.latent_dim)

    def forward(self, x):
        feats = torch.stack(x)  # (T, Bs, self.latent_dim)
        assert feats.ndim == 3
        # note: for lstms, hidden is the last timestep output
        _, hidden = self.lstm(feats)
        # assumes n_layers=1
        x = torch.squeeze(hidden[0].permute(1, 0, 2), dim=1)
        x = self.regressor(x)
        return x


class CTRNN(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.ctrnn = CTRNN_module(
            dim_input=self.latent_dim, dim_recurrent=1024, dim_output=1024
        )
        self.regressor = nn.Linear(1024, self.latent_dim)

    def forward(self, x):
        feats = torch.stack(x)  # (T, Bs, self.latent_dim)
        assert feats.ndim == 3
        # B, T, self.latent_dim
        feats = feats.permute(1, 0, 2)
        assert feats.ndim == 3
        output, _ = self.ctrnn(feats)
        # get last timestep output
        x = output[:, -1, :]
        x = self.regressor(x)
        return x


# ---Frozen Pretrained Encoder---
# Given sequence of images, predicts next latent
class FrozenPretrainedEncoder(nn.Module):
    def __init__(self, encoder, dynamics, n_past=7, full_rollout=False):
        super().__init__()

        self.layers = None
        self.full_rollout = full_rollout
        self.n_past = n_past
        self.encoder_name = encoder.lower()
        self.dynamics_name = dynamics.lower()
        Encoder = _get_encoder(self.encoder_name)
        self.encoder = Encoder()

        Dynamics = _get_dynamics(self.dynamics_name)
        dynamics_kwargs = {"latent_dim": self.encoder.latent_dim}
        if self.dynamics_name == "mlp":
            dynamics_kwargs["n_past"] = self.n_past
        self.dynamics = Dynamics(**dynamics_kwargs)

    def forward(self, x):
        self.layers = dict()

        # set frozen pretrained encoder to eval mode
        self.encoder.eval()
        # x is (Bs, T, 3, H, W)
        assert len(x.shape) == 5
        inputs = x[:, : self.n_past]
        input_states = self.get_encoder_feats(inputs)

        if self.full_rollout:
            # roll out the entire trajectory
            label_images = x[:, self.n_past :]
            rollout_steps = label_images.shape[1]
            assert rollout_steps > 0
        else:
            label_images = x[:, self.n_past]
            rollout_steps = 1

        observed_states = self.get_encoder_feats(label_images)
        simulated_states = []
        # to avoid overwriting input_states, since we want to preserve those for saving later
        prev_states = [v.clone() for v in input_states]
        for step in range(rollout_steps):
            # dynamics model predicts next latent from past latents
            pred_state = self.dynamics(prev_states)
            simulated_states.append(pred_state)
            # add most recent pred and delete oldest
            # (to maintain a temporal window of length n_past)
            prev_states.append(pred_state)
            prev_states.pop(0)

        input_states = torch.stack(input_states, axis=1)
        observed_states = torch.stack(observed_states, axis=1)
        simulated_states = torch.stack(simulated_states, axis=1)
        assert observed_states.shape == simulated_states.shape

        loss = nn.MSELoss()(simulated_states, observed_states)
        output = {
            "input_states": input_states,
            "observed_states": observed_states,
            "simulated_states": simulated_states,
            "loss": loss,
        }
        self.layers["encoder"] = torch.cat([input_states, observed_states], axis=1)
        self.layers["dynamics"] = torch.cat([input_states, simulated_states], axis=1)
        if self.full_rollout:
            # adding this one as a visualizable sanity check of feature extractor
            self.layers["inputs_test"] = torch.cat([inputs, label_images], axis=1)
            assert self.layers["inputs_test"].shape == x.shape
            assert np.array_equal(
                self.layers["inputs_test"].cpu().numpy(), x.cpu().numpy()
            )
            # should be matched in B and T dimensions
            assert (
                self.layers["dynamics"].shape[0] == self.layers["inputs_test"].shape[0]
            )
            assert (
                self.layers["dynamics"].shape[1] == self.layers["inputs_test"].shape[1]
            )
            assert self.layers["dynamics"].ndim >= 3
        return output

    def get_encoder_feats(self, x):
        # applies encoder to each image in x: (Bs, T, 3, H, W) or (Bs, 3, H, W)
        with torch.no_grad():  # TODO: best place to put this?
            if x.ndim == 4:  # (Bs, 3, H, W)
                feats = [self._extract_feats(x)]
            else:
                assert x.ndim == 5, "Expected input to be of shape (Bs, T, 3, H, W)"
                feats = []
                for _x in torch.split(x, 1, dim=1):
                    _x = torch.squeeze(
                        _x, dim=1
                    )  # _x is shape (Bs, 1, 3, H, W) => (Bs, 3, H, W) TODO: put this in _extract_feats?
                    feats.append(self._extract_feats(_x))
        return feats

    def _extract_feats(self, x):
        feats = torch.flatten(self.encoder(x), start_dim=1)  # (Bs, -1)
        return feats


# ---Utils---
def _get_encoder(encoder):
    if encoder == "vgg16":
        return VGG16_pretrained
    elif encoder == "deit":
        return DEIT_pretrained
    elif encoder == "clip":
        return CLIP_pretrained
    elif encoder == "dino":
        return DINO_pretrained
    elif encoder == "dinov2":
        return DINOv2_pretrained
    elif encoder == "resnet50":
        return ResNet50_pretrained
    elif encoder == "r3m":
        return R3M_pretrained
    elif encoder == "vip":
        return VIP_pretrained
    elif encoder == "vc1":
        return VC1_pretrained
    else:
        raise NotImplementedError(encoder)


def _get_dynamics(dynamics):
    if dynamics == "id":
        return ID
    elif dynamics == "mlp":
        return MLP
    elif dynamics == "lstm":
        return LSTM
    elif dynamics == "ctrnn":
        return CTRNN
    else:
        raise NotImplementedError(dynamics)


def pfDEIT_LSTM_physion(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="deit", dynamics="lstm", n_past=n_past, **kwargs
    )


def pfVGG16_LSTM_physion(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="vgg16", dynamics="lstm", n_past=n_past, **kwargs
    )


def pfCLIP_LSTM_physion(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="clip", dynamics="lstm", n_past=n_past, **kwargs
    )


def pfDINO_LSTM_physion(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="dino", dynamics="lstm", n_past=n_past, **kwargs
    )


def pfDINOv2_LSTM_physion(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="dinov2", dynamics="lstm", n_past=n_past, **kwargs
    )


def pfResNet50_LSTM_physion(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="resnet50", dynamics="lstm", n_past=n_past, **kwargs
    )


def pfR3M_LSTM_physion(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="r3m", dynamics="lstm", n_past=n_past, **kwargs
    )


def pfR3M_LSTM_k700(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="r3m", dynamics="lstm", n_past=n_past, **kwargs
    )


def pfVIP_LSTM_physion(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="vip", dynamics="lstm", n_past=n_past, **kwargs
    )


def pfVIP_LSTM_k700(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="vip", dynamics="lstm", n_past=n_past, **kwargs
    )


def pfVC1_LSTM_physion(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="vc1", dynamics="lstm", n_past=n_past, **kwargs
    )


def pfVC1_LSTM_k700(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="vc1", dynamics="lstm", n_past=n_past, **kwargs
    )


def pfDEIT_ID(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="deit", dynamics="id", n_past=n_past, **kwargs
    )


def pfVGG16_ID(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="vgg16", dynamics="id", n_past=n_past, **kwargs
    )


def pfCLIP_ID(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="clip", dynamics="id", n_past=n_past, **kwargs
    )


def pfDINO_ID(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="dino", dynamics="id", n_past=n_past, **kwargs
    )


def pfDINOv2_ID(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="dinov2", dynamics="id", n_past=n_past, **kwargs
    )


def pfResNet50_ID(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="resnet50", dynamics="id", n_past=n_past, **kwargs
    )


def pfR3M_ID(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="r3m", dynamics="id", n_past=n_past, **kwargs
    )


def pfVIP_ID(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="vip", dynamics="id", n_past=n_past, **kwargs
    )


def pfVC1_ID(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="vc1", dynamics="id", n_past=n_past, **kwargs
    )


def pfDEIT_CTRNN_physion(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="deit", dynamics="ctrnn", n_past=n_past, **kwargs
    )


def pfVGG16_CTRNN_physion(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="vgg16", dynamics="ctrnn", n_past=n_past, **kwargs
    )


def pfCLIP_CTRNN_physion(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="clip", dynamics="ctrnn", n_past=n_past, **kwargs
    )


def pfDINO_CTRNN_physion(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="dino", dynamics="ctrnn", n_past=n_past, **kwargs
    )


def pfDINOv2_CTRNN_physion(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="dinov2", dynamics="ctrnn", n_past=n_past, **kwargs
    )


def pfResNet50_CTRNN_physion(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="resnet50", dynamics="ctrnn", n_past=n_past, **kwargs
    )


def pfR3M_CTRNN_physion(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="r3m", dynamics="ctrnn", n_past=n_past, **kwargs
    )


def pfR3M_CTRNN_k700(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="r3m", dynamics="ctrnn", n_past=n_past, **kwargs
    )


def pfVIP_CTRNN_physion(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="vip", dynamics="ctrnn", n_past=n_past, **kwargs
    )


def pfVIP_CTRNN_k700(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="vip", dynamics="ctrnn", n_past=n_past, **kwargs
    )


def pfVC1_CTRNN_physion(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="vc1", dynamics="ctrnn", n_past=n_past, **kwargs
    )


def pfVC1_CTRNN_k700(n_past=7, **kwargs):
    return FrozenPretrainedEncoder(
        encoder="vc1", dynamics="ctrnn", n_past=n_past, **kwargs
    )
