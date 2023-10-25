# Adapted from: https://github.com/edenton/svg/blob/master/train_svg_lp.py

import functools
import torch
import torch.nn as nn
import mpmodels.models.lstm as lstm_models

__all__ = [
    "SVG",
    "svg_physion_64x64",
    "svg_physion_128x128",
]


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# mathematically the same as the fitvid one, but slightly different in formulation
# keeping it the same to match svg, in case there are numerical differences
def kl_criterion(mu1, logvar1, mu2, logvar2, batch_size):
    sigma1 = logvar1.mul(0.5).exp()
    sigma2 = logvar2.mul(0.5).exp()
    kld = (
        torch.log(sigma2 / sigma1)
        + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / (2 * torch.exp(logvar2))
        - 1 / 2
    )
    return kld.sum() / batch_size


class SVG(nn.Module):
    def __init__(
        self,
        input_size: int = 3,
        input_width: int = 64,
        train: bool = True,
        z_dim: int = 10,
        g_dim: int = 128,
        n_past: int = 7,
        beta: float = 1e-4,
        rnn_size: int = 256,
        predictor_rnn_layers: int = 2,
        posterior_rnn_layers: int = 1,
        prior_rnn_layers: int = 1,
        last_frame_skip: bool = True,
        **kwargs
    ):
        super().__init__()
        self.layers = None
        self.training = train
        self.input_size = input_size
        self.input_width = input_width
        self.z_dim = z_dim
        self.g_dim = g_dim
        self.n_past = n_past
        self.beta = beta
        self.rnn_size = rnn_size
        self.predictor_rnn_layers = predictor_rnn_layers
        self.posterior_rnn_layers = posterior_rnn_layers
        self.prior_rnn_layers = prior_rnn_layers
        self.last_frame_skip = last_frame_skip

        assert self.input_width in [64, 128]
        if self.input_width == 64:
            import mpmodels.models.vgg_64 as model
        else:
            assert self.input_width == 128
            import mpmodels.models.vgg_128 as model

        self.encoder = model.encoder(
            dim=self.g_dim, nc=self.input_size, train=self.training
        )
        self.decoder = model.decoder(
            dim=self.g_dim, nc=self.input_size, train=self.training
        )
        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)

        self.frame_predictor = lstm_models.lstm(
            input_size=self.g_dim + self.z_dim,
            output_size=self.g_dim,
            hidden_size=self.rnn_size,
            n_layers=self.predictor_rnn_layers,
            train=self.training,
        )
        self.posterior = lstm_models.gaussian_lstm(
            input_size=self.g_dim,
            output_size=self.z_dim,
            hidden_size=self.rnn_size,
            n_layers=self.posterior_rnn_layers,
            train=self.training,
        )
        self.prior = lstm_models.gaussian_lstm(
            input_size=self.g_dim,
            output_size=self.z_dim,
            hidden_size=self.rnn_size,
            n_layers=self.prior_rnn_layers,
            train=self.training,
        )
        self.frame_predictor.apply(init_weights)
        self.posterior.apply(init_weights)
        self.prior.apply(init_weights)

    def forward(self, video, actions=None):
        self.layers = dict()

        assert actions is None  # action conditioning not supported
        assert len(video.shape) == 5
        self.B, self.T = video.shape[:2]
        if video.dtype == torch.uint8:
            video = video.to(torch.float32) / 255.0
        kl = functools.partial(kl_criterion, batch_size=self.B)

        # initialize the hidden state.
        self.frame_predictor.hidden = self.frame_predictor.init_hidden(
            batch_size=self.B
        )
        self.posterior.hidden = self.posterior.init_hidden(batch_size=self.B)
        self.prior.hidden = self.prior.init_hidden(batch_size=self.B)

        encs = []
        h_preds = []
        preds = []
        if self.training:
            mse = 0
            kld = 0
            for i in range(1, self.T):
                h = self.encoder(video[:, i - 1])
                # note that encoder returns skips, as element 1
                h_target = self.encoder(video[:, i])[0]
                if self.last_frame_skip or i < self.n_past:
                    h, skip = h
                else:
                    h = h[0]
                encs.append(h)
                if i == self.T - 1:
                    # get output of encoder at final timestep T
                    encs.append(h_target)
                z_t, mu, logvar = self.posterior(h_target)
                _, mu_p, logvar_p = self.prior(h)
                h_pred = self.frame_predictor(torch.cat([h, z_t], 1))
                h_preds.append(h_pred)
                x_pred = self.decoder([h_pred, skip])
                preds.append(x_pred)
                mse += nn.MSELoss()(x_pred, video[:, i])
                kld += kl(mu1=mu, logvar1=logvar, mu2=mu_p, logvar2=logvar_p)

            loss = mse + kld * self.beta
        else:
            loss = 0
            x_in = video[:, 0]
            for i in range(1, self.T):
                h = self.encoder(x_in)
                if self.last_frame_skip or i < self.n_past:
                    h, skip = h
                else:
                    h, _ = h
                h = h.detach()
                if i < self.n_past:
                    encs.append(h)
                    h_target = self.encoder(video[:, i])[0].detach()
                    z_t, _, _ = self.posterior(h_target)
                    self.prior(h)
                    self.frame_predictor(torch.cat([h, z_t], 1))
                    x_in = video[:, i]
                else:
                    encs.append(h)
                    z_t, _, _ = self.prior(h)
                    h = self.frame_predictor(torch.cat([h, z_t], 1)).detach()
                    x_in = self.decoder([h, skip]).detach()

                h_preds.append(h)
                preds.append(x_in)
                # loss function is only reconstruction in this case
                loss += nn.MSELoss()(x_in, video[:, i])

        # get encoder output for last timestep T
        encs.append(self.encoder(x_in)[0])
        encs = torch.stack(encs, axis=1)
        assert encs.shape[1] == self.T
        self.layers["encoder"] = encs
        h_preds = torch.stack(h_preds, axis=1)
        preds = torch.stack(preds, axis=1)

        self.layers["dynamics"] = h_preds
        self.layers["decoder"] = preds

        # note: they do not divide the loss in time when optimizing, so I won't either
        return {
            "loss": loss,
        }


def svg_physion_64x64(**kwargs):
    return SVG(input_width=64, **kwargs)


def svg_physion_128x128(**kwargs):
    return SVG(input_width=128, **kwargs)
