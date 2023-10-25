# Adapted from: https://github.com/tkipf/c-swm/blob/eb02b6f0c5f8314ea2e810604f216e32495b82bf/modules.py

import numpy as np

import torch
from torch import nn

__all__ = [
    "small_CSWM_physion",
    "medium_CSWM_physion",
    "large_CSWM_physion",
]


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def get_act_fn(act_fn):
    if act_fn == "relu":
        return nn.ReLU()
    elif act_fn == "leaky_relu":
        return nn.LeakyReLU()
    elif act_fn == "elu":
        return nn.ELU()
    elif act_fn == "sigmoid":
        return nn.Sigmoid()
    elif act_fn == "softplus":
        return nn.Softplus()
    else:
        raise ValueError("Invalid argument for `act_fn`.")


def unsorted_segment_sum(tensor, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, tensor.size(1))
    result = tensor.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, tensor.size(1))
    result.scatter_add_(0, segment_ids, tensor)
    return result


def to_one_hot(indices, max_index):
    """Get one-hot encoding of index tensors."""
    zeros = torch.zeros(
        indices.size()[0], max_index, dtype=torch.float32, device=indices.device
    )
    return zeros.scatter_(1, indices.unsqueeze(1), 1)


class PassiveVideoCSWM(nn.Module):
    """
    Ignores actions and does a rolling temporal window to the encoder channels
    to enforce temporal dependency (since the Transition GNN of CSWM is purely spatial --
    there is no hidden state propagated through time).

    Best suited for learning a world model through passive viewing of videos.
    """

    def __init__(self, n_past=7, full_rollout=False, **kwargs):
        super(PassiveVideoCSWM, self).__init__()

        self.layers = None
        self.full_rollout = full_rollout
        self.n_past = n_past
        self.model = ContrastiveSWM(action_dim=0, ignore_action=True, **kwargs)
        self.model.apply(weights_init)

    def reshape_channels(self, x):
        assert len(x.shape) == 5
        assert x.shape[1] == self.n_past
        assert x.shape[2] == 3
        in_shape = list(x.shape)
        return x.reshape(in_shape[0], in_shape[1] * in_shape[2], *in_shape[3:])

    def forward(self, x):
        self.layers = dict()

        # x is (B, T, 3, H, W)
        assert len(x.shape) == 5
        assert x.shape[2] == 3
        assert self.model.num_channels == self.n_past * 3
        inputs = x[:, : self.n_past]
        obs = inputs

        if self.full_rollout:
            rollout_steps = x[:, self.n_past :].shape[1]
            assert rollout_steps > 0
        else:
            # the paper only does the loss for 1 timestep during training
            rollout_steps = 1

        loss = 0.0
        for step_idx in range(rollout_steps):
            step = step_idx + 1
            next_obs = x[:, step : self.n_past + step]
            loss += self.model.contrastive_loss(
                obs=self.reshape_channels(obs),
                action=0,
                next_obs=self.reshape_channels(next_obs),
            )
            obs = next_obs

        loss /= 1.0 * rollout_steps

        output = {"loss": loss}
        if self.full_rollout:
            # reserving this for evaluation just for speed up during training
            curr_obs = self.reshape_channels(inputs)
            state = self.model.obj_encoder(self.model.obj_extractor(curr_obs))
            pred_state = state  # initialize predicted state to input state
            input_states = [torch.flatten(state, start_dim=1)]
            observed_states = []
            simulated_states = []
            for step_idx in range(rollout_steps):
                step = step_idx + 1
                curr_obs = self.reshape_channels(x[:, step : self.n_past + step])
                obs_state = self.model.obj_encoder(self.model.obj_extractor(curr_obs))
                observed_states.append(torch.flatten(obs_state, start_dim=1))
                pred_trans = self.model.transition_model(states=pred_state, action=0)
                pred_state = pred_state + pred_trans
                simulated_states.append(torch.flatten(pred_state, start_dim=1))

            input_states = torch.stack(input_states, axis=1)
            output["input_states"] = input_states
            observed_states = torch.stack(observed_states, axis=1)
            output["observed_states"] = observed_states
            simulated_states = torch.stack(simulated_states, axis=1)
            output["simulated_states"] = simulated_states
            assert observed_states.shape == simulated_states.shape
            self.layers["encoder"] = torch.cat([input_states, observed_states], axis=1)
            self.layers["dynamics"] = torch.cat(
                [input_states, simulated_states], axis=1
            )
        return output


class ContrastiveSWM(nn.Module):
    """Main module for a Contrastively-trained Structured World Model (C-SWM).

    Args:
        embedding_dim: Dimensionality of abstract state space.
        input_dims: Shape of input observation.
        hidden_dim: Number of hidden units in encoder and transition model.
        action_dim: Dimensionality of action space.
        num_objects: Number of object slots.
    """

    def __init__(
        self,
        embedding_dim,
        input_dims,
        hidden_dim,
        action_dim,
        num_objects,
        hinge=1.0,
        sigma=0.5,
        encoder="large",
        ignore_action=False,
        copy_action=False,
    ):
        super(ContrastiveSWM, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.num_objects = num_objects
        self.hinge = hinge
        self.sigma = sigma
        self.ignore_action = ignore_action
        self.copy_action = copy_action

        self.pos_loss = 0
        self.neg_loss = 0

        self.num_channels = input_dims[0]
        width_height = input_dims[1:]

        if encoder == "small":
            self.obj_extractor = EncoderCNNSmall(
                input_dim=self.num_channels,
                hidden_dim=hidden_dim // 16,
                num_objects=num_objects,
            )
            # CNN image size changes
            width_height = np.array(width_height)
            width_height = width_height // 10
        elif encoder == "medium":
            self.obj_extractor = EncoderCNNMedium(
                input_dim=self.num_channels,
                hidden_dim=hidden_dim // 16,
                num_objects=num_objects,
            )
            # CNN image size changes
            width_height = np.array(width_height)
            width_height = width_height // 5
        elif encoder == "large":
            self.obj_extractor = EncoderCNNLarge(
                input_dim=self.num_channels,
                hidden_dim=hidden_dim // 16,
                num_objects=num_objects,
            )

        self.obj_encoder = EncoderMLP(
            input_dim=np.prod(width_height),
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_objects=num_objects,
        )

        self.transition_model = TransitionGNN(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            num_objects=num_objects,
            ignore_action=ignore_action,
            copy_action=copy_action,
        )

        self.width = width_height[0]
        self.height = width_height[1]

    def energy(self, state, action, next_state, no_trans=False):
        """Energy function based on normalized squared L2 norm."""

        norm = 0.5 / (self.sigma ** 2)

        if no_trans:
            diff = state - next_state
        else:
            pred_trans = self.transition_model(state, action)
            diff = state + pred_trans - next_state

        return norm * diff.pow(2).sum(2).mean(1)

    def transition_loss(self, state, action, next_state):
        return self.energy(state, action, next_state).mean()

    def contrastive_loss(self, obs, action, next_obs):

        objs = self.obj_extractor(obs)
        next_objs = self.obj_extractor(next_obs)

        state = self.obj_encoder(objs)
        next_state = self.obj_encoder(next_objs)

        # Sample negative state across episodes at random
        batch_size = state.size(0)
        perm = np.random.permutation(batch_size)
        neg_state = state[perm]

        self.pos_loss = self.energy(state, action, next_state)
        zeros = torch.zeros_like(self.pos_loss)

        self.pos_loss = self.pos_loss.mean()
        self.neg_loss = torch.max(
            zeros, self.hinge - self.energy(state, action, neg_state, no_trans=True)
        ).mean()

        loss = self.pos_loss + self.neg_loss

        return loss

    def forward(self, obs):
        return self.obj_encoder(self.obj_extractor(obs))


class TransitionGNN(torch.nn.Module):
    """GNN-based transition function."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        action_dim,
        num_objects,
        ignore_action=False,
        copy_action=False,
        act_fn="relu",
    ):
        super(TransitionGNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_objects = num_objects
        self.ignore_action = ignore_action
        self.copy_action = copy_action

        if self.ignore_action:
            self.action_dim = 0
        else:
            self.action_dim = action_dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim),
        )

        node_input_dim = hidden_dim + input_dim + self.action_dim

        self.node_mlp = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            get_act_fn(act_fn),
            nn.Linear(hidden_dim, input_dim),
        )

        self.edge_list = None
        self.batch_size = 0

    def _edge_model(self, source, target, edge_attr):
        del edge_attr  # Unused.
        out = torch.cat([source, target], dim=1)
        return self.edge_mlp(out)

    def _node_model(self, node_attr, edge_index, edge_attr):
        if edge_attr is not None:
            row, col = edge_index
            agg = unsorted_segment_sum(edge_attr, row, num_segments=node_attr.size(0))
            out = torch.cat([node_attr, agg], dim=1)
        else:
            out = node_attr
        return self.node_mlp(out)

    def _get_edge_list_fully_connected(self, batch_size, num_objects, cuda):
        # Only re-evaluate if necessary (e.g. if batch size changed).
        if self.edge_list is None or self.batch_size != batch_size:
            self.batch_size = batch_size

            # Create fully-connected adjacency matrix for single sample.
            adj_full = torch.ones(num_objects, num_objects)

            # Remove diagonal.
            adj_full -= torch.eye(num_objects)
            self.edge_list = adj_full.nonzero()

            # Copy `batch_size` times and add offset.
            self.edge_list = self.edge_list.repeat(batch_size, 1)
            offset = torch.arange(0, batch_size * num_objects, num_objects).unsqueeze(
                -1
            )
            offset = offset.expand(batch_size, num_objects * (num_objects - 1))
            offset = offset.contiguous().view(-1)
            self.edge_list += offset.unsqueeze(-1)

            # Transpose to COO format -> Shape: [2, num_edges].
            self.edge_list = self.edge_list.transpose(0, 1)

            if cuda:
                self.edge_list = self.edge_list.cuda()

        return self.edge_list

    def forward(self, states, action):

        cuda = states.is_cuda
        batch_size = states.size(0)
        num_nodes = states.size(1)

        # states: [batch_size (B), num_objects, embedding_dim]
        # node_attr: Flatten states tensor to [B * num_objects, embedding_dim]
        node_attr = states.view(-1, self.input_dim)

        edge_attr = None
        edge_index = None

        if num_nodes > 1:
            # edge_index: [B * (num_objects*[num_objects-1]), 2] edge list
            edge_index = self._get_edge_list_fully_connected(
                batch_size, num_nodes, cuda
            )

            row, col = edge_index
            edge_attr = self._edge_model(node_attr[row], node_attr[col], edge_attr)

        if not self.ignore_action:

            if self.copy_action:
                action_vec = to_one_hot(action, self.action_dim).repeat(
                    1, self.num_objects
                )
                action_vec = action_vec.view(-1, self.action_dim)
            else:
                action_vec = to_one_hot(action, self.action_dim * num_nodes)
                action_vec = action_vec.view(-1, self.action_dim)

            # Attach action to each state
            node_attr = torch.cat([node_attr, action_vec], dim=-1)

        node_attr = self._node_model(node_attr, edge_index, edge_attr)

        # [batch_size, num_nodes, hidden_dim]
        return node_attr.view(batch_size, num_nodes, -1)


class EncoderCNNSmall(nn.Module):
    """CNN encoder, maps observation to obj-specific feature maps."""

    def __init__(
        self, input_dim, hidden_dim, num_objects, act_fn="sigmoid", act_fn_hid="relu"
    ):
        super(EncoderCNNSmall, self).__init__()
        self.cnn1 = nn.Conv2d(input_dim, hidden_dim, (10, 10), stride=10)
        self.cnn2 = nn.Conv2d(hidden_dim, num_objects, (1, 1), stride=1)
        self.ln1 = nn.BatchNorm2d(hidden_dim)
        self.act1 = get_act_fn(act_fn_hid)
        self.act2 = get_act_fn(act_fn)

    def forward(self, obs):
        h = self.act1(self.ln1(self.cnn1(obs)))
        return self.act2(self.cnn2(h))


class EncoderCNNMedium(nn.Module):
    """CNN encoder, maps observation to obj-specific feature maps."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_objects,
        act_fn="sigmoid",
        act_fn_hid="leaky_relu",
    ):
        super(EncoderCNNMedium, self).__init__()

        self.cnn1 = nn.Conv2d(input_dim, hidden_dim, (9, 9), padding=4)
        self.act1 = get_act_fn(act_fn_hid)
        self.ln1 = nn.BatchNorm2d(hidden_dim)

        self.cnn2 = nn.Conv2d(hidden_dim, num_objects, (5, 5), stride=5)
        self.act2 = get_act_fn(act_fn)

    def forward(self, obs):
        h = self.act1(self.ln1(self.cnn1(obs)))
        h = self.act2(self.cnn2(h))
        return h


class EncoderCNNLarge(nn.Module):
    """CNN encoder, maps observation to obj-specific feature maps."""

    def __init__(
        self, input_dim, hidden_dim, num_objects, act_fn="sigmoid", act_fn_hid="relu"
    ):
        super(EncoderCNNLarge, self).__init__()

        self.cnn1 = nn.Conv2d(input_dim, hidden_dim, (3, 3), padding=1)
        self.act1 = get_act_fn(act_fn_hid)
        self.ln1 = nn.BatchNorm2d(hidden_dim)

        self.cnn2 = nn.Conv2d(hidden_dim, hidden_dim, (3, 3), padding=1)
        self.act2 = get_act_fn(act_fn_hid)
        self.ln2 = nn.BatchNorm2d(hidden_dim)

        self.cnn3 = nn.Conv2d(hidden_dim, hidden_dim, (3, 3), padding=1)
        self.act3 = get_act_fn(act_fn_hid)
        self.ln3 = nn.BatchNorm2d(hidden_dim)

        self.cnn4 = nn.Conv2d(hidden_dim, num_objects, (3, 3), padding=1)
        self.act4 = get_act_fn(act_fn)

    def forward(self, obs):
        h = self.act1(self.ln1(self.cnn1(obs)))
        h = self.act2(self.ln2(self.cnn2(h)))
        h = self.act3(self.ln3(self.cnn3(h)))
        return self.act4(self.cnn4(h))


class EncoderMLP(nn.Module):
    """MLP encoder, maps observation to latent state."""

    def __init__(self, input_dim, output_dim, hidden_dim, num_objects, act_fn="relu"):
        super(EncoderMLP, self).__init__()

        self.num_objects = num_objects
        self.input_dim = input_dim

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.ln = nn.LayerNorm(hidden_dim)

        self.act1 = get_act_fn(act_fn)
        self.act2 = get_act_fn(act_fn)

    def forward(self, ins):
        h_flat = ins.view(-1, self.num_objects, self.input_dim)
        h = self.act1(self.fc1(h_flat))
        h = self.act2(self.ln(self.fc2(h)))
        return self.fc3(h)


class DecoderMLP(nn.Module):
    """MLP decoder, maps latent state to image."""

    def __init__(self, input_dim, hidden_dim, num_objects, output_size, act_fn="relu"):
        super(DecoderMLP, self).__init__()

        self.fc1 = nn.Linear(input_dim + num_objects, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, np.prod(output_size))

        self.input_dim = input_dim
        self.num_objects = num_objects
        self.output_size = output_size

        self.act1 = get_act_fn(act_fn)
        self.act2 = get_act_fn(act_fn)

    def forward(self, ins):
        obj_ids = torch.arange(self.num_objects)
        obj_ids = to_one_hot(obj_ids, self.num_objects).unsqueeze(0)
        obj_ids = obj_ids.repeat((ins.size(0), 1, 1)).to(ins.get_device())

        h = torch.cat((ins, obj_ids), -1)
        h = self.act1(self.fc1(h))
        h = self.act2(self.fc2(h))
        h = self.fc3(h).sum(1)
        return h.view(-1, self.output_size[0], self.output_size[1], self.output_size[2])


class DecoderCNNSmall(nn.Module):
    """CNN decoder, maps latent state to image."""

    def __init__(self, input_dim, hidden_dim, num_objects, output_size, act_fn="relu"):
        super(DecoderCNNSmall, self).__init__()

        width, height = output_size[1] // 10, output_size[2] // 10

        output_dim = width * height

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.ln = nn.LayerNorm(hidden_dim)

        self.deconv1 = nn.ConvTranspose2d(
            num_objects, hidden_dim, kernel_size=1, stride=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            hidden_dim, output_size[0], kernel_size=10, stride=10
        )

        self.input_dim = input_dim
        self.num_objects = num_objects
        self.map_size = output_size[0], width, height

        self.act1 = get_act_fn(act_fn)
        self.act2 = get_act_fn(act_fn)
        self.act3 = get_act_fn(act_fn)

    def forward(self, ins):
        h = self.act1(self.fc1(ins))
        h = self.act2(self.ln(self.fc2(h)))
        h = self.fc3(h)

        h_conv = h.view(-1, self.num_objects, self.map_size[1], self.map_size[2])
        h = self.act3(self.deconv1(h_conv))
        return self.deconv2(h)


class DecoderCNNMedium(nn.Module):
    """CNN decoder, maps latent state to image."""

    def __init__(self, input_dim, hidden_dim, num_objects, output_size, act_fn="relu"):
        super(DecoderCNNMedium, self).__init__()

        width, height = output_size[1] // 5, output_size[2] // 5

        output_dim = width * height

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.ln = nn.LayerNorm(hidden_dim)

        self.deconv1 = nn.ConvTranspose2d(
            num_objects, hidden_dim, kernel_size=5, stride=5
        )
        self.deconv2 = nn.ConvTranspose2d(
            hidden_dim, output_size[0], kernel_size=9, padding=4
        )

        self.ln1 = nn.BatchNorm2d(hidden_dim)

        self.input_dim = input_dim
        self.num_objects = num_objects
        self.map_size = output_size[0], width, height

        self.act1 = get_act_fn(act_fn)
        self.act2 = get_act_fn(act_fn)
        self.act3 = get_act_fn(act_fn)

    def forward(self, ins):
        h = self.act1(self.fc1(ins))
        h = self.act2(self.ln(self.fc2(h)))
        h = self.fc3(h)

        h_conv = h.view(-1, self.num_objects, self.map_size[1], self.map_size[2])
        h = self.act3(self.ln1(self.deconv1(h_conv)))
        return self.deconv2(h)


class DecoderCNNLarge(nn.Module):
    """CNN decoder, maps latent state to image."""

    def __init__(self, input_dim, hidden_dim, num_objects, output_size, act_fn="relu"):
        super(DecoderCNNLarge, self).__init__()

        width, height = output_size[1], output_size[2]

        output_dim = width * height

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.ln = nn.LayerNorm(hidden_dim)

        self.deconv1 = nn.ConvTranspose2d(
            num_objects, hidden_dim, kernel_size=3, padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1
        )
        self.deconv3 = nn.ConvTranspose2d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1
        )
        self.deconv4 = nn.ConvTranspose2d(
            hidden_dim, output_size[0], kernel_size=3, padding=1
        )

        self.ln1 = nn.BatchNorm2d(hidden_dim)
        self.ln2 = nn.BatchNorm2d(hidden_dim)
        self.ln3 = nn.BatchNorm2d(hidden_dim)

        self.input_dim = input_dim
        self.num_objects = num_objects
        self.map_size = output_size[0], width, height

        self.act1 = get_act_fn(act_fn)
        self.act2 = get_act_fn(act_fn)
        self.act3 = get_act_fn(act_fn)
        self.act4 = get_act_fn(act_fn)
        self.act5 = get_act_fn(act_fn)

    def forward(self, ins):
        h = self.act1(self.fc1(ins))
        h = self.act2(self.ln(self.fc2(h)))
        h = self.fc3(h)

        h_conv = h.view(-1, self.num_objects, self.map_size[1], self.map_size[2])
        h = self.act3(self.ln1(self.deconv1(h_conv)))
        h = self.act4(self.ln1(self.deconv2(h)))
        h = self.act5(self.ln1(self.deconv3(h)))
        return self.deconv4(h)


def small_CSWM_physion(n_past=7, **kwargs):
    return PassiveVideoCSWM(
        n_past=n_past,
        encoder="small",
        input_dims=(n_past * 3, 224, 224),
        num_objects=10,
        hidden_dim=512,
        embedding_dim=128,
        **kwargs
    )


def medium_CSWM_physion(n_past=7, **kwargs):
    return PassiveVideoCSWM(
        n_past=n_past,
        encoder="medium",
        input_dims=(n_past * 3, 224, 224),
        num_objects=10,
        hidden_dim=512,
        embedding_dim=128,
        **kwargs
    )


def large_CSWM_physion(n_past=7, **kwargs):
    return PassiveVideoCSWM(
        n_past=n_past,
        encoder="large",
        input_dims=(n_past * 3, 224, 224),
        num_objects=10,
        hidden_dim=512,
        embedding_dim=128,
        **kwargs
    )
