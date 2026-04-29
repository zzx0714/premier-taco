import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import utils
from torch.nn.parallel import DistributedDataParallel as DDP

import torchvision.models as models


### Data Augmentation Module
class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class ResNetEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim):
        super().__init__()
        assert len(obs_shape) == 3

        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(obs_shape[0], 64, 7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.trunk = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh()
        )
        self.repr_dim = feature_dim

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.backbone(obs)
        h = h.view(h.shape[0], -1)
        return self.trunk(h)


class PremierTACO(nn.Module):
    """
    Premier TACO Module (DROID adaptation)
    action_dim = 7, nstep = 3
    """

    def __init__(self, repr_dim, feature_dim, action_shapes, hidden_dim, encoder, nstep, device):
        super(PremierTACO, self).__init__()

        self.nstep = nstep
        self.encoder = encoder
        self.device = device

        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_shapes)
        )

        self.proj_sa = nn.Sequential(
            nn.Linear(feature_dim + action_shapes * self.nstep, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim)
        )

        self.proj_aseq = nn.Sequential(
            nn.Linear(action_shapes * self.nstep,
                      action_shapes * self.nstep),
            nn.LayerNorm(action_shapes * self.nstep),
            nn.Tanh()
        )

        self.proj_s = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                    nn.LayerNorm(feature_dim),
                                    nn.Tanh())

        self.W = nn.Parameter(torch.rand(feature_dim, feature_dim))
        self.apply(utils.weight_init)

    def encode(self, x, ema=False):
        if ema:
            with torch.no_grad():
                z_out = self.proj_s(self.encoder(x))
        else:
            z_out = self.proj_s(self.encoder(x))
        return z_out

    def project_sa(self, s, a):
        x = torch.concat([s, a], dim=-1)
        return self.proj_sa(x)

    def compute_logits(self, z_a, z_pos, ):
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits


def _compute_loss(taco, aug, obs, action_seq, next_obs, neg_next_obs, device):
    """Compute Premier-TACO loss (shared by train and val)."""
    # Convert (B, H, W, C) → (B, C, H, W)
    obs = obs.permute(0, 3, 1, 2)
    next_obs = next_obs.permute(0, 3, 1, 2)
    neg_next_obs = neg_next_obs.permute(0, 3, 1, 2)

    obs_anchor = aug(obs.float())
    z_a = taco.encode(obs_anchor)
    action_seq_en = taco.proj_aseq(action_seq)
    curr_za = taco.project_sa(z_a, action_seq_en)

    batch_size = next_obs.shape[0]
    next_obs = torch.concat([next_obs, neg_next_obs], axis=0)
    next_z = taco.encode(aug(next_obs.float()), ema=True)
    next_z_pos = next_z[:batch_size]
    next_z_neg = next_z[batch_size:]

    logits_pos = torch.sum(next_z_pos * curr_za,
                           dim=-1, keepdim=True)
    logits_neg = torch.sum(next_z_neg * curr_za,
                           dim=-1, keepdim=True)
    logits = torch.concat([logits_pos, logits_neg], dim=0)
    logits = torch.concat([logits,
                           torch.zeros(2 * batch_size, 1).to(device)
                           ], dim=-1)
    labels = torch.concat([torch.zeros(batch_size).long(),
                           torch.ones(batch_size).long()]).to(device)
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss


class PremierTACORepresentation:
    def __init__(self, obs_shape, action_shapes, device, lr, feature_dim,
                 hidden_dim, nstep):
        self.device = device
        self.nstep = nstep
        self.encoder = ResNetEncoder(obs_shape, feature_dim).to(device)

        taco_module = PremierTACO(self.encoder.repr_dim, feature_dim, action_shapes, hidden_dim, self.encoder, nstep, device).to(device)
        if dist.is_initialized():
            self.TACO = DDP(taco_module)
        else:
            self.TACO = taco_module

        self.taco_opt = torch.optim.Adam(self.TACO.parameters(), lr=lr)
        self.aug = RandomShiftsAug(pad=4)

        self.train()

    @property
    def taco(self):
        return self.TACO.module if isinstance(self.TACO, DDP) else self.TACO

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.TACO.train(training)

    def update_premiertaco(self, obs, action_seq, next_obs, neg_next_obs):
        loss = _compute_loss(self.taco, self.aug, obs, action_seq,
                             next_obs, neg_next_obs, self.device)
        self.taco_opt.zero_grad()
        loss.backward()
        self.taco_opt.step()
        return {'premier_taco_loss': loss.item()}

    def validate_premiertaco(self, obs, action_seq, next_obs, neg_next_obs):
        with torch.no_grad():
            loss = _compute_loss(self.taco, self.aug, obs, action_seq,
                                 next_obs, neg_next_obs, self.device)
        return {'val_loss': loss.item()}

    def update(self, replay_iter, step):
        batch = next(replay_iter)
        obs, action_seq, r_next_obs, neg_next_obs = utils.to_torch(
            batch, self.device)
        return self.update_premiertaco(obs, action_seq, r_next_obs, neg_next_obs)
