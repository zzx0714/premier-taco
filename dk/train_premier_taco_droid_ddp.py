import json
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import utils
from logger_offline import Logger
from replay_buffer_droid import DroidReplayBuffer
from omegaconf import OmegaConf

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec
    cfg.action_shapes = action_spec
    cfg.device = cfg.device if hasattr(cfg, 'device') else 'cuda'
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        self.cfg = cfg

        # DDP setup
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            dist.init_process_group('nccl')
            self.rank = int(os.environ['RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.device = f'cuda:{self.rank}'
            torch.cuda.set_device(self.device)
        else:
            self.rank = 0
            self.world_size = 1
            self.device = cfg.device

        if self.rank == 0:
            print(f'workspace: {self.work_dir}')
            print(f'DDP: rank={self.rank}, world_size={self.world_size}')

        utils.set_seed_everywhere(cfg.seed + self.rank)
        self.start_epoch = 1

        self.agent = make_agent(
            (3, cfg.image_size, cfg.image_size),
            cfg.action_dim,
            self.cfg.agent,
        )
        self.timer = utils.Timer()
        self._global_step = 0

        if self.rank == 0:
            self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb, offline=True)

        split_path = self.cfg.split_file
        with open(split_path, 'r') as f:
            splits = json.load(f)

        self.train_buffer = DroidReplayBuffer(
            data_dir=self.cfg.offline_data_dir,
            episode_indices=splits['train'],
            nstep=self.cfg.nstep,
            window_size=self.cfg.window_size,
            num_chunks=self.cfg.num_chunks,
            steps_per_chunk=self.cfg.steps_per_chunk,
        )

        if self.rank == 0:
            self.val_buffer = DroidReplayBuffer(
                data_dir=self.cfg.offline_data_dir,
                episode_indices=splits['val'],
                nstep=self.cfg.nstep,
                window_size=self.cfg.window_size,
                num_chunks=1,
                steps_per_chunk=self.cfg.val_steps + 1,
            )

        # Resume from checkpoint if specified
        if hasattr(cfg, 'resume_from') and cfg.resume_from:
            self._resume(cfg.resume_from)

    def validate(self):
        if self.rank == 0:
            val_losses = []
            for _ in range(self.cfg.val_steps):
                obs, action_seq, next_obs, neg_next_obs = self.val_buffer.sample_batch(
                    self.cfg.batch_size, self.device)
                metrics = self.agent.validate_premiertaco(
                    obs, action_seq, next_obs, neg_next_obs)
                val_losses.append(metrics['val_loss'])
            val_loss = np.mean(val_losses)
        else:
            val_loss = 0.0

        # All ranks must participate in broadcast
        if self.world_size > 1:
            val_tensor = torch.tensor([val_loss], device=self.device)
            dist.broadcast(val_tensor, src=0)
            val_loss = val_tensor.item()

        return val_loss

    def _resume(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.agent.encoder.load_state_dict(ckpt['encoder'])
        self.agent.taco_opt.load_state_dict(ckpt['optimizer'])
        self.start_epoch = ckpt['epoch'] + 1
        self._global_step = ckpt['global_step']
        best_val = ckpt.get('best_val_loss', float('inf'))
        if self.rank == 0:
            print(f'Resumed from {ckpt_path}: epoch {ckpt["epoch"]}, '
                  f'global_step {self._global_step}, best_val_loss {best_val:.6f}')

    def train(self):
        best_val_loss = float('inf')

        for epoch in range(self.start_epoch, self.cfg.num_epochs + 1):
            epoch_losses = []
            t0 = self.timer.reset()

            for step in range(self.cfg.steps_per_epoch):
                self._global_step += 1
                obs, action_seq, next_obs, neg_next_obs = self.train_buffer.sample_batch(
                    self.cfg.batch_size, self.device)
                metrics = self.agent.update_premiertaco(
                    obs, action_seq, next_obs, neg_next_obs)
                epoch_losses.append(metrics['premier_taco_loss'])

                if self.rank == 0 and self._global_step % 100 == 0:
                    print(f'  step {self._global_step} | '
                          f'loss={metrics["premier_taco_loss"]:.6f}')

            # Aggregate train loss across ranks
            train_loss = np.mean(epoch_losses)
            if self.world_size > 1:
                loss_tensor = torch.tensor([train_loss], device=self.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                train_loss = loss_tensor.item()

            val_loss = self.validate()

            _, total_time = t0
            if self.rank == 0:
                print(f'[Epoch {epoch}/{self.cfg.num_epochs}] '
                      f'train_loss={train_loss:.6f} | '
                      f'val_loss={val_loss:.6f} | '
                      f'time={total_time:.1f}s')

            if self.rank == 0:
                self.logger.log_metrics({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'epoch': epoch,
                }, self._global_step, ty='train')

            if epoch % self.cfg.save_every_n_epochs == 0:
                self.save_checkpoint(f'epoch_{epoch}', best_val_loss)
                self.save_encoder_only(f'epoch_{epoch}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('best', best_val_loss)
                self.save_encoder_only('best')

        self.save_checkpoint(f'epoch_{self.cfg.num_epochs}', best_val_loss)
        self.save_encoder_only(f'epoch_{self.cfg.num_epochs}')
        if self.rank == 0:
            print(f'Training complete. Best val loss: {best_val_loss:.6f}')

        if self.world_size > 1:
            dist.destroy_process_group()

    def save_checkpoint(self, tag, best_val_loss=None):
        if self.rank != 0:
            return
        snapshot = self.work_dir / f'checkpoint_{tag}.pt'
        torch.save({
            'encoder': self.agent.encoder.state_dict(),
            'taco': self.agent.taco.state_dict(),
            'optimizer': self.agent.taco_opt.state_dict(),
            'epoch': self._get_current_epoch(),
            'global_step': self._global_step,
            'best_val_loss': best_val_loss if best_val_loss is not None else float('inf'),
        }, snapshot)
        print(f'  Saved: {snapshot}')

    def save_encoder_only(self, tag):
        if self.rank != 0:
            return
        snapshot = self.work_dir / f'encoder_{tag}.pt'
        payload = self.agent.encoder.state_dict()
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def _get_current_epoch(self):
        return self._global_step // self.cfg.steps_per_epoch


@hydra.main(config_path='cfgs', config_name='premier_taco_droid_config')
def main(cfg):
    if dist.is_initialized():
        pass  # already initialized in Workspace.__init__
    print(OmegaConf.to_yaml(cfg))
    workspace = Workspace(cfg)
    workspace.train()


if __name__ == '__main__':
    main()
