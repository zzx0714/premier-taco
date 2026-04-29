import gc
import random
from pathlib import Path

import numpy as np
import torch


CACHE_DIR_NAME = 'premier_taco_cache'


class DroidEpisode:
    __slots__ = ['images', 'actions', 'num_strided']

    def __init__(self, images, actions):
        self.images = images          # (num_strided, H, W, 3) uint8 — already a copy
        self.actions = actions        # (num_strided, 7) float32
        self.num_strided = len(images)

    def release(self):
        self.images = None
        self.actions = None


class DroidReplayBuffer:
    """Rotating replay buffer — loads preprocessed .npz from disk cache.

    Memory: only one chunk of episodes in memory at a time.
    """

    def __init__(self, data_dir, episode_indices, nstep, window_size,
                 num_chunks=24, steps_per_chunk=750):
        self._data_dir = data_dir
        self._cache_dir = Path(data_dir) / CACHE_DIR_NAME
        self._nstep = nstep
        self._window_size = window_size
        self._steps_per_chunk = steps_per_chunk
        self._steps_in_current_chunk = 0

        # Filter to episodes that have cache files
        valid = []
        for ep_idx in episode_indices:
            cache = self._cache_dir / f'ep_{int(ep_idx):06d}_s5_r256.npz'
            if cache.exists():
                valid.append(int(ep_idx))
        print(f'[ReplayBuffer] {len(valid)}/{len(episode_indices)} episodes have cache.')

        random.shuffle(valid)
        self._all_chunks = np.array_split(valid, num_chunks)
        self._num_chunks = len(self._all_chunks)
        self._current_chunk_idx = 0

        self._episodes = []

        self._load_chunk(0)
        print(f'{self._num_chunks} chunks (~{len(self._all_chunks[0])} each), '
              f'rotate every {steps_per_chunk} steps.')

    def _load_chunk(self, chunk_idx):
        # Release old
        for ep in self._episodes:
            ep.release()
        self._episodes = []
        gc.collect()

        indices = self._all_chunks[chunk_idx]
        print(f'[ReplayBuffer] Loading chunk {chunk_idx + 1}/{self._num_chunks} '
              f'({len(indices)} episodes) ...')

        valid_count = 0
        for ep_idx in indices:
            cache_file = self._cache_dir / f'ep_{int(ep_idx):06d}_s5_r256.npz'
            if not cache_file.exists():
                continue
            data = np.load(cache_file)
            images = data['images'].copy()   # COPY to own the memory
            actions = data['actions'].copy()
            ep = DroidEpisode(images, actions)
            if ep.num_strided > self._nstep + 2 * self._window_size:
                self._episodes.append(ep)
                valid_count += 1

        gc.collect()
        print(f'[ReplayBuffer] Chunk {chunk_idx + 1} loaded: '
              f'{valid_count} valid episodes in memory.')

    def _maybe_rotate(self):
        self._steps_in_current_chunk += 1
        if self._steps_in_current_chunk >= self._steps_per_chunk:
            self._steps_in_current_chunk = 0
            self._current_chunk_idx = (self._current_chunk_idx + 1) % self._num_chunks
            self._load_chunk(self._current_chunk_idx)

    def sample_batch(self, batch_size, device):
        nstep = self._nstep
        ws = self._window_size
        action_dim = self._episodes[0].actions.shape[1]
        img_shape = self._episodes[0].images.shape[1:]  # (256, 256, 3)

        obs_buf = np.zeros((batch_size, *img_shape), dtype=np.uint8)
        nxt_buf = np.zeros((batch_size, *img_shape), dtype=np.uint8)
        neg_buf = np.zeros((batch_size, *img_shape), dtype=np.uint8)
        act_buf = np.zeros((batch_size, nstep * action_dim), dtype=np.float32)

        for j in range(batch_size):
            episode = random.choice(self._episodes)

            i_min = max(0, ws - nstep)
            i_max = episode.num_strided - nstep - ws - 1
            if i_max <= i_min:
                continue

            i = random.randint(i_min, i_max)

            obs_buf[j] = episode.images[i]
            nxt_buf[j] = episode.images[i + nstep]

            for k in range(nstep):
                act_buf[j, k * action_dim:(k + 1) * action_dim] = episode.actions[i + k]

            offset = random.randint(1, ws)
            if random.random() < 0.5:
                neg_buf[j] = episode.images[i + nstep + offset]
            else:
                neg_buf[j] = episode.images[i + nstep - offset]

        self._maybe_rotate()

        obs = torch.as_tensor(obs_buf, device=device)
        action_seq = torch.as_tensor(act_buf, device=device)
        next_obs = torch.as_tensor(nxt_buf, device=device)
        neg_next_obs = torch.as_tensor(neg_buf, device=device)

        return obs, action_seq, next_obs, neg_next_obs
