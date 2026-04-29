"""Preprocess DROID dataset: decode mp4 + parquet → strided, resized .npz files.

Usage:
    python preprocess_droid.py

Output:
    droid_lerobot_dataset/premier_taco_cache/ep_XXXXXX_s5_r256.npz

Each .npz contains:
    images: (num_strided, 256, 256, 3) uint8 — one random camera, strided + resized
    actions: (num_strided, 7) float32 — pre-computed state differences
"""

import json
import random
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed


CAMERAS = [
    'observation.images.exterior_image_1_left',
    'observation.images.exterior_image_2_left',
]
PAD_INDEX = 6

DATA_DIR = '/data3/v-jepa/pre-taco/droid_lerobot_dataset'
SPLIT_FILE = '/data3/v-jepa/pre-taco/1w2splits.json'
CACHE_DIR = Path(DATA_DIR) / 'premier_taco_cache'
IMAGE_SIZE = 256
STRIDE = 5
NUM_WORKERS = 4


def _episode_video_path(ep_idx, camera):
    chunk = ep_idx // 1000
    return (Path(DATA_DIR) / 'videos' / f'chunk-{chunk:03d}' / camera /
            f'episode_{ep_idx:06d}.mp4')


def _episode_parquet_path(ep_idx):
    chunk = ep_idx // 1000
    return (Path(DATA_DIR) / 'data' / f'chunk-{chunk:03d}' /
            f'episode_{ep_idx:06d}.parquet')


def _cache_path(ep_idx):
    return CACHE_DIR / f'ep_{ep_idx:06d}_s{STRIDE}_r{IMAGE_SIZE}.npz'


def _resize_frames(frames, target_size):
    if frames.shape[1] == target_size and frames.shape[2] == target_size:
        return frames
    out = np.empty((frames.shape[0], target_size, target_size, 3), dtype=np.uint8)
    for i in range(frames.shape[0]):
        out[i] = cv2.resize(frames[i], (target_size, target_size),
                            interpolation=cv2.INTER_LINEAR)
    return out


def process_episode(ep_idx):
    """Process one episode: decode video, read states, stride, compute actions, save .npz."""
    cache_file = _cache_path(ep_idx)
    if cache_file.exists():
        return True, ep_idx

    # Read states
    pq_path = _episode_parquet_path(ep_idx)
    if not pq_path.exists():
        return False, ep_idx
    df = pd.read_parquet(pq_path)
    states = np.stack(df['observation.state'].values)
    states = np.delete(states, PAD_INDEX, axis=1)  # (T, 7)

    # Random camera
    camera = random.choice(CAMERAS)
    vid_path = _episode_video_path(ep_idx, camera)
    if not vid_path.exists():
        return False, ep_idx

    # Decode video
    container = av.open(str(vid_path))
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        frames.append(frame.to_ndarray(format='rgb24'))
    container.close()
    images = np.stack(frames)  # (T, H, W, 3)

    # Resize
    images = _resize_frames(images, IMAGE_SIZE)

    # Stride sampling — COPY to break reference to full array
    images = images[::STRIDE].copy()  # (num_strided, 256, 256, 3)

    # Pre-compute actions: action[k] = strided_state[k+1] - strided_state[k]
    strided_states = states[::STRIDE]
    num_strided = len(strided_states)
    actions = np.zeros((num_strided, states.shape[1]), dtype=np.float32)
    valid = num_strided - 1
    if valid > 0:
        actions[:valid] = strided_states[1:] - strided_states[:-1]

    # Save
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, images=images, actions=actions)

    return True, ep_idx


def main():
    with open(SPLIT_FILE, 'r') as f:
        splits = json.load(f)

    # Process all splits (train + val + test)
    all_indices = []
    for key in ['train', 'val', 'test']:
        if key in splits:
            all_indices.extend(splits[key])
    print(f'Total episodes to process: {len(all_indices)}')

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    done = 0
    skipped = 0
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_episode, idx): idx for idx in all_indices}
        for future in as_completed(futures):
            success, idx = future.result()
            if success:
                done += 1
            else:
                skipped += 1
            if done % 100 == 0:
                print(f'  Progress: {done} done, {skipped} skipped, '
                      f'{len(all_indices) - done - skipped} remaining ...')

    print(f'Done: {done} episodes processed, {skipped} skipped.')
    print(f'Cache directory: {CACHE_DIR}')


if __name__ == '__main__':
    main()
