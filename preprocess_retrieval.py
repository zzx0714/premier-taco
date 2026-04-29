"""
Preprocess DROID episodes for retrieval evaluation.
Decodes BOTH cameras per episode, applies stride & resize, saves per-camera npz files.

Usage:
    python preprocess_retrieval.py

Output:
    retrieval_cache/ep_{ep_idx:06d}_cam{1|2}_s{stride}_r{image_size}.npz

Each .npz contains:
    images:  (num_strided, 256, 256, 3) uint8
    actions: (num_strided, 7) float32 — pre-computed state differences
"""

import json
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed


CAMERAS = {
    1: 'observation.images.exterior_image_1_left',
    2: 'observation.images.exterior_image_2_left',
}
PAD_INDEX = 6

DATA_DIR = '/data3/v-jepa/pre-taco/droid_lerobot_dataset'
BACKGROUND_JSON = '/data3/v-jepa/pre-taco/background.json'
CACHE_DIR = Path('/data3/v-jepa/pre-taco/retrieval_cache')
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


def _cache_path(ep_idx, cam_id):
    return CACHE_DIR / f'ep_{ep_idx:06d}_cam{cam_id}_s{STRIDE}_r{IMAGE_SIZE}.npz'


def _resize_frames(frames, target_size):
    if frames.shape[1] == target_size and frames.shape[2] == target_size:
        return frames
    out = np.empty((frames.shape[0], target_size, target_size, 3), dtype=np.uint8)
    for i in range(frames.shape[0]):
        out[i] = cv2.resize(frames[i], (target_size, target_size),
                            interpolation=cv2.INTER_LINEAR)
    return out


def process_episode_camera(args):
    """Process one episode-camera pair: decode video, stride, compute actions, save npz."""
    ep_idx, cam_id = args
    cache_file = _cache_path(ep_idx, cam_id)
    if cache_file.exists():
        return True, ep_idx, cam_id

    camera_name = CAMERAS[cam_id]

    # Read states
    pq_path = _episode_parquet_path(ep_idx)
    if not pq_path.exists():
        return False, ep_idx, cam_id
    df = pd.read_parquet(pq_path)
    states = np.stack(df['observation.state'].values)
    states = np.delete(states, PAD_INDEX, axis=1)  # (T, 7)

    # Decode video
    vid_path = _episode_video_path(ep_idx, camera_name)
    if not vid_path.exists():
        return False, ep_idx, cam_id

    container = av.open(str(vid_path))
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        frames.append(frame.to_ndarray(format='rgb24'))
    container.close()
    images = np.stack(frames)  # (T, H, W, 3)

    # Resize
    images = _resize_frames(images, IMAGE_SIZE)

    # Stride sampling
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

    return True, ep_idx, cam_id


def main():
    # Extract unique episode IDs from background.json
    with open(BACKGROUND_JSON, 'r') as f:
        background_groups = json.load(f)

    episode_ids = set()
    for group_name, entries in background_groups.items():
        for entry in entries:
            ep_id = int(entry.split('_')[0])
            episode_ids.add(ep_id)
    episode_ids = sorted(episode_ids)
    print(f'Unique episodes from background.json: {len(episode_ids)}')
    print(f'  -> {episode_ids}')

    # Build task list: (ep_idx, cam_id) for all episode-camera pairs
    tasks = [(ep_id, cam_id) for ep_id in episode_ids for cam_id in [1, 2]]
    print(f'Total episode-camera pairs to process: {len(tasks)}')

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    done = 0
    skipped = 0
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_episode_camera, t): t for t in tasks}
        for future in as_completed(futures):
            success, ep_idx, cam_id = future.result()
            if success:
                done += 1
            else:
                skipped += 1
                print(f'  SKIP: ep={ep_idx}, cam={cam_id}')
            if (done + skipped) % 20 == 0:
                print(f'  Progress: {done} done, {skipped} skipped, '
                      f'{len(tasks) - done - skipped} remaining ...')

    print(f'\nDone: {done} episode-camera pairs processed, {skipped} skipped.')
    print(f'Cache directory: {CACHE_DIR}')


if __name__ == '__main__':
    main()
