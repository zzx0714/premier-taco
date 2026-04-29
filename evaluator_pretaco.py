"""
Premier-TACO Group-Level Retrieval Evaluation (Dual-camera, Dot-product similarity)

Usage:
    python evaluator_pretaco.py --checkpoint <path> --nstep 3

Output:
    Per-nstep result files with Hit@1/5/10 metrics and per-query Top-10 details.
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse

sys.path.insert(0, '/data3/v-jepa/pre-taco/premier-taco')
from premier_taco_droid import ResNetEncoder, PremierTACO


# ---------------------------------------------------------------------------
# Episode-key helpers (same convention as background.json: "529_1", "529_2")
# ---------------------------------------------------------------------------

def _make_ep_key(ep_id, cam_id):
    """ep_id: int, cam_id: 1 or 2 → '000529_1'"""
    return f"{ep_id:06d}_{cam_id}"


def _parse_ep_key(key):
    """'000529_1' → (529, 1)"""
    parts = key.rsplit('_', 1)
    return int(parts[0]), int(parts[1])


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class PreTACORetrievalWrapper(nn.Module):
    def __init__(self, encoder, taco_module, device='cuda'):
        super().__init__()
        self.encoder = encoder
        self.taco = taco_module
        self.device = device

    @torch.no_grad()
    def compute_key(self, target_image):
        """
        target_image: (B, C, H, W) float tensor, range [0, 255]
        Returns: (B, 512)
        """
        z = self.taco.encode(target_image)
        return z

    @torch.no_grad()
    def compute_query(self, init_image, action_seq):
        """
        init_image: (B, C, H, W) float tensor, range [0, 255]
        action_seq: (B, nstep * action_dim) float tensor
        Returns: (B, 512)
        """
        z = self.taco.encode(init_image)
        action_en = self.taco.proj_aseq(action_seq)
        za = self.taco.project_sa(z, action_en)
        return za


# ---------------------------------------------------------------------------
# Dataset: load npz, sliding window
# ---------------------------------------------------------------------------

class RetrievalDataset:
    """Loads per-camera npz files and generates sliding-window samples."""

    # The model was trained with nstep=3; action sequences are always
    # padded/truncated to this length for the forward pass.
    NSTEP_MODEL = 3

    def __init__(self, cache_dir, background_json, nstep):
        self.cache_dir = Path(cache_dir)
        self.nstep = nstep  # evaluation horizon (target frame offset)

        with open(background_json, 'r') as f:
            self.background_groups = json.load(f)

        # Collect all unique episode-camera pairs
        self.ep_keys = []
        self.ep_data = {}  # ep_key → {'images': ndarray, 'actions': ndarray}

        for group_name, entries in self.background_groups.items():
            for entry in entries:
                ep_id_str, cam_id_str = entry.rsplit('_', 1)
                ep_id = int(ep_id_str)
                cam_id = int(cam_id_str)
                ep_key = _make_ep_key(ep_id, cam_id)
                if ep_key in self.ep_data:
                    continue
                self.ep_keys.append(ep_key)
                data = self._load_ep(ep_id, cam_id)
                if data is not None:
                    self.ep_data[ep_key] = data

        print(f"Loaded {len(self.ep_data)} episode-camera pairs")

    def _load_ep(self, ep_id, cam_id):
        cache_file = self.cache_dir / f'ep_{ep_id:06d}_cam{cam_id}_s5_r256.npz'
        if not cache_file.exists():
            print(f"  WARNING: cache not found for ep={ep_id}, cam={cam_id}: {cache_file}")
            return None
        data = np.load(cache_file)
        return {
            'images': data['images'],    # (N, 256, 256, 3) uint8
            'actions': data['actions'],  # (N, 7) float32
        }

    def get_all_samples(self):
        """
        Returns dict: ep_key → list of {
            'init_image': ndarray (3, 256, 256) uint8,
            'target_image': ndarray (3, 256, 256) uint8,
            'action_seq': ndarray (nstep * 7,) float32,
            'frame_idx': int,
            'target_frame_idx': int,
        }
        """
        all_samples = {}
        for ep_key in self.ep_keys:
            if ep_key not in self.ep_data:
                continue
            data = self.ep_data[ep_key]
            images = data['images']       # (N, H, W, 3)
            actions = data['actions']     # (N, 7)
            N = len(images)
            action_dim = actions.shape[1]
            nstep = self.nstep

            samples = []
            for i in range(N - nstep):
                # Need enough frames for both the target and the action context
                if i + max(nstep, self.NSTEP_MODEL) > N:
                    continue

                init_img = images[i].transpose(2, 0, 1)          # (3, H, W)
                target_img = images[i + nstep].transpose(2, 0, 1) # (3, H, W)

                # Build action sequence: always NSTEP_MODEL actions for the model,
                # padded by repeating the last action or truncated from the front.
                if nstep >= self.NSTEP_MODEL:
                    act_seq = actions[i: i + self.NSTEP_MODEL].reshape(-1)
                else:
                    # nstep < 3: take available actions, pad by repeating last
                    avail = actions[i: i + nstep]  # (nstep, 7)
                    pad_count = self.NSTEP_MODEL - nstep
                    act_seq = np.concatenate(
                        [avail] + [avail[-1:]] * pad_count, axis=0
                    ).reshape(-1)
                samples.append({
                    'init_image': init_img,
                    'target_image': target_img,
                    'action_seq': act_seq,
                    'frame_idx': i,
                    'target_frame_idx': i + nstep,
                })
            all_samples[ep_key] = samples
        return all_samples


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _normalize_background_groups(background_groups):
    """
    Normalize background.json entries to _make_ep_key format (6-digit padded).
    Returns (normalized_groups, ep_key_to_idx, all_ep_keys_in_order).
    """
    ep_key_to_idx = {}
    all_ep_keys = []
    normalized_groups = {}

    for group_name, entries in background_groups.items():
        normalized_groups[group_name] = []
        for entry in entries:
            ep_id_str, cam_id_str = entry.rsplit('_', 1)
            ep_id = int(ep_id_str)
            cam_id = int(cam_id_str)
            norm_key = _make_ep_key(ep_id, cam_id)
            if norm_key not in ep_key_to_idx:
                ep_key_to_idx[norm_key] = len(all_ep_keys)
                all_ep_keys.append(norm_key)
            normalized_groups[group_name].append(norm_key)

    return normalized_groups, ep_key_to_idx, all_ep_keys


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_retrieval(model, dataset, device, background_json_path,
                       log_path="pretaco_results.txt", top_k=[1, 5, 10]):
    """
    Group-level retrieval evaluation with dot-product similarity.
    """
    model.eval()

    with open(background_json_path, 'r') as f:
        background_groups = json.load(f)
    normalized_groups, ep_key_to_idx, all_ep_keys = _normalize_background_groups(background_groups)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Premier-TACO Group-Level Retrieval Evaluation (Dual-camera)\n")
        f.write("Metric: Dot-product Similarity\n")
        f.write(f"Background JSON: {background_json_path}\n")
        f.write(f"Nstep: {dataset.nstep}\n")
        f.write("=" * 100 + "\n")

    # Phase 1: Collect all key/query per episode-camera
    print(">>> Phase 1: Collecting key/query data from all episodes...")
    all_samples = dataset.get_all_samples()

    # ep_key → {'keys': {frame_idx: tensor}, 'queries': [...]}
    ep_data = {}
    for ep_key, samples in tqdm(all_samples.items(), desc="Encoding"):
        if not samples:
            continue
        keys_dict = {}
        queries_list = []

        # Batch encode for efficiency
        batch_size = 64
        for start in range(0, len(samples), batch_size):
            batch = samples[start: start + batch_size]
            target_imgs = torch.from_numpy(
                np.stack([s['target_image'] for s in batch])
            ).float().to(device)  # (B, 3, H, W) uint8 → float, encoder normalizes internally

            init_imgs = torch.from_numpy(
                np.stack([s['init_image'] for s in batch])
            ).float().to(device)

            act_seqs = torch.from_numpy(
                np.stack([s['action_seq'] for s in batch])
            ).float().to(device)

            key_tokens = model.compute_key(target_imgs).cpu()
            query_tokens = model.compute_query(init_imgs, act_seqs).cpu()

            for j, s in enumerate(batch):
                keys_dict[s['target_frame_idx']] = key_tokens[j]
                queries_list.append({
                    'pred_tokens': query_tokens[j],
                    'target_frame_idx': s['target_frame_idx'],
                })

        ep_data[ep_key] = {'keys': keys_dict, 'queries': queries_list}

    # Phase 2: Per-group retrieval
    print(">>> Phase 2: Processing groups...")
    group_hits = {k: 0 for k in top_k}
    group_total = 0
    group_log_buffers = {}

    for group_name, ep_key_list in tqdm(normalized_groups.items(), desc="Groups"):
        group_ep_keys = [ek for ek in ep_key_list if ek in ep_data]
        group_log_buffer = []
        g_hits = {k: 0 for k in top_k}
        g_total = 0

        # Build key stack for all episodes in this group
        all_keys_ep = []    # (ep_key, frame_idx)
        all_key_tensors = []

        for ep_key in group_ep_keys:
            keys_dict = ep_data[ep_key]['keys']
            frame_ids = sorted(keys_dict.keys())
            for fid in frame_ids:
                all_keys_ep.append((ep_key, fid))
                all_key_tensors.append(keys_dict[fid])

        if not all_key_tensors:
            continue

        key_stack = torch.stack(all_key_tensors)  # (N_group_keys, 512)

        # For each query in the group
        for ep_key in group_ep_keys:
            queries_list = ep_data[ep_key]['queries']
            for q_item in queries_list:
                g_total += 1
                q_tokens = q_item['pred_tokens'].to(device).unsqueeze(0)  # (1, 512)
                gt_target = q_item['target_frame_idx']
                gt_ep_key = ep_key

                # Dot product similarity
                sim_scores = torch.matmul(q_tokens, key_stack.to(device).T).squeeze(0)  # (N,)
                sorted_scores, sorted_indices = torch.sort(sim_scores, descending=True)

                # Find GT rank
                gt_global_idx = None
                for gi, (ek, fid) in enumerate(all_keys_ep):
                    if ek == gt_ep_key and fid == gt_target:
                        gt_global_idx = gi
                        break

                if gt_global_idx is None:
                    continue

                gt_score = sorted_scores[gt_global_idx].item()
                rank_of_gt = (sorted_indices == gt_global_idx).nonzero(as_tuple=True)[0].item()

                for k in top_k:
                    if rank_of_gt < k:
                        g_hits[k] += 1

                # Log top-10 details (limit to first 500 queries per group)
                if g_total <= 500:
                    score_details = []
                    for rank in range(min(10, len(sorted_indices))):
                        idx = sorted_indices[rank].item()
                        ret_ep_key, ret_fid = all_keys_ep[idx]
                        score = sorted_scores[rank].item()
                        ret_ep_id, ret_cam = _parse_ep_key(ret_ep_key)
                        label = f"{ret_ep_id:06d}_{ret_cam}-{ret_fid}({score:.4f})"
                        score_details.append(label)

                    gt_ep_id, gt_cam = _parse_ep_key(gt_ep_key)
                    log_line = (
                        f"Group_{group_name}_Q{g_total}\t"
                        f"Top10=[{' | '.join(score_details)}]\t"
                        f"GT={gt_ep_id:06d}_{gt_cam}-{gt_target}(sim={gt_score:.4f})"
                    )
                    group_log_buffer.append(log_line)

        for k in top_k:
            group_hits[k] += g_hits[k]
        group_total += g_total

        if group_log_buffer:
            group_log_buffers[group_name] = group_log_buffer

    total_queries = group_total

    # Write results
    with open(log_path, "a", encoding="utf-8") as f:
        for group_name, lines in group_log_buffers.items():
            for line in lines:
                f.write(line + "\n")

        summary = f"\nAggregated | Samples: {total_queries} | " + " | ".join(
            [f"Hit@{k}: {group_hits[k]/total_queries:.4f}" if total_queries > 0 else f"Hit@{k}: N/A"
             for k in top_k]
        )
        f.write("\n" + "=" * 80 + "\n" + summary + "\n")

    metrics = {f"Hit@{k}": group_hits[k] / total_queries for k in top_k} if total_queries > 0 else {}
    print(f"\nFinished. Results: {summary}")
    return metrics


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, device='cuda:0'):
    """
    Load Premier-TACO model from checkpoint.
    Supports:
      - Full checkpoint (dict with 'encoder' key + PremierTACO state)
      - Encoder-only checkpoint (OrderedDict of ResNetEncoder state_dict)
    """
    obs_shape = (3, 256, 256)
    feature_dim = 512
    hidden_dim = 1024
    action_dim = 7
    nstep = 3  # default, will be overridden by wrapper at eval time

    encoder = ResNetEncoder(obs_shape, feature_dim).to(device)
    taco = PremierTACO(
        repr_dim=encoder.repr_dim,
        feature_dim=feature_dim,
        action_shapes=action_dim,
        hidden_dim=hidden_dim,
        encoder=encoder,
        nstep=nstep,
        device=device,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and 'encoder' in ckpt:
        # Full checkpoint
        encoder.load_state_dict(ckpt['encoder'])
        if 'taco' in ckpt:
            taco.load_state_dict(ckpt['taco'], strict=False)
            print("Loaded full PremierTACO checkpoint (encoder + projection heads)")
        elif 'taco_state_dict' in ckpt:
            taco.load_state_dict(ckpt['taco_state_dict'], strict=False)
            print("Loaded full PremierTACO checkpoint (encoder + projection heads)")
        else:
            print("WARNING: Checkpoint contains encoder only. Projection heads (proj_s, proj_sa, proj_aseq) are randomly initialized.")
            print("  For proper retrieval, you need a full checkpoint with projection head weights.")
    elif isinstance(ckpt, dict):
        # Try loading as encoder state_dict directly
        try:
            encoder.load_state_dict(ckpt)
            print("Loaded encoder-only checkpoint. Projection heads are randomly initialized.")
            print("  For proper retrieval, you need a full checkpoint with projection head weights.")
        except Exception:
            raise ValueError(f"Unrecognized checkpoint format: {type(ckpt)}")
    else:
        raise ValueError(f"Unrecognized checkpoint format: {type(ckpt)}")

    return encoder, taco


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="Premier-TACO Group-Level Retrieval Evaluation")
    parser.add_argument("--checkpoint", type=str,
                        default="/data3/v-jepa/pre-taco/premier-taco/exp_local/droid_pretrain/encoder_best.pt")
    parser.add_argument("--data_root", type=str, default="/data3/v-jepa/pre-taco/droid_lerobot_dataset")
    parser.add_argument("--cache_dir", type=str, default="/data3/v-jepa/pre-taco/retrieval_cache")
    parser.add_argument("--background_json", type=str,
                        default="/data3/v-jepa/pre-taco/background.json")
    parser.add_argument("--nstep", type=int, nargs='+', default=[1, 3, 5],
                        help="nstep values to evaluate (separate result files per value)")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for output result files")
    parser.add_argument("--result_root", type=str, default=None)
    parser.add_argument("--run_tag", type=str, default=None)
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device(args.device)

    print(f"Loading model from {args.checkpoint}")
    encoder, taco = load_model(args.checkpoint, device=args.device)

    wrapper = PreTACORetrievalWrapper(encoder, taco, device)

    # Prepare output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.result_root:
        run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.result_root) / run_tag
    else:
        output_dir = Path('/data3/v-jepa/pre-taco/premier-taco/exp_local/droid_pretrain/retrieval_results')
        run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_dir / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    for nstep in args.nstep:
        print(f"\n{'='*60}")
        print(f"Evaluating with nstep={nstep}")
        print(f"{'='*60}")

        dataset = RetrievalDataset(
            cache_dir=args.cache_dir,
            background_json=args.background_json,
            nstep=nstep,
        )

        log_path = str(output_dir / f"retrieval_results_nstep{nstep}.txt")
        print(f"Results will be saved to {log_path}")

        metrics = evaluate_retrieval(
            wrapper, dataset, device,
            background_json_path=args.background_json,
            log_path=log_path,
        )

        print(f"\nResults (nstep={nstep}):")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
