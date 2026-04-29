"""
JEPA-WMS L1 Retrieval Evaluation (Group-level, dual-camera)
"""
import sys
import os
import json
from pathlib import Path
from datetime import datetime
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse

sys.path.insert(0, '/data3/v-jepa/vjepa2')
from dataset.dataset import ContrastiveDataset


class JEPAWMSRetrievalWrapper(nn.Module):
    def __init__(self, model, preprocessor, img_size, device='cuda'):
        super().__init__()
        self.model = model
        self.preprocessor = preprocessor
        self.img_size = img_size
        self.device = device
        self.grid_size = model.grid_size

    def _needs_proprio(self):
        return hasattr(self.model, "encode_proprio") and self.model.encode_proprio is not None

    def compute_key(self, target_image, target_state=None):
        B = target_image.shape[0]
        target_image = target_image.unsqueeze(1)
        obs = {'visual': target_image, 'proprio': None}
        if self._needs_proprio() and target_state is not None:
            obs['proprio'] = target_state.unsqueeze(1)
        z = self.model.encode_obs(obs)
        z_visual = z['visual']
        B, T, V, H_grid, W_grid, D = z_visual.shape
        z_flat = z_visual.reshape(B, H_grid * W_grid, D)
        return z_flat

    def compute_query(self, init_image, actions, states):
        B, Horizon, _ = actions.shape
        init_image = init_image.unsqueeze(1)
        init_proprio = None
        if self._needs_proprio() and states is not None:
            init_proprio = states[:, :1]
        obs = {'visual': init_image, 'proprio': init_proprio}
        z_ctxt = self.model.encode_obs(obs)
        if z_ctxt["proprio"] is None:
            z_ctxt_for_unroll = z_ctxt['visual']
        else:
            z_ctxt_for_unroll = z_ctxt
        act_suffix = actions.permute(1, 0, 2)
        z_pred = self.model.unroll(z_ctxt_for_unroll, act_suffix)
        if isinstance(z_pred, dict) or (hasattr(z_pred, "keys") and "visual" in z_pred.keys()):
            z_final = z_pred['visual'][-1]
        else:
            z_final = z_pred[-1]
        B, V, H_grid, W_grid, D = z_final.shape
        z_final_flat = z_final.reshape(B, H_grid * W_grid, D)
        return z_final_flat


def prepare_batch_for_jepa_wms(batch, device, target_size, normalize_mean=[0.485, 0.456, 0.406], normalize_std=[0.229, 0.224, 0.225]):
    images = batch['images'].to(device)
    actions = batch['actions'].to(device)
    states = batch['states'].to(device)
    B, T, C, H, W = images.shape
    images_flat = images.reshape(B * T, C, H, W)
    images_float = images_flat.float() / 255.0
    images_resized = F.interpolate(images_float, size=(target_size, target_size), mode='bilinear', align_corners=False)
    mean = torch.tensor(normalize_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(normalize_std, device=device).view(1, 3, 1, 1)
    images_normalized = (images_resized - mean) / std
    images_normalized = images_normalized.reshape(B, T, C, target_size, target_size)
    return {
        'images': images_normalized,
        'actions': actions,
        'states': states,
        'episode_idx': batch['episode_idx'],
        'frame_idx': batch['frame_idx'],
        'camera_index': batch.get('camera_index', torch.zeros(B, dtype=torch.long)),
    }


def _pad_ep(ep_id):
    """Convert episode id to 6-digit padded string."""
    return f"{ep_id:06d}"


def _make_ep_key(ep_id, camera_idx):
    """Unique key for a (episode, camera) pair."""
    cam_suffix = "_2" if camera_idx == 1 else "_1"
    return f"{_pad_ep(ep_id)}{cam_suffix}"


def _parse_ep_key(key):
    """Parse ep_key back to (padded_ep_str, camera_idx)."""
    if key.endswith("_2"):
        return key[:-2], 1
    else:
        return key[:-2], 0


def _normalize_background_groups(background_groups):
    """
    Normalize background.json ep_keys to match _make_ep_key format (6-digit padded).
    Also builds the canonical ep_key_to_idx mapping.
    Returns (normalized_groups, ep_key_to_idx, all_ep_keys_in_order).
    """
    ep_key_to_idx = {}
    all_ep_keys_in_order = []
    normalized_groups = {}
    for group_name, ep_key_list in background_groups.items():
        normalized_groups[group_name] = []
        for ep_key in ep_key_list:
            ep_str, cam_idx = _parse_ep_key(ep_key)
            # Strip leading zeros to get integer, then re-pad to 6 digits
            ep_id = int(ep_str.lstrip('0') or '0')
            normalized_key = _make_ep_key(ep_id, cam_idx)
            if normalized_key not in ep_key_to_idx:
                ep_key_to_idx[normalized_key] = len(all_ep_keys_in_order)
                all_ep_keys_in_order.append(normalized_key)
            normalized_groups[group_name].append(normalized_key)
    return normalized_groups, ep_key_to_idx, all_ep_keys_in_order


@torch.no_grad()
def evaluate_l1_retrieval_group_level(model, val_loader, device, background_json_path,
                                       log_path="jepa_wms_l1_results.txt",
                                       top_k=[1, 5, 10], log_all_scores=False):
    """
    Group-level retrieval evaluation.
    Each group contains 10 "episodes" (5 original episodes × 2 cameras).
    For each query, search across ALL 10 episodes in the group (self + 9 others).
    """
    model.eval()

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("JEPA-WMS L1 Retrieval Evaluation (Group-level, dual-camera)\n")
        f.write("Metric: Negative L1 Distance\n")
        f.write(f"Background JSON: {background_json_path}\n")
        f.write("="*100 + "\n")

    # Load background groups and normalize to _make_ep_key format
    with open(background_json_path, 'r') as f:
        background_groups = json.load(f)
    normalized_groups, ep_key_to_idx, all_ep_keys_in_order = _normalize_background_groups(background_groups)

    num_total_eps = len(all_ep_keys_in_order)
    print(f">>> Total unique episode-camera pairs across all groups: {num_total_eps}")

    # Per-episode-camera data: ep_key -> {'keys': {frame_id: tensor}, 'queries': [...]}
    ep_data = {ek: {'keys': {}, 'queries': []} for ek in all_ep_keys_in_order}

    # Buffer for episodes whose data has been fully collected
    # ep_key -> whether we have flushed its data
    flushed = {ek: False for ek in all_ep_keys_in_order}

    # Mapping from ep_key to its group name
    ep_key_to_group = {}
    for group_name, ep_key_list in background_groups.items():
        for ek in ep_key_list:
            ep_key_to_group[ek] = group_name

    # Track current batch position for flush decisions
    current_ep_key = None
    total_queries = 0
    hits = {k: 0 for k in top_k}

    print(">>> Phase 1: Collecting key/query data from all episodes...")
    pbar = tqdm(val_loader, desc="Collecting")

    for batch in pbar:
        batch = prepare_batch_for_jepa_wms(batch, device, model.img_size)
        B, T_seq = batch['images'].shape[:2]
        horizon = T_seq - 1
        init_frames = batch['images'][:, 0]
        target_frames = batch['images'][:, -1]
        target_states = None if batch['states'] is None else batch['states'][:, -1]

        key_tokens = model.compute_key(target_frames, target_state=target_states).detach().cpu()
        query_tokens = model.compute_query(init_frames, batch['actions'], batch['states']).detach().cpu()

        for i in range(B):
            raw_ep_id = batch['episode_idx'][i].item()
            cam_idx = batch['camera_index'][i].item()
            padded_ep = _pad_ep(raw_ep_id)
            ep_key = _make_ep_key(raw_ep_id, cam_idx)

            # Skip if not in our background groups
            if ep_key not in ep_key_to_idx:
                continue

            frame_idx = batch['frame_idx'][i].item()
            target_f = frame_idx + horizon

            ep_data[ep_key]['keys'][target_f] = key_tokens[i].clone()
            ep_data[ep_key]['queries'].append({
                'pred_tokens': query_tokens[i].clone(),
                'target_frame': target_f,
            })

    print(f">>> Phase 2: Processing groups...")
    group_log_buffers = {}
    group_hits = {k: 0 for k in top_k}
    group_total = 0

    for group_name, ep_key_list in tqdm(normalized_groups.items(), desc="Groups"):
        group_ep_keys = [ek for ek in ep_key_list if ek in ep_data and ek in ep_key_to_idx]
        group_log_buffer = []
        group_g_hits = {k: 0 for k in top_k}
        g_total = 0

        # Build key stack for all episodes in this group
        all_keys_ep = []   # (ep_key, frame_id)
        all_key_tensors = []

        for ep_key in group_ep_keys:
            keys_dict = ep_data[ep_key]['keys']
            frame_ids = sorted(keys_dict.keys())
            for fid in frame_ids:
                all_keys_ep.append((ep_key, fid))
                all_key_tensors.append(keys_dict[fid])

        if not all_key_tensors:
            continue

        key_stack = torch.stack(all_key_tensors)  # (N_group_keys, H*W, D)

        # For each query across all episodes in the group
        for ep_key in group_ep_keys:
            queries_list = ep_data[ep_key]['queries']
            for q_item in queries_list:
                g_total += 1
                q_tokens = q_item['pred_tokens'].to(device).unsqueeze(0)
                gt_target = q_item['target_frame']
                gt_ep_key = ep_key

                l1_diff = torch.abs(q_tokens - key_stack.to(device))
                l1_scores = torch.mean(l1_diff, dim=[1, 2])
                sim_scores = -l1_scores
                sorted_scores, sorted_indices = torch.sort(sim_scores, descending=True)

                # Find GT rank among ALL group keys
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
                        group_g_hits[k] += 1

                if g_total <= 500:
                    score_details = []
                    loop_limit = len(sorted_indices) if log_all_scores else 10
                    for rank, idx in enumerate(sorted_indices[:loop_limit]):
                        global_key_idx = idx.item()
                        ret_ep_key, ret_fid = all_keys_ep[global_key_idx]
                        score = sorted_scores[rank].item()
                        # Format: padded_ep-cam_frame(score) or padded_ep_frame(score)
                        ret_padded, ret_cam = _parse_ep_key(ret_ep_key)
                        if ret_cam == 1:
                            label = f"{ret_padded}_2-{ret_fid}({score:.4f})"
                        else:
                            label = f"{ret_padded}_1-{ret_fid}({score:.4f})"
                        score_details.append(label)

                    log_line = (
                        f"Group_{group_name}_Q{g_total}\t"
                        f"Top10=[{' | '.join(score_details)}]\t"
                        f"GT={gt_ep_key}_1-{gt_target}(L1={-gt_score:.4f})"
                    )
                    group_log_buffer.append(log_line)

        for k in top_k:
            group_hits[k] += group_g_hits[k]
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
        f.write("\n" + "="*80 + "\n" + summary + "\n")

    metrics = {f"Hit@{k}": group_hits[k] / total_queries for k in top_k} if total_queries > 0 else {}
    print(f"\nFinished. Results: {summary}")
    return metrics


def get_args():
    parser = argparse.ArgumentParser(description="JEPA-WMS L1 Retrieval Evaluation (Group-level)")
    parser.add_argument("--model_name", type=str, default="jepa_wm_droid", choices=["jepa_wm_droid", "dino_wm_droid", "vjepa2_ac_droid"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data_root", type=str, default="/data3/v-jepa/vjepa2/dataset/droid_lerobot_dataset")
    parser.add_argument("--splits_path", type=str, default="/data3/v-jepa/vjepa2/dataset/droid_1000/stride6/splits.json")
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--result_root", type=str, default=None, help="Root directory for per-baseline outputs")
    parser.add_argument("--run_tag", type=str, default=None, help="Optional run tag under result_root")
    parser.add_argument("--log_all_scores", action="store_true")
    parser.add_argument("--background_json", type=str,
                        default="/data3/v-jepa/jepa-wms/evals/l1_retrieval/background.json",
                        help="Path to background groups JSON")
    return parser.parse_args()


def _resolve_existing_path(base_dir: Path, candidates):
    for rel in candidates:
        p = base_dir / rel
        if p.exists() and p.is_file():
            return p
    return None


def _resolve_model_checkpoint(base_dir: Path, model_name: str):
    candidate_map = {
        "jepa_wm_droid": [
            "checkpoints/jepa_wm/jepa_wm_droid.pth.tar",
            "checkpoints/jepa_wm_droid.pth.tar",
        ],
        "dino_wm_droid": [
            "checkpoints/dino_wm/dino_wm_droid.pth.tar",
            "checkpoints/dino_wm_droid.pth.tar",
        ],
        "vjepa2_ac_droid": [
            "checkpoints/vjepa2/vjepa2_ac_droid.pth.tar",
            "checkpoints/vjepa2_ac_droid.pth.tar",
            "checkpoints/vjepa2/vjepa2-ac-vitg.pt",
        ],
    }
    ckpt_path = _resolve_existing_path(base_dir, candidate_map[model_name])
    if ckpt_path is None:
        listed = "\n".join([str(base_dir / x) for x in candidate_map[model_name]])
        raise FileNotFoundError(f"Could not locate checkpoint for {model_name}. Tried:\n{listed}")
    return ckpt_path


def _patch_pretrain_paths(base_dir: Path, model_name: str, args_eval: dict):
    os.environ.setdefault("JEPAWM_OSSCKPT", str(base_dir / "checkpoints"))
    if model_name != "vjepa2_ac_droid":
        return args_eval
    visual_encoder = args_eval.get("model_kwargs", {}).get("pretrain_kwargs", {}).get("visual_encoder", {})
    if not isinstance(visual_encoder, dict):
        return args_eval
    candidate_paths = [
        base_dir / "checkpoints" / "vjepa2" / "vjepa2-ac-vitg.pt",
        base_dir.parent / "vjepa2" / "checkpoint" / "vjepa2-ac-vitg.pt",
        base_dir / "checkpoints" / "vjepa2_opensource" / "vjepa2_vit_giant.pth",
    ]
    found = None
    for c in candidate_paths:
        if c.exists() and c.is_file():
            found = c
            break
    if found is None:
        raise FileNotFoundError("Could not locate V-JEPA2 giant encoder checkpoint. Tried:\n" + "\n".join(str(p) for p in candidate_paths))
    visual_encoder["pretrain_enc_path"] = str(found)
    return args_eval


def _prepare_output_paths(args, base_dir: Path):
    if args.output_file:
        output_file = Path(args.output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        return output_file, None
    result_root = Path(args.result_root) if args.result_root else (base_dir / "background_retrieval" / "results")
    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = result_root / run_tag / args.model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / "l1_results.txt", run_dir


def load_model_local(model_name, device="cuda:0"):
    """Load model from local config and checkpoint files."""
    import yaml
    sys.path.insert(0, '/data3/v-jepa/jepa-wms')
    from app.plan_common.datasets import get_data_stats
    from app.plan_common.datasets.preprocessor import Preprocessor
    from app.plan_common.datasets.transforms import make_inverse_transforms, make_transforms
    from evals.simu_env_planning.eval import init_module
    from src.utils.yaml_utils import expand_env_vars

    model_configs = {
        "jepa_wm_droid": "configs/evals/simu_env_planning/droid/jepa-wm/droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0_ep64_decode.yaml",
        "dino_wm_droid": "configs/evals/simu_env_planning/droid/dino-wm/droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r224_alpha0_ep64_decode.yaml",
        "vjepa2_ac_droid": "configs/evals/simu_env_planning/droid/vj2ac_oss/droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0_ep64_decode.yaml",
    }

    base_dir = Path('/data3/v-jepa/jepa-wms')
    config_rel = model_configs[model_name]
    config_path = base_dir / config_rel
    checkpoint_path = _resolve_model_checkpoint(base_dir, model_name)

    with open(config_path, "r") as f:
        args_eval = yaml.safe_load(f)
    args_eval = _patch_pretrain_paths(base_dir, model_name, args_eval)
    args_eval = expand_env_vars(args_eval)

    model_kwargs = args_eval["model_kwargs"]
    cfgs_data = model_kwargs.get("data", {})
    cfgs_data_aug = model_kwargs.get("data_aug", {})
    wrapper_kwargs = model_kwargs.get("wrapper_kwargs", {})
    pretrain_kwargs = model_kwargs.get("pretrain_kwargs", {})

    env_name = model_name.split("_")[-1]
    data_stats = get_data_stats(env_name)
    action_dim = data_stats["action_dim"]
    proprio_dim = data_stats["proprio_dim"]

    img_size = cfgs_data.get("img_size", 224)
    transform = make_transforms(
        img_size=img_size,
        normalize=cfgs_data_aug.get("normalize", [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]),
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
    )
    inverse_transform = make_inverse_transforms(img_size=img_size, **cfgs_data_aug)

    preprocessor = Preprocessor(
        action_mean=torch.tensor(data_stats["action_mean"]),
        action_std=torch.tensor(data_stats["action_std"]),
        state_mean=torch.tensor(data_stats["state_mean"]),
        state_std=torch.tensor(data_stats["state_std"]),
        proprio_mean=torch.tensor(data_stats["proprio_mean"]),
        proprio_std=torch.tensor(data_stats["proprio_std"]),
        transform=transform,
        inverse_transform=inverse_transform,
    )

    module_name = model_kwargs.get("module_name")
    model = init_module(
        folder=str(checkpoint_path.parent),
        checkpoint=checkpoint_path.name,
        module_name=module_name,
        model_kwargs=pretrain_kwargs,
        wrapper_kwargs=wrapper_kwargs,
        cfgs_data=cfgs_data,
        device=device,
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        preprocessor=preprocessor,
    )
    return model, preprocessor, img_size, str(config_path), str(checkpoint_path), args_eval


def main():
    args = get_args()
    device = torch.device(args.device)
    print(f"Loading model: {args.model_name}")

    model, preprocessor, img_size, used_config_path, used_checkpoint_path, used_cfg = load_model_local(
        args.model_name, device=args.device)

    print(f"Model loaded. Grid size: {model.grid_size}")
    print(f"Loading dataset from {args.data_root}")

    # Build episode list from background.json
    with open(args.background_json, 'r') as f:
        background_groups = json.load(f)
    episode_ids = []
    # Normalize background.json keys to match _make_ep_key format (6-digit padded ep)
    normalized_groups = {}
    for group_name, ep_key_list in background_groups.items():
        normalized_groups[group_name] = []
        for ep_key in ep_key_list:
            ep_str, cam_idx = _parse_ep_key(ep_key)
            # ep_str may or may not have leading zeros; normalize to 6-digit padded
            ep_id = int(ep_str.lstrip('0') or '0')
            episode_ids.append(ep_id)
            normalized_groups[group_name].append(_make_ep_key(ep_id, cam_idx))
    # Re-build ep_key_to_idx with normalized (padded) keys
    ep_key_to_idx = {}
    all_ep_keys_in_order = []
    for group_name, ep_key_list in normalized_groups.items():
        for ek in ep_key_list:
            if ek not in ep_key_to_idx:
                ep_key_to_idx[ek] = len(all_ep_keys_in_order)
                all_ep_keys_in_order.append(ek)
    episode_ids = sorted(set(episode_ids))
    print(f"Target episodes from background.json: {len(episode_ids)} unique episodes")
    print(f"  -> {episode_ids}")

    dataset = ContrastiveDataset(
        repo_id=args.data_root,
        root=args.data_root,
        stride=args.stride,
        horizon=args.horizon,
        mode='eval',
        dual_cam_eval=True,
        episode_ids=episode_ids,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    print(f"Dataset loaded. Total samples (dual-cam): {len(dataset)}")

    wrapper = JEPAWMSRetrievalWrapper(model, preprocessor, img_size, device)
    base_dir = Path('/data3/v-jepa/jepa-wms')
    output_file_path, run_dir = _prepare_output_paths(args, base_dir)
    output_file = str(output_file_path)

    if run_dir is not None:
        import yaml
        shutil.copy2(used_config_path, run_dir / Path(used_config_path).name)
        shutil.copy2(args.background_json, run_dir / "background.json")
        with open(run_dir / "resolved_eval_config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(used_cfg, f, sort_keys=False)
        with open(run_dir / "run_meta.txt", "w", encoding="utf-8") as f:
            f.write(f"model_name={args.model_name}\n")
            f.write(f"checkpoint={used_checkpoint_path}\n")
            f.write(f"config={used_config_path}\n")
            f.write(f"device={args.device}\n")
            f.write(f"batch_size={args.batch_size}\n")
            f.write(f"num_workers={args.num_workers}\n")
            f.write(f"stride={args.stride}\n")
            f.write(f"horizon={args.horizon}\n")
            f.write(f"data_root={args.data_root}\n")
            f.write(f"splits_path={args.splits_path}\n")
            f.write(f"background_json={args.background_json}\n")
            f.write(f"dual_cam_eval=True\n")

    print(f"Starting evaluation. Results will be saved to {output_file}")
    metrics = evaluate_l1_retrieval_group_level(
        wrapper, dataloader, device,
        background_json_path=args.background_json,
        log_path=output_file,
        log_all_scores=args.log_all_scores,
    )

    print(f"\nFinal Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
