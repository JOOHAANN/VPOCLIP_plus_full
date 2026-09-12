"""Create an isolated weighted-fusion anchor for the already trained old v5."""
from pathlib import Path
import json
import numpy as np
import torch

ROOT = Path('/home/youhan/ws/VPOCLIP_plus_full')
import sys
sys.path.insert(0, str(ROOT))
from rl import fusion_weighted_common_v1 as common
from rl import multistep_angle_object_trajectory_policy_v1 as base
from rl import multistep_angle_object_trajectory_policy_v5_no_category_id as v5
from rl import weighted_fusion_policy_variants_v1 as variants

DEVICE = torch.device('cuda:0')
CACHE = ROOT / 'data/rl_candidate_rank_v6_new50_5/cache'
TRACKS = ROOT / 'data/angle_object_trajectory_v1'
POLICY_ROOT = ROOT / 'work_dir/multistep_angle_object_trajectory_policy_v5_no_category_id_no_slots_oldsplit'
SOURCE = ROOT / 'work_dir/weighted_fusion_policy_variant_comparison_v1/v7_old_split_414'
OUT = ROOT / 'work_dir/published_selectors_v1/v5_old_weighted_anchor'
SEEDS = common.EVAL_SEEDS
TRUE_UNSEEN = [0, 2, 26, 34, 50]
SEEN = sorted(set(range(55)) - set(TRUE_UNSEEN))


@torch.inference_mode()
def policy_path(model, raw, ids, starts):
    base.build_trajectory_state = v5.build_trajectory_state_v5
    base.AngleObjectTrajectoryPolicy = v5.CategoryFreeObjectTrajectoryPolicy
    state = base.build_trajectory_state(raw, ids, starts[:, None], torch.ones_like(ids))
    action = model(state).masked_fill(~state['mask'], -torch.inf).argmax(-1)
    return torch.cat((starts[:, None], action[:, None]), 1)


def make_group(group, raw, classes, checkpoint):
    bank = common.class_bank(classes, DEVICE)
    model = v5.CategoryFreeObjectTrajectoryPolicy().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=False)['online'])
    model.eval()
    for protocol in ['fixed0','fixed1','fixed2','fixed3','random_start']:
        source = np.load(SOURCE / group / protocol / 'details' / f'{protocol}_moves_1_random.npz')
        ids = torch.as_tensor(source['episode_indices'], device=DEVICE).long()
        gate = variants.load_gate(ROOT / 'work_dir/weighted_fusion_policy_variant_comparison_v1/gates/v7_old_split/seed_20260909/best.pt')
        paths = []
        preds = {m: [] for m in common.METHODS}
        weights = {m: [] for m in common.METHODS}
        for s in range(len(SEEDS)):
            starts = torch.as_tensor(source['paths'][s, :, 0], device=DEVICE).long()
            path = policy_path(model, raw, ids, starts)
            paths.append(path.cpu().numpy())
            for method in common.METHODS:
                row = common.path_metrics(raw, ids, path, bank, method, gate)
                preds[method].append(row['prediction'].cpu().numpy())
                weights[method].append(row['weights'].cpu().numpy())
        out = OUT / group / protocol / 'details'
        out.mkdir(parents=True, exist_ok=True)
        payload = {'episode_indices': source['episode_indices'], 'paths': np.asarray(paths), 'eval_seeds': np.asarray(SEEDS)}
        for method in common.METHODS:
            payload[f'prediction_{method}'] = np.asarray(preds[method])
            payload[f'weights_{method}'] = np.asarray(weights[method])
        np.savez_compressed(out / f'{protocol}_moves_1_policy.npz', **payload)
        np.savez_compressed(out / f'{protocol}_moves_1_random.npz', **{k: source[k] for k in source.files})
        report = {'protocol': protocol, 'moves': 1, 'fusion_views': 2, 'eval_seeds': list(SEEDS), 'path_types': {}}
        for label, file in [('policy', out / f'{protocol}_moves_1_policy.npz'), ('random', out / f'{protocol}_moves_1_random.npz')]:
            d = np.load(file)
            report['path_types'][label] = {'methods': {}}
            for method in common.METHODS:
                pred = d[f'prediction_{method}']
                labels = np.searchsorted(np.asarray(sorted(classes)), raw.labels[ids].cpu().numpy())
                correct = pred == labels[None]
                report['path_types'][label]['methods'][method] = {'top1': float(correct.mean()), 'episodes': int(len(ids))}
            report['path_types'][label]['episodes'] = int(len(ids))
            report['path_types'][label]['details'] = str(file)
        (OUT / group / protocol).mkdir(parents=True, exist_ok=True)
        (OUT / group / protocol / 'moves_1.json').write_text(json.dumps(report, indent=2))


def main():
    raws = {}
    for split in ['seen_test','test']:
        raw = common.load_raw(CACHE, split, DEVICE)
        base.attach_object_tracks(raw, TRACKS, split)
        raws[split] = raw
    checkpoint = POLICY_ROOT / 'seed_20260909' / 'best.pt'
    for group, raw, classes in [('true_unseen_5way', raws['test'], TRUE_UNSEEN), ('seen', raws['seen_test'], SEEN)]:
        make_group(group, raw, classes, checkpoint)
    (OUT / 'config.json').write_text(json.dumps({'checkpoint': str(checkpoint), 'source_paths': str(SOURCE), 'eval_seeds': list(SEEDS), 'no_historical_outputs_modified': True}, indent=2))
    print('OLD_V5_WEIGHTED_ANCHOR_COMPLETE', flush=True)


if __name__ == '__main__': main()
