"""Test the frozen pseudo-selected reward checkpoint with sequential replay."""
import json
import os
from pathlib import Path
import torch
from . import experiment as b
from . import strict_sequential as strict
from . import sequential_far_loss as far


def main():
    root = Path(os.environ.get('STRICT_SELECTED_ROOT', 'work_dir/strict45_5_sequential_reward_search'))
    selected = json.loads((root/'selected_reward.json').read_text())
    output = root/'final_unseen_test'
    bank = [25,39,46,52,54]
    far.LOSS_WEIGHT = selected['ce']
    far.SEPARATION_WEIGHT = selected['separation']
    far.HARD_NONADJACENT = False
    device = torch.device('cuda:0')
    torch.set_float32_matmul_precision('high')
    raw, audit = strict.load(Path('work_dir/frame0_body_angle_rank_strict45_10/cache'),
        Path('work_dir/frame0_body_angle_rank_strict45_10/object_tracks_full'), 'test', device,
        b.load_depth_lookup(Path('data/frame0_body_angle_rank_strict45_10_depth')))
    gate = b.fusion_gate.load_gate(Path('work_dir/fusion_lightweight_gating_strict45_5/models/seed_20260910/best.pt'),device)
    fused = b.pair_fused_logits(raw,gate)
    model = b.load_policy(Path(selected['selected']['checkpoint']), 'full_depth19', device)
    results = []
    for seed in range(20260920,20260950):
        methods = {}
        for name in ('full_depth19','random_pair','cyclic_pair','cyclic_adjacent_pair'):
            rollout = far.replay_rollout(raw,model if name=='full_depth19' else None,name,seed,bank)
            methods[name] = far._evaluate_rollout(raw,fused,rollout,bank,selected['movement'])
        row = {'seed':seed,'bank':bank,'methods':methods}
        results.append(row)
        b.dump_json(output/f'seed_{seed}.json',row)
        print('EVAL',len(results),flush=True)
    summary = far.summarize_seed_results(results)
    b.dump_json(output/'summary.json',summary)
    b.dump_json(output/'per_seed.json',results)
    b.dump_json(output/'protocol.json',{'selected':selected,'bank':bank,'audit':audit,
        'protocol':'subject-wise sequential replay; pseudo-selected checkpoint frozen; 30 evaluation seeds'})
    lines=['# Pseudo-selected reward: final unseen sequential test','',
        'Test bank: [25,39,46,52,54]. 30 stream-order/start seeds; 95% normal CI.',
        'Model and reward coefficients frozen before this evaluation. Movement is a normalized angular proxy, not meters.','',
        '|Model|Accuracy %|95% CI|Move|Move/random|Fused CE|Entropy|GT margin|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    ref=summary['random_pair']['mean_movement_cost']['mean']
    for name,v in summary.items():
        a=v['accuracy']; m=v['mean_movement_cost']['mean']
        lines.append(f"|{name}|{100*a['mean']:.2f}|[{100*a['ci95_low']:.2f}, {100*a['ci95_high']:.2f}]|{m:.4f}|{m/ref:.3f}|{v['recognition_loss']['mean']:.4f}|{v['normalized_entropy']['mean']:.4f}|{v['mean_margin']['mean']:.4f}|")
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    main()
