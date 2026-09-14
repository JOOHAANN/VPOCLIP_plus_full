"""Pseudo-unseen-only reward search for sequential DDQN."""
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys


def worker():
    from . import experiment as b
    from . import strict_sequential as strict
    from . import sequential_far_loss as far

    def validation(model, raw, fused, episodes, a, c, variant, bank, penalty):
        rows = []
        refs = []
        for seed in (731, 732, 733):
            out = far.replay_rollout(raw, model, variant, seed, bank)
            rows.append(far._evaluate_rollout(raw, fused, out, bank, penalty))
            ref = far.replay_rollout(raw, None, 'random_pair', seed, bank)
            refs.append(b.pair_metrics(raw, fused, ref, bank)['mean_movement_cost'])
        result = {k: sum(row[k] for row in rows)/len(rows) for k in rows[0]}
        ratio = result['mean_movement_cost'] / (sum(refs)/len(refs))
        result['physical_reward'] = result['mean_reward']
        result['movement_ratio_to_random'] = ratio
        if os.environ.get('STRICT_ACCURACY_FIRST') == '1':
            # Accuracy increments are >= 1/(142*3); 1e-7 only breaks ties.
            result['mean_reward'] = round(result['accuracy'], 7) - 1e-7 * result['mean_movement_cost']
        else:
            result['mean_reward'] = result['accuracy'] if ratio <= .4 else -ratio
        return result

    b.load_raw = strict.load
    b.build_transition_pool = far.pool
    b.evaluate_policy_once = validation
    b.main()


def search():
    root = Path('work_dir/strict45_5_sequential_reward_search')
    root.mkdir(parents=True, exist_ok=True)
    configs = list(itertools.product((.5, 2., 8.), (.1, .25), (0., .5)))
    manifest = {'protocol': 'strict45/5/5 sequential', 'pseudo': [1,7,14,15,18],
                'final_test': [25,39,46,52,54], 'selection_seeds': [731,732,733],
                'rule': 'maximize pseudo accuracy subject to movement/random <= .4; otherwise lowest ratio',
                'training_seeds': [20260909], 'screening_only': True,
                'fixed': {'margin':1., 'entropy':.5, 'adjacency_soft':.05},
                'grid': [{'ce':ce,'movement':move,'separation':sep} for ce,move,sep in configs]}
    (root/'search_manifest.json').write_text(json.dumps(manifest, indent=2))
    results = []
    for ce, move, sep in configs:
        output = root/f'ce{ce:g}_move{move:g}_sep{sep:g}'
        env = dict(os.environ, FAR_LOSS_WEIGHT=str(ce), FAR_SEPARATION_WEIGHT=str(sep),
                   FAR_MARGIN_WEIGHT='1', FAR_ENTROPY_WEIGHT='.5', FAR_ADJACENCY_PENALTY='.05',
                   FAR_HARD_NONADJACENT='0')
        args = [sys.executable,'-u','-m','rl.dual_robot_depth_ddqn_v1.strict_reward_search','--worker',
                '--cache-root','work_dir/frame0_body_angle_rank_strict45_10/cache',
                '--track-root','work_dir/frame0_body_angle_rank_strict45_10/object_tracks_full',
                '--depth-root','data/frame0_body_angle_rank_strict45_10_depth',
                '--gate-checkpoint','work_dir/fusion_lightweight_gating_strict45_5/models/seed_20260910/best.pt',
                '--output-root',str(output),'--train-seeds','20260909','--variants','full_depth19',
                '--epochs','30','--batch-size','32768','--updates-per-epoch','96',
                '--distance-lambda',str(move),'--validation-interval','2',
                '--true-unseen-classes','1','7','14','15','18','25','39','46','52','54',
                '--pseudo-unseen-classes','1','7','14','15','18','--validation-only']
        print('START',output,flush=True)
        with (root/f'{output.name}.log').open('a') as log:
            subprocess.run(args, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        selected=json.loads((output/'selection_full_depth19.json').read_text())['chosen']
        records=[json.loads(line) for line in (output/'models/full_depth19/seed_20260909/train.log').read_text().splitlines()]
        best=next(row for row in records if row['epoch']==selected['best_epoch'])
        results.append({'ce':ce,'movement':move,'separation':sep,'selected':selected,'validation':best['validation']})
        (root/'pseudo_comparison.json').write_text(json.dumps(results, indent=2))
        print('DONE',json.dumps(results[-1]),flush=True)
    chosen=max(results,key=lambda row:row['selected']['best_score'])
    (root/'selected_reward.json').write_text(json.dumps(chosen,indent=2))
    print('PSEUDO_SELECTED',json.dumps(chosen),flush=True)


if __name__ == '__main__':
    if '--worker' in sys.argv:
        sys.argv.remove('--worker')
        worker()
    else:
        search()
