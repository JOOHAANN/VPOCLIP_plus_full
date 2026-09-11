"""Build all held-out-class four-view windows, then evaluate both checkpoints."""
import copy
import json
from pathlib import Path
import torch
from .build_cache import load_config, build_episode_manifest, export_cache
from .evaluate_rl import Evaluator

def main():
    torch.set_num_threads(4)
    cfg = load_config(Path(__file__).with_name('config_rl.yaml'))
    cfg = copy.deepcopy(cfg)
    unseen = set(cfg['data']['exclude_labels'])
    cfg['data']['exclude_labels'] = sorted(set(range(55)) - unseen)
    cfg['data']['max_windows_per_clip'] = None
    cfg['cache']['root'] = '../data/active_multiview_cache_unseen'
    cfg['raw']['decode_workers'] = 4
    cfg['raw']['reuse_pose_frames'] = True
    manifest = build_episode_manifest(cfg, 'test')
    export_cache(cfg, manifest)
    out = (Path(cfg['_config_path']).parent / cfg['train']['output_dir']).resolve()
    for name in ('best', 'last'):
        cfg['_checkpoint_path'] = str(out / f'{name}.pt')
        result = Evaluator(cfg).run('test')
        result['protocol_note'] = 'Held-out five classes; 55-way predictions; all sliding windows; fixed initial C001'
        result['checkpoint'] = cfg['_checkpoint_path']
        (out / f'evaluation_unseen_test_{name}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(name, result['metrics'], flush=True)
    print('status=complete', flush=True)

if __name__ == '__main__':
    main()
