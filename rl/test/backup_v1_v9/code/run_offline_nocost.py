"""Independent full-clip feature RL experiment; no movement reward penalty."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT.parent
EXP = ROOT / 'data/rl_offline_new50_5_nocost'
CFG = ROOT / 'rl/handoff_configs/new50_5_offline_nocost.yaml'


def stage(name, command, cwd=ROOT):
    marker = EXP / (name + '.done')
    if marker.exists():
        print('SKIP', name, flush=True)
        return
    print('START', name, command, flush=True)
    subprocess.run([sys.executable, *map(str, command)], cwd=cwd, check=True)
    marker.touch()
    print('DONE', name, flush=True)


def prepare():
    EXP.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / 'rl/handoff_configs/new50_5.yaml').read_text())
    cfg['cache']['root'] = str(EXP / 'cache')
    cfg['reward']['movement_weight'] = 0.0
    cfg['train']['output_dir'] = str(ROOT / 'work_dir/active_view_ddqn_new50_5_offline_nocost')
    cfg['train']['resume'] = None
    # High-throughput batches without the previous 268 million draws per epoch.
    cfg['fast']['batch_size'] = 65536
    cfg['fast']['updates_per_epoch'] = 64
    cfg['offline'] = {'features': str(EXP), 'source_features': str(ROOT / 'data/allviews_cs45_lowview50_5_features'),
                      'sampling': 'full_clip_uniform_13_frames', 'no_sliding_windows': True}
    cfg['_config_path'] = str(CFG)
    CFG.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg


def build_rgb(cfg):
    rgb = EXP / 'rgb_dqn'
    rgb.mkdir(exist_ok=True)
    source = Path(cfg['data']['source_split_dir'])
    names = np.load(source / 'dqn_sample_names.npy').astype(str)
    labels = np.load(source / 'dqn_labels.npy')
    seen = cfg['evaluation']['seen_class_ids']
    cameras = cfg['cache']['candidate_cameras']
    keep = np.isin(labels, seen) & np.array([Path(n).stem.rsplit('_', 1)[1] in cameras for n in names])
    selected = names[keep]
    assert all(Path(n).stem.split('_')[1] in cfg['data']['subject_groups']['dqn_train'] for n in selected)
    spec = importlib.util.spec_from_file_location('original_rgb_builder', WS / 'X3D_full/tools/build_etri_rgb_tensor.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = [{'relative_path': n, 'label': int(y), 'path': Path(cfg['data']['rgb_root']) / n}
               for n, y in zip(selected, labels[keep])]
    print('OFFLINE RGB: selected DQN low-view clips', len(records), flush=True)
    module.build_split('train', records, rgb, 13, 160, 16, False)
    if not np.load(rgb / 'train_valid.npy').all():
        raise RuntimeError('DQN full-clip RGB contains failed decodes; inspect failures before aligning features')


def main():
    cfg = prepare()
    build_rgb(cfg)
    raw = cfg['raw']
    for folder in ('x3d', 'object', 'pose'):
        (EXP / folder).mkdir(exist_ok=True)
    stage('x3d_dqn', [WS/'X3D_full/tools/extract_x3d_features.py', '--config', raw['x3d_config'],
          '--checkpoint', raw['x3d_checkpoint'], '--data-dir', EXP/'rgb_dqn', '--split', 'train',
          '--output', EXP/'x3d/train_video.npy', '--batch-size', '512', '--num-workers', '8',
          '--tensor-resize-size', '182', '--spatial-size', '6', '--save-sidecars'], WS/'X3D_full')
    stage('object_dqn', [WS/'yolov5_full/extract_frame7_objects.py', '--data-dir', EXP/'rgb_dqn',
          '--weights', raw['yolo_weights'], '--split', 'train', '--output', EXP/'object/train_object.npy',
          '--batch-size', '256', '--device', '0'], WS/'yolov5_full')
    stage('object_maps', [ROOT/'tools/build_object_rs_maps.py', '--paths', EXP/'object/train_object.npy',
          '--backup-dir', EXP/'object/backups', '--overwrite'])
    stage('pose_dqn', [ROOT/'tools/extract_current_ctrgcn_features.py', '--ctrgcn-root', raw['ctrgcn_root'],
          '--config', raw['ctrgcn_config'], '--weights', raw['ctrgcn_weights'],
          '--npz', WS/'CTR-GCN_17_full/data/etri_coco17/allviews_cs45/ETRI_55_CS_rtmpose_coco17_13.npz',
          '--split', 'dqn', '--output-dir', EXP/'pose', '--batch-size', '1024'])
    for split in ('dqn_train', 'val', 'test', 'seen_test'):
        stage('cache_' + split, ['-m', 'rl.build_offline_cache', '--config', CFG, '--split', split])
    stage('train', ['-m', 'rl.train_rl_fast', '--config', CFG])
    from .evaluate_rl import Evaluator
    out = Path(cfg['train']['output_dir'])
    for split in ('test', 'seen_test'):
        evaluate_cfg = copy.deepcopy(cfg)
        evaluate_cfg['evaluation']['class_ids'] = cfg['evaluation']['unseen_class_ids' if split == 'test' else 'seen_class_ids']
        coverage = json.loads((Path(cfg['cache']['root'])/split/'coverage.json').read_text())
        for ck in ('best', 'last'):
            evaluate_cfg['_checkpoint_path'] = str(out / (ck + '.pt'))
            result = Evaluator(evaluate_cfg).run(split)
            result['coverage'] = coverage
            result['protocol'] = 'unseen_5way' if split == 'test' else 'seen_50way_partial_coverage'
            (out/f'evaluation_{split}_{ck}.json').write_text(json.dumps(result, indent=2))
            print('EVALUATION', split, ck, json.dumps(result['metrics']), flush=True)
    print('OFFLINE_NO_COST_COMPLETE', flush=True)


if __name__ == '__main__':
    if '--pose-only' in sys.argv:
        cfg = yaml.safe_load(CFG.read_text())
        raw = cfg['raw']
        (EXP/'pose').mkdir(exist_ok=True)
        stage('pose_dqn', [ROOT/'tools/extract_current_ctrgcn_features.py', '--ctrgcn-root', raw['ctrgcn_root'],
              '--config', raw['ctrgcn_config'], '--weights', raw['ctrgcn_weights'],
              '--npz', WS/'CTR-GCN_17_full/data/etri_coco17/allviews_cs45/ETRI_55_CS_rtmpose_coco17_13.npz',
              '--split', 'dqn', '--output-dir', EXP/'pose', '--batch-size', '1024'])
    else:
        main()
