"""Queue an independent, leakage-safe ZSL experiment after the original run.

Selection uses availability only, never recognition scores. Complete low-view
quartets are recorded separately for future RL; VPO still uses all eight cameras.
"""
import argparse
import collections
import fcntl
import json
from pathlib import Path
import re
import shutil
import sys
import time

import numpy as np
import run_allviews_pipeline as pipeline

BASE_RUN = pipeline.RUN
OUT = pipeline.VPO / 'logs/allviews_lowview50_5_20260907'
LOW = {'C001', 'C003', 'C005', 'C007'}
EXCLUDED_UNSEEN_ACTIONS = {'A053', 'A041'}
REQUESTED_UNSEEN_ACTIONS = {'A001', 'A003', 'A027', 'A035', 'A051'}


def inventory(cached=False):
    groups = json.loads(pipeline.MANIFEST.read_text())['subject_groups']
    owner = {p: g for g, people in groups.items() for p in people}
    events = collections.defaultdict(set)
    valid_names = None
    if cached:
        valid_names = set()
        for split in ('train', 'val', 'test'):
            names = np.load(pipeline.CACHE / f'{split}_sample_names.npy').astype(str)
            valid = np.load(pipeline.CACHE / f'{split}_valid.npy')
            if not valid.all():
                raise RuntimeError(f'{split} contains failed decodes; review before selecting classes')
            valid_names.update(Path(n).stem for n in names[valid])
    for path in pipeline.RGB.rglob('*.mp4'):
        m = re.fullmatch(r'(A\d{3})_(P\d{3})_(G\d{3})_(C\d{3})', path.stem)
        if not m:
            continue
        a, person, repeat, camera = m.groups()
        if cached and owner[person] != 'dqn_train' and path.stem not in valid_names:
            continue
        events[a, person, repeat].add(camera)
    rows = []
    for action in range(1, 56):
        aid = f'A{action:03d}'
        items = [(p, g, cameras) for (a, p, g), cameras in events.items() if a == aid]
        complete = [(p, g) for p, g, cameras in items if LOW <= cameras]
        group_counts = {g: sum(owner[p] == g for p, _ in complete) for g in groups}
        rows.append(dict(action=aid, zero_based=action-1, total_recordings=len(items),
                         complete_quartets=len(complete),
                         complete_fraction=len(complete)/max(1, len(items)),
                         complete_quartets_by_subject_group=group_counts,
                         camera_counts=dict(collections.Counter(c for _, _, cs in items for c in cs)),
                         complete_recording_ids=[f'{aid}_{p}_{g}' for p, g in sorted(complete)]))
    # Require >=98% paired coverage and nonempty quartets in every subject group,
    # then maximize actual complete-recording counts with deterministic ID ties.
    eligible = [r for r in rows if r['action'] not in EXCLUDED_UNSEEN_ACTIONS and r['complete_fraction'] >= .98 and
                min(r['complete_quartets_by_subject_group'].values()) > 0]
    selected = sorted([r for r in eligible if r['action'] in REQUESTED_UNSEEN_ACTIONS],
                      key=lambda r: r['action'])
    if len(selected) != 5:
        raise RuntimeError('Requested unseen classes fail coverage checks; review rather than silently replace')
    report = dict(status='cache_verified' if cached else 'provisional_raw_inventory',
                  excluded_unseen_actions=sorted(EXCLUDED_UNSEEN_ACTIONS),
                  low_cameras=sorted(LOW), canonical_subject_manifest=str(pipeline.MANIFEST),
                  rule='User-fixed A001/A003/A027/A035/A051; require >=98% complete quartets and all four subject groups represented',
                  note='Incomplete recordings are excluded from unseen evaluation and RL quartets. VPO uses all available cameras of retained recordings. DQN availability is file-level until RTMPose completes.',
                  unseen_classes_zero_based=sorted(r['zero_based'] for r in selected),
                  unseen_actions=sorted(r['action'] for r in selected),
                  selected=selected, all_classes=rows)
    pipeline.save(OUT / ('class_split.json' if cached else 'provisional_class_split.json'), report)
    print(json.dumps({k: report[k] for k in ('status', 'unseen_actions', 'rule')}, indent=2), flush=True)
    for r in selected:
        print(r['action'], r['complete_quartets'], r['complete_fraction'],
              r['complete_quartets_by_subject_group'], flush=True)
    return report


def wait_for(marker):
    print('WAIT', marker, flush=True)
    while not marker.exists():
        time.sleep(20)
    print('READY', marker, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.inventory_only:
        inventory()
        return
    lock = (OUT / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    wait_for(BASE_RUN / '01_rgb_cache.done')
    report = inventory(cached=True)
    # User prioritizes new 50/5. Old supervised runs are paused.
    wait_for(BASE_RUN / '02_rtmpose.done')
    old_features = pipeline.FEATURES
    count = sum(len(np.load(pipeline.CACHE/f'{s}_labels.npy')) for s in ('train','val','test'))
    extra = 2 * count * (13*192*6*6*2 + 2*64*13*17*4 + 50*6*6*4) + 15 * 2**30
    free = shutil.disk_usage(pipeline.WS).free
    pipeline.save(OUT / 'disk_preflight.json', dict(required_bytes=extra, free_bytes=free))
    if free < extra:
        raise RuntimeError(f'Need {extra/2**30:.1f} GiB additional space; only {free/2**30:.1f} GiB free. No existing data deleted.')
    pipeline.UNSEEN = report['unseen_classes_zero_based']
    pipeline.CLASS_SPLIT_PATH = OUT / 'class_split.json'
    pipeline.RUN = OUT
    pipeline.RUN_NAME = 'allviews_lowview50_5_20260907'
    pipeline.OUTPUT_PREFIX = 'allviews_lowview50_5_'
    pipeline.XOUT = pipeline.X3D / 'outputs/etri_allviews_cs45_lowview50_5'
    pipeline.COUT = pipeline.CTR / 'work_dir/etri_coco17/allviews_cs45_lowview50_5'
    pipeline.FEATURES = pipeline.VPO / 'data/allviews_cs45_lowview50_5_features'
    pipeline.ZDATA = pipeline.VPO / 'data/allviews_cs45_lowview50_5_zsl'
    pipeline.CDATA = pipeline.VPO / 'data/allviews_cs45_lowview50_5_unused_closed'
    pipeline.OBJECTS = pipeline.FEATURES / 'object'
    pipeline.ZSL_ONLY = True
    pipeline.X3D_MAX_ITER = 7000  # User-requested total optimizer iterations, not epochs.
    pipeline.REUSE_OBJECTS = False
    for name in ('01_rgb_cache', '02_rtmpose'):
        source = BASE_RUN / f'{name}.done'
        if not source.exists():
            raise RuntimeError(f'Missing shared stage: {source}')
        shutil.copy2(source, OUT / source.name)
    pipeline.save(OUT / 'shared_cache_provenance.json', dict(source_run=str(BASE_RUN),
                  rgb_cache=str(pipeline.CACHE), rtmpose_npz=str(pipeline.NPZ),
                  object_cache=str(pipeline.OBJECTS), class_split=str(OUT / 'class_split.json')))
    sys.argv = [sys.argv[0]]
    pipeline.main()


if __name__ == '__main__':
    main()
