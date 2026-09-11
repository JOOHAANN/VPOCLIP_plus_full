"""Export four-view RL evidence from the original full-clip VPO inputs."""
import argparse
import copy
import fcntl
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .build_cache import load_config, _skeleton_path, _load_angles, _angle
from .body_relative_geometry import BodyRelativeResolver, json_safe, wrap_degrees
from .vpoclip_adapter import VPOCLIPAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--split', choices=('dqn_train', 'val', 'test', 'seen_test'), required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    torch.set_num_threads(4)
    root = Path(cfg['cache']['root']) / args.split
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'build.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    if (root/'complete.json').exists():
        print('CACHE COMPLETE', root, flush=True)
        return
    experiment = Path(cfg['offline']['features'])
    raw_source = Path(cfg['data']['source_split_dir'])
    dqn = args.split == 'dqn_train'
    source_split = 'dqn' if dqn else ('val' if args.split == 'val' else 'test')
    prefix = 'train' if dqn else source_split
    features = experiment if dqn else Path(cfg['offline']['source_features'])
    names_root = experiment/'rgb_dqn' if dqn else raw_source
    names = np.load(names_root/f'{prefix}_sample_names.npy').astype(str)
    labels = np.load(names_root/f'{prefix}_labels.npy')
    pose_prefix = 'dqn' if dqn else prefix
    pnames = np.load(features/f'pose/{pose_prefix}_sample_names.npy').astype(str)
    plabels = np.load(features/f'pose/{pose_prefix}_labels.npy')
    pose_lookup = {n: i for i, n in enumerate(pnames)}
    assert len(pose_lookup) == len(pnames)
    video = np.load(features/f'x3d/{prefix}_video.npy', mmap_mode='r')
    xlabels = np.load(features/f'x3d/{prefix}_video.labels.npy')
    xrows = np.load(features/f'x3d/{prefix}_video.valid_indices.npy')
    assert np.array_equal(xlabels, labels) and np.array_equal(xrows, np.arange(len(names)))
    pose = np.load(features/f'pose/{pose_prefix}_pose_coco17.npy', mmap_mode='r')
    joints = np.load(features/f'pose/{pose_prefix}_joint_xy_coco17.npy', mmap_mode='r')
    objects = np.load(features/f'object/{prefix}_object.npy', mmap_mode='r')
    assert len(video) == len(objects) == len(names)
    expected = cfg['evaluation']['unseen_class_ids' if args.split == 'test' else 'seen_class_ids']
    subjects = set(cfg['data']['subject_groups']['test' if args.split == 'seen_test' else args.split])
    cams = cfg['cache']['candidate_cameras']
    allowed_recordings = None
    if args.split == 'test':
        selection = json.loads(Path(cfg['data']['recording_manifest']).read_text())
        allowed_recordings = {n for item in selection['selected'] for n in item['complete_recording_ids']}
    grouped = {}
    merged_raw = {}
    variable = bool(cfg.get('ranking', {}).get('variable_views', False))
    merged_labels = set(cfg.get('ranking', {}).get('merge_group_labels', []))
    for i, (name, label) in enumerate(zip(names, labels)):
        stem = Path(name).stem
        base, camera = stem.rsplit('_', 1)
        if int(label) not in expected or stem.split('_')[1] not in subjects or camera not in cams:
            continue
        if allowed_recordings is not None and base not in allowed_recordings:
            continue
        assert int(plabels[pose_lookup[name]]) == int(label)
        if int(label) in merged_labels:
            # A049 is recorded as four two-camera Groups. The Group identity,
            # rather than the repeated camera label, is the action candidate.
            merge_base, group_id = base.rsplit('_', 1)
            part = merged_raw.setdefault(merge_base, {}).setdefault(group_id, {})
            if camera in part:
                raise RuntimeError(f'Ambiguous duplicate camera in {merge_base}_{group_id}: {camera}')
            part[camera] = i
        else:
            if camera in grouped.get(base, {}):
                raise RuntimeError(f'Ambiguous duplicate camera in {base}: {camera}')
            grouped.setdefault(base, {})[camera] = i
    groups = [(base, [views[c] for c in cams if c in views]) for base, views in sorted(grouped.items())
              if (len(views) >= 2 if variable else all(c in views for c in cams))]
    merged_groups = []
    for merge_base, parts in sorted(merged_raw.items()):
        selected = []
        for group_id, views in sorted(parts.items()):
            preferred = [c for c in cams if c in views]
            chosen = preferred[0] if preferred else sorted(views)[0]
            selected.append(views[chosen])
        # The policy has four destination slots. Keep one representative
        # lower view per A049 Group, in Group order, and retain the provenance
        # in metadata through the selected source index.
        if len(selected) >= 2:
            merged_groups.append((merge_base + '_MERGED', selected[:4]))
    groups.extend(merged_groups)
    assert groups
    present = sorted({int(labels[indices[0]]) for _, indices in groups})
    missing = sorted(set(expected)-set(present))
    coverage = {'classes': present, 'missing_classes': missing, 'candidate_classes': expected,
                'episodes': len(groups), 'complete': not missing, 'sampling': 'one_full_clip_per_recording',
                'excluded_incomplete_recordings': len(grouped) + len(merged_raw) - len(groups),
                'view_count_histogram': {str(k): sum(len(v)==k for _,v in groups) for k in (2,3,4)},
                'merged_group_labels': sorted(merged_labels)}
    if args.split == 'test' and missing:
        raise RuntimeError(f'Incomplete unseen coverage: {missing}')
    (root/'coverage.json').write_text(json.dumps(coverage, indent=2))
    print('COVERAGE', args.split, json.dumps(coverage), flush=True)
    # Resolve camera/body geometry over the same full-clip sample span.
    metadata_path = root/'metadata.json'
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        rows = metadata['episodes']
        assert [r['base_sample'] for r in rows] == [g[0] for g in groups]
    else:
        resolver = BodyRelativeResolver()
        camera_angles = _load_angles(Path(cfg['data']['angle_csv']))
        previous = {}
        reuse_root = cfg.get('ranking', {}).get('reuse_cache')
        if reuse_root and (Path(reuse_root)/args.split/'metadata.json').exists():
            previous = {r['base_sample']: r for r in json.loads((Path(reuse_root)/args.split/'metadata.json').read_text())['episodes']}
        rows = []
        for e, (base, indices) in enumerate(groups):
            subject = base.split('_')[1]
            current_names = [str(Path(cfg['data']['rgb_root']) / names[i]) for i in indices]
            old = previous.get(base)
            if old and [v['video'] for v in old['views']] == current_names:
                row = copy.deepcopy(old)
                row.update(episode_id=e, offline_source_indices=indices)
                rows.append(row)
                continue
            views = []
            for idx in indices:
                cam = Path(names[idx]).stem.rsplit('_',1)[1]
                path = Path(cfg['data']['rgb_root']) / names[idx]
                cap = cv2.VideoCapture(str(path))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                assert total > 0
                source_base = path.stem.rsplit('_',1)[0]
                angle, angle_source = _angle(camera_angles, source_base, cam)
                views.append({'camera': cam, 'video': str(path), 'frame_count': total,
                              'angle_deg': angle, 'angle_source': angle_source, 'source_base': source_base,
                              'skeleton': str(_skeleton_path(Path(cfg['data']['skeleton_root']), path.stem, subject))})
            row = {'episode_id': e, 'base_sample': base, 'label': int(labels[indices[0]]), 'subject': subject,
                   'views': views, 'window': {'sample_frames': np.linspace(0, views[0]['frame_count']-1, 13).round().astype(int).tolist()},
                   'offline_source_indices': indices}
            if base.endswith('_MERGED'):
                # Cross-group recordings are not frame-synchronized: resolve each
                # camera relative to its own observed body, never register frames across G.
                parts = []
                for v in views:
                    part = {'base_sample': v['source_base'], 'views': [v],
                            'window': {'sample_frames': np.linspace(0,v['frame_count']-1,13).round().astype(int).tolist()}}
                    parts.append(BodyRelativeResolver().resolve(part))
                geo = {'views': [p['views'][0] for p in parts],
                       'body_yaw_confidence': float(np.mean([p['body_yaw_confidence'] for p in parts])),
                       'cross_group_not_synchronized': True}
            else:
                geo = resolver.resolve(row)
            row['body_relative_geometry'] = {k:v for k,v in geo.items() if k != 'views'}
            for v, gv in zip(views, geo['views']):
                v.update(gv)
            rows.append(row)
            if (e+1) % 250 == 0:
                print('GEOMETRY', args.split, e+1, '/', len(groups), flush=True)
        metadata = {'episodes': rows, 'num_views': 4, 'classes': 55, 'embedding_dim': 512,
                    'format': 'active_view_cache_v2', 'windowing': None,
                    'feature_source': str(features), 'recognizer': cfg['recognizer'], 'coverage': coverage}
        metadata_path.write_text(json.dumps(json_safe(metadata), allow_nan=False))
    n = len(rows)
    shapes = {'z': (n,4,512), 'logits': (n,4,55), 'pose': (n,4,442), 'object_map': (n,4,1800),
              'image_quality': (n,4,4), 'view_geometry': (n,4,4), 'reachable': (n,4,4),
              'move_cost': (n,4,4), 'labels': (n,), 'target_columns': (n,)}
    arrays = {}
    for key, shape in shapes.items():
        dtype = np.bool_ if key == 'reachable' else (np.int64 if key in ('labels','target_columns') else np.float32)
        path = root/(key+'.npy')
        arrays[key] = np.load(path, mmap_mode='r+') if path.exists() else np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)
        assert arrays[key].shape == shape
    done_path = root/'done.npy'
    done = np.load(done_path, mmap_mode='r+') if done_path.exists() else np.lib.format.open_memmap(done_path, mode='w+', dtype=np.bool_, shape=(n,))
    adapter = VPOCLIPAdapter.from_config(cfg['recognizer']['config'], cfg['recognizer']['checkpoint'], cfg['runtime']['device'])
    npz_path = Path(cfg['raw']['ctrgcn_root'])/'data/etri_coco17/allviews_cs45/ETRI_55_CS_rtmpose_coco17_13.npz'
    with np.load(npz_path) as npz:
        pose_scores = npz[f'x_{source_split}'][:,2]
        score_names = npz[f'{source_split}_sample_name'].astype(str)
    score_lookup = {name:i for i,name in enumerate(score_names)}
    verified = False
    valid_views = np.zeros((n,4), dtype=bool)
    for e, (_, ids) in enumerate(groups):
        valid_views[e,:len(ids)] = True
    np.save(root/'view_valid.npy', valid_views)
    for start in range(0, n, 32):
        stop = min(n,start+32)
        if done[start:stop].all():
            continue
        ids = np.array([i for _, inds in groups[start:stop] for i in (inds + [inds[0]]*(4-len(inds)))])
        pids = np.array([pose_lookup[names[i]] for i in ids])
        input_data = {k: torch.from_numpy(np.array(a[sel], copy=True)).float().to(adapter.device)
                      for k,a,sel in [('video',video,ids),('pose',pose,pids),('object',objects,ids),('joint_xy',joints,pids)]}
        with torch.inference_mode():
            encoded = adapter.encode_all_views({k:v.unsqueeze(0) for k,v in input_data.items()})
            if not verified:
                reference = adapter.model(input_data['video'], input_data['pose'], input_data['object'], input_data['joint_xy'])
                torch.testing.assert_close(encoded['logits'][0], reference, rtol=1e-4, atol=1e-4)
                parity = {'max_logit_error': float((encoded['logits'][0]-reference).abs().max()),
                          'same_inputs_same_model_forward': True, 'views_checked': len(ids)}
                (root/'forward_parity.json').write_text(json.dumps(parity))
                print('FORWARD PARITY PASS', parity, flush=True)
                verified = True
        for key in ('z','logits'):
            value = encoded[key][0].cpu().numpy().reshape(stop-start,4,-1)
            assert np.isfinite(value).all()
            arrays[key][start:stop] = value
        arrays['pose'][start:stop] = joints[pids].reshape(stop-start,4,442)
        arrays['object_map'][start:stop] = objects[ids].reshape(stop-start,4,1800)
        for e in range(start,stop):
            row = rows[e]
            angles = np.array([v['relative_bearing_deg'] for v in row['views']])
            count = len(angles)
            arrays['view_geometry'][e] = 0
            arrays['view_geometry'][e,:count] = np.stack([np.sin(np.radians(angles)),np.cos(np.radians(angles)),np.zeros(count),np.zeros(count)],axis=1)
            arrays['move_cost'][e] = 0
            arrays['move_cost'][e,:count,:count] = np.abs((angles[None,:]-angles[:,None]+180)%360-180)/180
            arrays['reachable'][e] = np.eye(4, dtype=bool)
            arrays['reachable'][e,:count,:count] = True
            arrays['labels'][e] = arrays['target_columns'][e] = row['label']
            for v,i in enumerate(groups[e][1]):
                sc = pose_scores[score_lookup[names[i]]]
                arrays['image_quality'][e,v] = [(sc>0).mean(),sc[sc>0].mean() if (sc>0).any() else 0,sc.max(),row['body_relative_geometry']['body_yaw_confidence']]
            for key in ('z','logits','pose','object_map','image_quality'):
                arrays[key][e,count:] = 0
        for a in arrays.values():
            a.flush()
        done[start:stop] = True
        done.flush()
        print('OFFLINE CACHE', args.split, stop, '/', n, flush=True)
    np.save(root/'text_prototypes.npy', adapter.model.text_features.detach().float().cpu().numpy())
    assert done.all()
    for a in arrays.values():
        assert np.isfinite(a).all()
    class_ids = np.array(expected)
    scores = arrays['logits'][:,0,:][:,class_ids]
    top1 = float((class_ids[scores.argmax(-1)] == arrays['labels']).mean())
    (root/'complete.json').write_text(json.dumps({'episodes':n, 'single_top1':top1,'coverage':coverage}))
    print('CACHE COMPLETE',args.split,'single_top1',top1,flush=True)


if __name__ == '__main__':
    main()
