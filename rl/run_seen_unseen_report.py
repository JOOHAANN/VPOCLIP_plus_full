"""Audit unseen results and evaluate held-out subjects on all seen classes."""
import copy
import json
from pathlib import Path

import numpy as np
import yaml

from .build_cache import load_config, build_episode_manifest, export_cache
from .evaluate_rl import Evaluator


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / 'rl/handoff_configs/new50_5.yaml')
    out = Path(cfg['train']['output_dir'])
    summary = {}
    unseen = set(cfg['evaluation']['unseen_class_ids'])
    for name in ('best', 'last'):
        result = json.loads((out / f'evaluation_test_{name}.json').read_text())
        actual = {r['target'] for r in result['predictions']}
        if actual != unseen:
            raise RuntimeError(f'Unseen coverage mismatch: {actual} != {unseen}')
        summary[f'unseen_5way_{name}'] = {'episodes': result['episodes'], 'classes': sorted(actual), 'metrics': result['metrics']}
    print(json.dumps(summary), flush=True)
    seen_cfg = copy.deepcopy(cfg)
    seen = list(cfg['evaluation']['seen_class_ids'])
    seen_cfg['evaluation']['class_ids'] = seen
    seen_cfg['data']['allowed_labels_by_split']['test'] = seen
    seen_cfg['data']['exclude_labels_by_split']['test'] = sorted(unseen)
    seen_cfg['cache']['root'] = str(root / 'data/active_multiview_cache_new50_5_seen_test')
    path = root / 'rl/handoff_configs/new50_5_seen_test.yaml'
    path.write_text(yaml.safe_dump(seen_cfg, sort_keys=False))
    seen_cfg['_config_path'] = str(path)
    manifest = build_episode_manifest(seen_cfg, 'test')
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    actual = {r['label'] for r in rows}
    missing = sorted(set(seen) - actual)
    if not rows or actual - set(seen):
        raise RuntimeError('Seen manifest is empty or contains unexpected classes')
    coverage = {'expected_classes': seen, 'observed_classes': sorted(actual),
                'missing_classes': missing, 'complete': not missing,
                'episodes': len(rows), 'candidate_count': len(seen),
                'note': 'Same-recording four-low-view samples only; no cross-group substitution.'}
    (out / 'seen_test_coverage.json').write_text(json.dumps(coverage, indent=2))
    print('SEEN COVERAGE: ' + json.dumps(coverage), flush=True)
    export_cache(seen_cfg, manifest)
    for name in ('best', 'last'):
        seen_cfg['_checkpoint_path'] = str(out / f'{name}.pt')
        result = Evaluator(seen_cfg).run('test')
        result['protocol'] = 'seen_only_50way_partial_coverage' if missing else 'seen_only_50way'
        result['coverage'] = coverage
        result['candidate_class_ids'] = seen
        result['checkpoint'] = seen_cfg['_checkpoint_path']
        suffix = '_partial' if missing else ''
        (out / f'evaluation_seen_50way{suffix}_{name}.json').write_text(json.dumps(result, indent=2))
        summary[f'seen_50way_{name}'] = {'episodes': result['episodes'], 'classes': sorted(actual), 'coverage': coverage, 'metrics': result['metrics']}
        print(json.dumps({name: summary[f'seen_50way_{name}']}), flush=True)
    (out / 'seen_unseen_summary.json').write_text(json.dumps(summary, indent=2))
    print('SEEN_UNSEEN_TEST_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
