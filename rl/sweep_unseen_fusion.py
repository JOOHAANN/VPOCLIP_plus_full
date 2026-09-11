"""Frozen-choice, unseen-only fusion-weight ablation using cached logits."""
import csv
import json
from pathlib import Path

import numpy as np
import yaml


def main():
    project = Path(__file__).resolve().parents[1]
    config_path = project / 'rl/handoff_configs/new50_5.yaml'
    cfg = yaml.safe_load(config_path.read_text())
    cache = (config_path.parent / cfg['cache']['root']).resolve() / 'test'
    out = Path(cfg['train']['output_dir']) / 'unseen_fusion_sweep'
    out.mkdir(exist_ok=True)
    logits = np.load(cache / 'logits.npy')
    labels = np.load(cache / 'labels.npy')
    costs = np.load(cache / 'move_cost.npy')
    reachable = np.load(cache / 'reachable.npy')
    classes = np.array(cfg['evaluation']['unseen_class_ids'])
    assert set(labels.tolist()) == set(classes.tolist()) and len(classes) == 5
    assert np.isfinite(logits).all()
    idx = np.arange(len(labels))
    results, flat = {}, []
    for checkpoint in ('best', 'last'):
        base = json.loads((out.parent / f'evaluation_test_{checkpoint}.json').read_text())
        rows = base['predictions']
        assert len(rows) == len(labels)
        assert all(r['episode_id'] == i and r['target'] == int(labels[i]) and r['initial_view'] == 0 for i, r in enumerate(rows))
        results[checkpoint] = {}
        for ratio in (1, 1.5, 2, 3, 5, 10, None):
            ratio_label = '0:1' if ratio is None else f'1:{ratio}'
            a = np.float32(0 if ratio is None else 1 / (1 + ratio))
            b = np.float32(1 if ratio is None else ratio / (1 + ratio))
            paired = a * logits[:, :1, :] + b * logits
            valid = reachable[:, 0, :].copy()
            valid[:, 0] = False
            true_scores = paired[idx, :, labels]
            true_scores = np.where(valid, true_scores, -np.inf)
            # Match evaluator's max((true_class_logit, view_id)) tie rule.
            oracle = 3 - np.argmax(true_scores[:, ::-1], axis=1)
            metrics = {}
            for method in ('single', 'fixed', 'random', 'policy', 'oracle'):
                if method == 'single':
                    chosen = np.zeros(len(labels), dtype=int)
                    scores = logits[:, 0, :]
                else:
                    chosen = oracle if method == 'oracle' else np.array([r['selected'][method] for r in rows])
                    assert valid[idx, chosen].all()
                    scores = paired[idx, chosen]
                restricted = scores[:, classes]
                pred = classes[restricted.argmax(axis=1)]
                def entropy(values):
                    values = values.astype(np.float64)
                    exps = np.exp(values - values.max(axis=1, keepdims=True))
                    prob = exps / exps.sum(axis=1, keepdims=True)
                    return float((-(prob * np.log(np.maximum(prob, 1e-300))).sum(axis=1) / np.log(values.shape[1])).mean())
                m = {'top1': float((pred == labels).mean()), 'top5': 1.0,
                     'mean_entropy_55way': entropy(scores), 'mean_entropy_5way': entropy(restricted),
                     'mean_movement_cost': float(costs[idx, 0, chosen].astype(np.float64).mean()),
                     'per_class_top1': {f'A{int(c)+1:03d}': float((pred[labels == c] == c).mean()) for c in classes}}
                if ratio == 1:
                    assert abs(m['top1'] - base['metrics'][method]['top1']) < 1e-12, (checkpoint, method, m)
                    assert abs(m['mean_movement_cost'] - base['metrics'][method]['mean_movement_cost']) < 1e-7
                metrics[method] = m
                flat.append({'checkpoint': checkpoint, 'first_to_second': ratio_label, 'method': method,
                             **{k: v for k, v in m.items() if k != 'per_class_top1'}})
            results[checkpoint][ratio_label] = metrics
            print(checkpoint, ratio_label, {k: round(v['top1'] * 100, 4) for k,v in metrics.items()}, flush=True)
    report = {'protocol': 'unseen_5way', 'episodes': len(labels), 'classes': classes.tolist(),
              'formula': '(L_first + ratio * L_second) / (1 + ratio); 0:1 uses L_second only; single always uses L_first',
              'choices': 'Frozen original policy/random/fixed actions; oracle reselected by true-class fused logit.',
              'note': 'Test-set ablation, not validation-based weight selection. No model retraining.',
              'baseline_1_to_1_verified': True, 'results': results}
    (out / 'results.json').write_text(json.dumps(report, indent=2))
    with (out / 'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print('COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()
