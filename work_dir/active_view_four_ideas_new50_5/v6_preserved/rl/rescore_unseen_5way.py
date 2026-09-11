"""Evaluate frozen view choices with the five held-out classification candidates."""
import json
from pathlib import Path
import numpy as np

def main():
    root = Path(__file__).resolve().parents[1]
    cache = root / 'data/active_multiview_cache_unseen/test'
    out = root / 'work_dir/active_view_ddqn_body_relative_v2'
    classes = np.array([9, 10, 11, 17, 49])
    logits = np.load(cache / 'logits.npy')
    labels = np.load(cache / 'labels.npy')
    assert np.isfinite(logits).all() and np.isin(labels, classes).all()
    for checkpoint in ('best', 'last'):
        previous = json.loads((out / f'evaluation_unseen_test_{checkpoint}.json').read_text())
        predictions = {}
        for method in ('single', 'random', 'fixed', 'policy'):
            scores = []
            for row in previous['predictions']:
                i, a = row['episode_id'], row['initial_view']
                score = logits[i, a] if method == 'single' else (logits[i, a] + logits[i, row['selected'][method]]) / 2
                scores.append(score[classes])
            predictions[method] = classes[np.asarray(scores).argmax(-1)]
        metrics = {}
        for method, pred in predictions.items():
            per_class = {str(c): {'count': int((labels == c).sum()), 'top1': float((pred[labels == c] == c).mean()) if (labels == c).any() else None} for c in classes}
            metrics[method] = {'top1': float((pred == labels).mean()), 'macro_top1_present_classes': float(np.mean([v['top1'] for v in per_class.values() if v['top1'] is not None])), 'per_class': per_class}
        result = {'protocol': 'conventional_zsl_5way', 'candidate_labels_zero_based': classes.tolist(), 'episodes': len(labels), 'policy_state': 'unchanged 55-class evidence as trained; frozen label-free view choices', 'metrics': metrics, 'predictions': {k: v.tolist() for k,v in predictions.items()}, 'missing_labels': [int(c) for c in classes if not (labels == c).any()]}
        (out / f'evaluation_unseen_5way_{checkpoint}.json').write_text(json.dumps(result, indent=2))
        print(checkpoint, json.dumps(metrics), flush=True)

if __name__ == '__main__':
    main()
