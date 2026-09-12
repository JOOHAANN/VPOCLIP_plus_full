"""Read-only audit of outputs, with an isolated audit JSON."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('runner', HERE / 'run.py')
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def main():
    torch.set_num_threads(4)
    raw = r.common.load_raw(r.variants.NEW_CACHE, 'test', r.DEVICE)
    r.attach(raw)
    ids = (raw.valid.sum(-1) == 4).nonzero().flatten()[:16]
    path = torch.tensor([0, 2], device=r.DEVICE).expand(len(ids), -1)
    checks = {}
    for name, cls in [('mvselect', r.MVSelect), ('mflstm_a2c', r.MissingFrameA2C)]:
        selected = json.loads((HERE / name / 'selection.json').read_text())['chosen']
        model = cls().to(r.DEVICE).eval()
        model.load_state_dict(torch.load(selected['checkpoint'], weights_only=False, map_location=r.DEVICE)['model'])
        with torch.no_grad():
            q = model(raw, ids, path)[0]
            old_z, old_p = raw.z[ids].clone(), raw._pairdist[ids].clone()
            raw.z[ids[:, None], torch.tensor([1, 3], device=r.DEVICE)] = 1e5
            raw._pairdist[ids[:, None], torch.tensor([1, 3], device=r.DEVICE)] = -1e5
            changed = model(raw, ids, path)[0]
            raw.z[ids] = old_z; raw._pairdist[ids] = old_p
            assert torch.equal(q, changed), 'unobserved-feature leakage'
            if name == 'mvselect':
                visited = torch.zeros(len(ids), 4, device=r.DEVICE).scatter_(1, path, 1.)
                _, (_, official_q, _, _) = model.selector(raw.z[ids, :, :, None, None], visited, raw.valid[ids])
                legal = r.mask(raw, ids, path)
                assert torch.allclose(q[legal], official_q[legal], atol=1e-6), 'official Q mismatch'
        count = 0
        for f in (HERE / name).glob('seed_*/evaluation/**/*.npz'):
            d = np.load(f)
            assert d['paths'].shape[0] == 30
            assert np.array_equal(d['eval_seeds'], r.common.EVAL_SEEDS)
            assert all(len(set(p)) == len(p) for p in d['paths'].reshape(-1,d['paths'].shape[-1]))
            if 'true_unseen_5way' in str(f):
                assert len(d['labels']) == 271
                assert set(d['labels'].tolist()) == set(range(5))
                for m in r.common.METHODS: assert d[f'prediction_{m}'].max() < 5
            count += 1
        assert count == 60, (name, count)
        checks[name] = {'future_feature_perturbation': 'pass', 'prediction_files': count,
                        'no_repeated_views': True, 'unseen_5way_271': True, 'thirty_seed_ids_match': True}
    files = [HERE / 'run.py', HERE / 'README.md', HERE / 'upstream/MVSelect/src/models/mvselect.py',
             r.ROOT / 'rl/fusion_weighted_common_v1.py', r.ROOT / 'rl/weighted_fusion_policy_variants_v1.py',
             r.variants.NEW_POLICY_ROOT / 'v5/selection.json', r.variants.NEW_POLICY_ROOT / 'v5/seed_20260910/best.pt']
    r.dump(HERE / 'verification.json', {'checks': checks, 'sha256': {str(f): sha(f) for f in files}})
    print('VERIFICATION_PASS', flush=True)


if __name__ == '__main__': main()
