import importlib.util
from pathlib import Path
import torch
import argparse

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('runner', ROOT / 'run.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
torch.set_num_threads(4)
raw = r.common.load_raw(r.variants.NEW_CACHE, 'dqn_train', r.DEVICE)
val = r.common.load_raw(r.variants.NEW_CACHE, 'val', r.DEVICE)
r.attach(raw); r.attach(val)
ids = (raw.valid.sum(-1) == 4).nonzero().flatten()[:8]
path = torch.zeros((len(ids), 1), dtype=torch.long, device=r.DEVICE)
for cls in [r.MVSelect, r.MissingFrameA2C]:
    m = cls().to(r.DEVICE).eval()
    with torch.no_grad():
        before = m(raw, ids, path)[0]
        z = raw.z[ids, 1:].clone(); p = raw._pairdist[ids, 1:].clone()
        raw.z[ids, 1:] = 12345.; raw._pairdist[ids, 1:] = 54321.
        after = m(raw, ids, path)[0]
        raw.z[ids, 1:] = z; raw._pairdist[ids, 1:] = p
        assert torch.equal(before, after), 'future observation leakage'
        selected = r.rollout(m, raw, ids, path[:, 0], 2)
        assert all(len(set(x)) == 3 for x in selected.tolist())
    print(cls.__name__, 'causality and no-repeat PASS', flush=True)
r.HERE = ROOT / 'smoke'
args = argparse.Namespace(batch=32, epochs=1, updates=2, pretrain=1)
for name in ['mvselect','mflstm_a2c']:
    print(r.train(name, 7, raw, val, args), flush=True)
print('SMOKE_PASS', flush=True)
