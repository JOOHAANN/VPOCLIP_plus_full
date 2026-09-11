"""Train and evaluate v9-derived improvements in isolated directories."""
import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .causal_rank_v6 import causal_state, evaluate, task_banks
from .causal_rank_v7 import utility
from .run_four_ideas import SEEDS, validation, verify_v6
from .train_rl_fast import GPUCache
from .v9_improvement_models import decision_loss, make_model


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'work_dir/active_view_v9_improvements_new50_5'
VERSIONS = ['v12', 'v13', 'v14', 'v15']
EPOCHS = 40


def dump(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def atomic_save(obj, path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)


def telemetry(stop):
    path = ROOT / 'logs/rl_v9_improvements_new50_5/gpu_telemetry.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', buffering=1) as f:
        while not stop.is_set():
            r = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw',
                                '--format=csv,noheader,nounits'], capture_output=True, text=True)
            f.write(json.dumps({'time': time.time(), 'gpu': r.stdout.strip()}) + '\n')
            stop.wait(2)


def dropout_state(state, generator):
    """v15 candidate masking; preserve at least one legal action per row."""
    valid = state['mask'].clone()
    count = valid.sum(-1)
    eligible = count > 1
    choose = (torch.rand(valid.shape[0], device=valid.device, generator=generator) < .25) & eligible
    ids = torch.randint(0, 4, (valid.shape[0],), device=valid.device, generator=generator)
    picked = valid & torch.nn.functional.one_hot(ids, 4).bool()
    drop = choose[:, None] & picked
    valid &= ~drop
    out = dict(state, mask=valid, candidate=state['candidate'].clone())
    out['candidate'][..., -1] = valid.float()
    return out


def train_version(version, train, val, present, vp, cfg, pool, hold, seen, proxy_eps, proxy_bank, device):
    version_out = OUT / version
    version_out.mkdir(parents=True, exist_ok=True)
    if (version_out / 'summary.json').exists():
        return json.loads((version_out / 'summary.json').read_text())
    flat = (present & torch.isin(train.labels, torch.tensor(pool, device=device))[:, None]).flatten().nonzero().flatten()
    episodes, starts = (flat // 4).repeat(4), (flat % 4).repeat(4)
    labels = train.labels[episodes]
    freq = torch.bincount(labels, minlength=55).float()
    uniform = 1 / freq[labels]; uniform /= uniform.sum()
    records = []
    for seed in SEEDS:
        run = version_out / f'seed_{seed}'; run.mkdir(exist_ok=True)
        if (run / 'complete.json').exists():
            records.append(json.loads((run / 'complete.json').read_text())); continue
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        generator = torch.Generator(device=device).manual_seed(seed)
        net = make_model(version).to(device)
        compiled = torch.compile(net, mode='reduce-overhead')
        opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=.03)
        best, begin = -float('inf'), 1
        for epoch in range(begin, EPOCHS + 1):
            started = time.monotonic()
            banks = task_banks(train, episodes, pool, generator, .25)
            state = causal_state(train, episodes, starts, banks)
            u, selective = utility(train, episodes, starts, banks, state['mask'])
            priority = uniform * selective
            priority = priority / priority.sum() if priority.sum() > 0 else uniform
            probability = .5 * uniform + .5 * priority
            net.train(); losses = []
            for _ in range(48):
                ids = torch.multinomial(probability, 8192, True, generator=generator)
                s = {k: v[ids] for k, v in state.items()}
                if version == 'v15':
                    s = dropout_state(s, generator)
                loss = decision_loss(version, compiled(s), u[ids], s['mask'])
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2.); opt.step()
                losses.append(loss.detach())
            score, regular, proxy = validation(net, val, vp, seen, hold, proxy_eps, proxy_bank)
            improved = score > best
            if improved:
                best = score
            saved = dict(online=net.state_dict(), optimizer=opt.state_dict(), generator=generator.get_state(),
                         cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), best=best,
                         epoch=epoch, version=version, seed=seed)
            atomic_save(saved, run / 'last.pt')
            if improved:
                atomic_save(saved, run / 'best.pt')
            row = dict(version=version, seed=seed, epoch=epoch, loss=torch.stack(losses).mean().item(),
                       seconds=time.monotonic() - started, selective=int(selective.sum()), selection=score,
                       best=best, validation=regular, proxy=proxy)
            with (run / 'train.log').open('a') as f: f.write(json.dumps(row) + '\n')
            print(json.dumps({k: row[k] for k in ('version', 'seed', 'epoch', 'loss', 'selection', 'best')}), flush=True)
        result = dict(version=version, seed=seed, score=best, path=str(run))
        dump(run / 'complete.json', result); records.append(result)
        del compiled, net, opt
    chosen = max(records, key=lambda x: x['score'])
    selection = dict(version=version, chosen=chosen, runs=records, criterion='validation_only')
    dump(version_out / 'selection.json', selection)
    net = make_model(version).to(device)
    saved = torch.load(Path(chosen['path']) / 'best.pt', map_location=device, weights_only=False)
    net.load_state_dict(saved['online']); net.eval()
    reports = {}
    for split, classes in [('seen_test', cfg['evaluation']['seen_class_ids']),
                           ('test', cfg['evaluation']['unseen_class_ids'])]:
        test = GPUCache(Path(cfg['cache']['root']) / split, device)
        test_present = torch.from_numpy(np.load(Path(cfg['cache']['root']) / split / 'view_valid.npy')).to(device)
        for protocol in ('fixed0', 'random'):
            report = evaluate(net, test, test_present, classes, protocol)
            report['selection'] = selection
            report['coverage'] = json.loads((Path(cfg['cache']['root']) / split / 'coverage.json').read_text())
            dump(version_out / f'evaluation_{split}_{protocol}_best.json', report)
            reports[f'{split}_{protocol}'] = report['metrics']
            print('EVALUATION', version, split, protocol, json.dumps(report['metrics']), flush=True)
        del test
    del net
    result = dict(selection=selection, metrics=reports)
    dump(version_out / 'summary.json', result)
    return result


def main():
    verify_v6()
    torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / 'logs/rl_candidate_rank_v6_new50_5/config.yaml').read_text())
    cfg['v9_improvements'] = dict(versions=VERSIONS, epochs=EPOCHS, seeds=SEEDS, batch_size=8192,
                                  updates_per_epoch=48, movement_weight=0., base='v9_pairwise_ranking',
                                  selection='validation_only', test_classes=cfg['evaluation']['unseen_class_ids'])
    (OUT / 'config_resolved.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    for name in ('v9_improvement_models.py', 'run_v9_improvements.py'):
        (OUT / name).write_bytes((ROOT / 'rl' / name).read_bytes())
    device = torch.device(cfg['runtime']['device']); cache_root = Path(cfg['cache']['root'])
    train, val = [GPUCache(cache_root / x, device) for x in ('dqn_train', 'val')]
    present, vp = [torch.from_numpy(np.load(cache_root / x / 'view_valid.npy')).to(device) for x in ('dqn_train', 'val')]
    pool, hold = cfg['v6']['policy_train_classes'], cfg['v6']['policy_holdout_classes']
    seen = cfg['evaluation']['seen_class_ids']
    proxy_eps = torch.isin(val.labels, torch.tensor(hold, device=device)).nonzero().flatten().repeat(4)
    proxy_bank = task_banks(val, proxy_eps, hold, torch.Generator(device=device).manual_seed(90117), 0)
    summaries = {}
    for version in VERSIONS:
        summaries[version] = train_version(version, train, val, present, vp, cfg, pool, hold, seen, proxy_eps, proxy_bank, device)
        torch.cuda.empty_cache()
    dump(OUT / 'comparison.json', dict(base_v9='work_dir/active_view_four_ideas_new50_5/v9', versions=summaries,
                                       v6_preservation_verified=True))
    verify_v6()
    print('V9_IMPROVEMENTS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
