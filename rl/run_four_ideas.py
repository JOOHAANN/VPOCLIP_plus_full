"""Sequential v8-v11 experiments, validation-only selection, immutable v6 audit."""
import argparse
import hashlib
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
from .four_ideas_models import Ensemble, decision_loss, make_model
from .train_rl_fast import GPUCache


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'work_dir/active_view_four_ideas_new50_5'
SEEDS = [20260909, 20260910, 20260911]
VERSIONS = ['v8', 'v9', 'v10', 'v11']


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2)); temp.replace(path)


def preserve():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / 'v6_preserved/manifest.json'
    if not manifest.exists():
        files = list((ROOT / 'rl').rglob('*.py')) + list((ROOT / 'rl').rglob('*.yaml'))
        files += [ROOT / 'logs/rl_candidate_rank_v6_new50_5/config.yaml',
                  ROOT / 'work_dir/active_view_candidate_rank_v6_new50_5/summary.json']
        hashes = {}
        for source in files:
            relative = source.relative_to(ROOT)
            target = manifest.parent / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            content = source.read_bytes(); target.write_bytes(content)
            hashes[str(relative)] = hashlib.sha256(content).hexdigest()
        dump(manifest, hashes)
    verify_v6()


def verify_v6():
    manifest = json.loads((OUT / 'v6_preserved/manifest.json').read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, f'Preserved source changed: {name}'


def telemetry(stop):
    with (OUT / 'gpu_telemetry.jsonl').open('a', buffering=1) as f:
        while not stop.is_set():
            r = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw', '--format=csv,noheader,nounits'], capture_output=True, text=True)
            f.write(json.dumps(dict(time=time.time(), gpu=r.stdout.strip())) + '\n')
            stop.wait(2)


def validation(net, val, present, seen, hold, proxy_eps, proxy_bank):
    regular = {m: evaluate(net, val, present, seen, m) for m in ('fixed0', 'random')}
    proxy = {m: evaluate(net, val, present, hold, m, proxy_eps, proxy_bank) for m in ('fixed0', 'random')}
    score = float(.75 * np.mean([x['macro_choice_gain'] for x in proxy.values()]) + .25 * np.mean([x['macro_choice_gain'] for x in regular.values()]))
    return score, {k: v['metrics'] for k, v in regular.items()}, {k: v['metrics'] for k, v in proxy.items()}


def run_version(version, epochs):
    out = OUT / version
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        return json.loads((out / 'summary.json').read_text())
    cfg = yaml.safe_load((ROOT / 'logs/rl_candidate_rank_v6_new50_5/config.yaml').read_text())
    cache_root = Path(cfg['cache']['root']); device = torch.device(cfg['runtime']['device'])
    cfg['experiment'] = dict(version=version, epochs=epochs, seeds=SEEDS, batch_size=8192, updates_per_epoch=48,
                             initial_protocols=['fixed0', 'random'], dynamic_tasks=True, priority_mixture=.5,
                             movement_weight=0., fusion=[1., 1.], selection='validation_only',
                             ensemble=version == 'v10', two_view_decision_weight=0.)
    cfg['train']['output_dir'] = str(out); cfg['train']['epochs'] = epochs
    cfg['agent']['algorithm'] = 'offline_one_step_candidate_selection'
    cfg['reward']['movement_weight'] = 0.
    (out / 'config_resolved.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    for name in ('run_four_ideas.py', 'four_ideas_models.py'):
        (out / name).write_bytes((ROOT / 'rl' / name).read_bytes())
    pool, hold = cfg['v6']['policy_train_classes'], cfg['v6']['policy_holdout_classes']
    seen, unseen = cfg['evaluation']['seen_class_ids'], cfg['evaluation']['unseen_class_ids']
    assert not (set(pool) & set(hold) or (set(pool) | set(hold)) & set(unseen))
    train, val = [GPUCache(cache_root / split, device) for split in ('dqn_train', 'val')]
    present, vp = [torch.from_numpy(np.load(cache_root / split / 'view_valid.npy')).to(device) for split in ('dqn_train', 'val')]
    flat = (present & torch.isin(train.labels, torch.tensor(pool, device=device))[:, None]).flatten().nonzero().flatten()
    episodes, starts = (flat // 4).repeat(4), (flat % 4).repeat(4)
    labels = train.labels[episodes]; freq = torch.bincount(labels, minlength=55).float()
    uniform = 1 / freq[labels]; uniform /= uniform.sum()
    proxy_eps = torch.isin(val.labels, torch.tensor(hold, device=device)).nonzero().flatten().repeat(4)
    proxy_bank = task_banks(val, proxy_eps, hold, torch.Generator(device=device).manual_seed(90117), 0)
    records = []
    for seed in SEEDS:
        run = out / f'seed_{seed}'; run.mkdir(exist_ok=True)
        if (run / 'complete.json').exists():
            records.append(json.loads((run / 'complete.json').read_text())); continue
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        gen = torch.Generator(device=device).manual_seed(seed)
        net = make_model(version).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=.03)
        compiled = torch.compile(net, mode='reduce-overhead')
        best, begin = -float('inf'), 1
        if (run / 'last.pt').exists():
            saved = torch.load(run / 'last.pt', map_location=device, weights_only=False)
            net.load_state_dict(saved['online']); opt.load_state_dict(saved['optimizer'])
            gen.set_state(saved['generator']); torch.set_rng_state(saved['cpu_rng'].cpu()); torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
            best, begin = saved['best'], saved['epoch'] + 1
        for epoch in range(begin, epochs + 1):
            started = time.monotonic()
            banks = task_banks(train, episodes, pool, gen, .25)
            state = causal_state(train, episodes, starts, banks)
            u, selective = utility(train, episodes, starts, banks, state['mask'])
            priority = uniform * selective
            priority = priority / priority.sum() if priority.sum() > 0 else uniform
            probability = .5 * uniform + .5 * priority
            net.train(); losses = []
            for _ in range(48):
                ix = torch.multinomial(probability, 8192, True, generator=gen)
                s = {k: v[ix] for k, v in state.items()}
                loss = decision_loss(version, compiled(s), u[ix], s['mask'])
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 2.); opt.step()
                losses.append(loss.detach())
            score, regular, proxy = validation(net, val, vp, seen, hold, proxy_eps, proxy_bank)
            improved = score > best; best = max(best, score)
            saved = dict(online=net.state_dict(), optimizer=opt.state_dict(), generator=gen.get_state(), cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), best=best, epoch=epoch, version=version, seed=seed)
            torch.save(saved, run / 'last.tmp.pt'); (run / 'last.tmp.pt').replace(run / 'last.pt')
            if improved:
                torch.save(saved, run / 'best.tmp.pt'); (run / 'best.tmp.pt').replace(run / 'best.pt')
            record = dict(version=version, seed=seed, epoch=epoch, loss=torch.stack(losses).mean().item(), selective=int(selective.sum()), seconds=time.monotonic()-started, selection=score, best=best, validation=regular, proxy=proxy)
            with (run / 'train.log').open('a') as f: f.write(json.dumps(record) + '\n')
            print(json.dumps({k: v for k, v in record.items() if k not in ('validation', 'proxy')}), flush=True)
        record = dict(seed=seed, score=best, path=str(run))
        dump(run / 'complete.json', record); records.append(record)
        del net, compiled, opt
    chosen = max(records, key=lambda r: r['score'])
    members = []; selected_epochs = []
    for record in records if version == 'v10' else [chosen]:
        net = make_model(version).to(device)
        saved = torch.load(Path(record['path']) / 'best.pt', map_location=device, weights_only=False)
        net.load_state_dict(saved['online']); net.eval(); members.append(net); selected_epochs.append(saved['epoch'])
    net = Ensemble(members).to(device) if version == 'v10' else members[0]
    score, _, _ = validation(net, val, vp, seen, hold, proxy_eps, proxy_bank)
    selection = dict(version=version, validation_score=score, runs=records, chosen=chosen if version != 'v10' else 'equal_mean_of_three_members', epochs=selected_epochs, criterion='validation_only')
    dump(out / 'selection.json', selection)
    torch.save(dict(online=net.state_dict(), selection=selection, config=cfg), out / 'selected_policy.pt')
    reports = {}
    del train, val, state
    for split in ('val', 'seen_test', 'test'):
        cache = GPUCache(cache_root / split, device)
        real = torch.from_numpy(np.load(cache_root / split / 'view_valid.npy')).to(device)
        classes = unseen if split == 'test' else seen
        for protocol in ('fixed0', 'random'):
            report = evaluate(net, cache, real, classes, protocol)
            report['selection'] = selection
            report['coverage'] = json.loads((cache_root / split / 'coverage.json').read_text())
            assert not report['coverage']['missing_classes']
            dump(out / f'evaluation_{split}_{protocol}_best.json', report)
            reports[f'{split}_{protocol}'] = report['metrics']
            print('EVALUATION', version, split, protocol, json.dumps(report['metrics']), flush=True)
        del cache
    result = dict(selection=selection, metrics=reports)
    dump(out / 'summary.json', result)
    verify_v6()
    print('VERSION_COMPLETE', version, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', choices=VERSIONS + ['all'], default='all')
    parser.add_argument('--epochs', type=int, default=40)
    args = parser.parse_args()
    preserve(); torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    stop = threading.Event(); worker = threading.Thread(target=telemetry, args=(stop,), daemon=True); worker.start()
    try:
        for version in VERSIONS if args.version == 'all' else [args.version]:
            run_version(version, args.epochs)
            torch.cuda.empty_cache()
        available = {v: json.loads((OUT / v / 'summary.json').read_text()) for v in VERSIONS if (OUT / v / 'summary.json').exists()}
        winner = max(available, key=lambda v: available[v]['selection']['validation_score'])
        dump(OUT / 'comparison.json', dict(versions=available, validation_selected_version=winner, v6_preservation_verified=True))
        verify_v6(); print('FOUR_IDEAS_COMPLETE', flush=True)
    finally:
        stop.set(); worker.join(timeout=5)


if __name__ == '__main__':
    main()
