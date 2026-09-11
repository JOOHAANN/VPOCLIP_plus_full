"""Dynamic seen-only tasks and relative candidate utility regression."""
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import yaml

from .causal_rank_v6 import causal_state, evaluate, task_banks
from .train_rl_fast import GPUCache


class Ranker(nn.Module):
    def __init__(self, semantic=True):
        super().__init__()
        self.semantic = semantic
        self.z = nn.Sequential(nn.Linear(512, 16), nn.LayerNorm(16), nn.GELU())
        self.pose = nn.Sequential(nn.Linear(442, 16), nn.LayerNorm(16), nn.GELU())
        self.obj = nn.Sequential(nn.Linear(1800, 16), nn.LayerNorm(16), nn.GELU())
        self.context = nn.Sequential(nn.Linear(71 if semantic else 23, 64), nn.GELU(), nn.Dropout(.35))
        self.geometry = nn.Sequential(nn.Linear(7, 32), nn.GELU())
        self.head = nn.Sequential(nn.Linear(96, 64), nn.GELU(), nn.Dropout(.2), nn.Linear(64, 1))

    def forward(self, s):
        parts = [s['quality'], s['evidence']]
        if self.semantic:
            parts += [F.dropout(self.z(s['z']), .4, self.training),
                      F.dropout(self.pose(s['pose']), .4, self.training),
                      F.dropout(self.obj(s['object']), .4, self.training)]
        h = self.context(torch.cat(parts, -1))[:, None].expand(-1, 4, -1)
        q = self.head(torch.cat((h, self.geometry(s['candidate'])), -1)).squeeze(-1)
        return q


def utility(cache, episodes, starts, banks, valid):
    fused = (cache.logits[episodes, starts, None] + cache.logits[episodes]) * .5
    labels = cache.labels[episodes]
    true = fused.gather(-1, labels[:, None, None].expand(-1, 4, 1)).squeeze(-1)
    rivals = banks.clone().scatter_(1, labels[:, None], False)
    margin = true - fused.masked_fill(~rivals[:, None], -torch.inf).amax(-1)
    correct = fused.masked_fill(~banks[:, None], -torch.inf).argmax(-1).eq(labels[:, None])
    # Correctness dominates; bounded margins provide a weaker dense signal.
    u = correct.float() + .2 * torch.tanh(margin / 10)
    selective = (correct & valid).any(-1) & ((~correct) & valid).any(-1)
    return u, selective


def relative_loss(q, u, valid):
    count = valid.sum(-1).clamp_min(1)
    center = lambda x: x - (x * valid).sum(-1, keepdim=True) / count[:, None]
    regression = (F.smooth_l1_loss(center(q), center(u), reduction='none') * valid).sum(-1) / count
    delta = u[:, :, None] - u[:, None, :]
    pairs = valid[:, :, None] & valid[:, None, :] & (delta > .05)
    diff = q[:, :, None] - q[:, None, :]
    weights = delta.clamp_min(0) * pairs
    ranking = (F.softplus(-diff / .2) * weights).sum((1, 2)) / weights.sum((1, 2)).clamp_min(1e-6)
    # Two-view recordings have no second-view choice; zero decision gradient.
    active = (count > 1).float()
    return ((regression + .2 * ranking) * active).sum() / active.sum().clamp_min(1)


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / 'logs/rl_candidate_rank_v6_new50_5/config.yaml').read_text())
    out = root / 'work_dir/active_view_candidate_rank_v7_new50_5'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        print('V7_ALREADY_COMPLETE', flush=True)
        return
    cfg['v7'] = dict(epochs=40, batch_size=8192, updates=48, seeds=[20260909, 20260910],
                     variants=['semantic', 'geometry'], utility='correct + 0.2*tanh(true_margin/10)',
                     pair_tolerance=.05, dynamic_tasks=True, movement_weight=0)
    (out / 'config_resolved.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / 'causal_rank_v7.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    device = torch.device(cfg['runtime']['device'])
    cache_root = Path(cfg['cache']['root'])
    train, val = [GPUCache(cache_root / k, device) for k in ('dqn_train', 'val')]
    present, vp = [torch.from_numpy(np.load(cache_root / k / 'view_valid.npy')).to(device) for k in ('dqn_train', 'val')]
    pool, hold = cfg['v6']['policy_train_classes'], cfg['v6']['policy_holdout_classes']
    seen, unseen = cfg['evaluation']['seen_class_ids'], cfg['evaluation']['unseen_class_ids']
    assert not (set(pool) & set(hold) or (set(pool) | set(hold)) & set(unseen))
    eligible = present & torch.isin(train.labels, torch.tensor(pool, device=device))[:, None]
    flat = eligible.flatten().nonzero().flatten()
    episodes, starts = (flat // 4).repeat(4), (flat % 4).repeat(4)
    labels = train.labels[episodes]
    freq = torch.bincount(labels, minlength=55).float()
    uniform = 1 / freq[labels]; uniform /= uniform.sum()
    proxy_eps = torch.isin(val.labels, torch.tensor(hold, device=device)).nonzero().flatten()
    # Multiple fixed five-way proxy tasks reduce dependence on one task draw.
    proxy_eps = proxy_eps.repeat(4)
    proxy_bank = task_banks(val, proxy_eps, hold, torch.Generator(device=device).manual_seed(90117), 0)
    runs = []
    for variant in cfg['v7']['variants']:
        for seed in cfg['v7']['seeds']:
            run = out / f'{variant}_{seed}'; run.mkdir(exist_ok=True)
            if (run / 'complete.json').exists():
                runs.append(json.loads((run / 'complete.json').read_text())); continue
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            gen = torch.Generator(device=device).manual_seed(seed)
            net = Ranker(variant == 'semantic').to(device)
            model = torch.compile(net, mode='reduce-overhead')
            opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=.03)
            best, start_epoch = -float('inf'), 1
            if (run / 'last.pt').exists():
                saved = torch.load(run / 'last.pt', map_location=device, weights_only=False)
                net.load_state_dict(saved['online']); opt.load_state_dict(saved['optimizer'])
                gen.set_state(saved['generator']); torch.set_rng_state(saved['cpu_rng'].cpu()); torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
                best, start_epoch = saved['best'], saved['epoch'] + 1
            for epoch in range(start_epoch, cfg['v7']['epochs'] + 1):
                t = time.monotonic()
                banks = task_banks(train, episodes, pool, gen, .25)
                state = causal_state(train, episodes, starts, banks)
                u, selective = utility(train, episodes, starts, banks, state['mask'])
                priority = uniform * selective; priority = priority / priority.sum() if priority.sum() else uniform
                probability = .5 * uniform + .5 * priority
                net.train(); losses = []
                for _ in range(cfg['v7']['updates']):
                    ix = torch.multinomial(probability, cfg['v7']['batch_size'], True, generator=gen)
                    s = {k: v[ix] for k, v in state.items()}
                    loss = relative_loss(model(s), u[ix], s['mask'])
                    opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 2); opt.step()
                    losses.append(loss.detach())
                regular = {m: evaluate(net, val, vp, seen, m) for m in ('fixed0', 'random')}
                proxy = {m: evaluate(net, val, vp, hold, m, proxy_eps, proxy_bank) for m in ('fixed0', 'random')}
                score = float(.75 * np.mean([v['macro_choice_gain'] for v in proxy.values()]) + .25 * np.mean([v['macro_choice_gain'] for v in regular.values()]))
                improved = score > best; best = max(best, score)
                saved = dict(online=net.state_dict(), optimizer=opt.state_dict(), generator=gen.get_state(), cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), epoch=epoch, best=best, variant=variant, seed=seed)
                torch.save(saved, run / 'last.pt')
                if improved: torch.save(saved, run / 'best.pt')
                record = dict(variant=variant, seed=seed, epoch=epoch, loss=torch.stack(losses).mean().item(), seconds=time.monotonic()-t, selective=int(selective.sum()), selection=score, best=best,
                              validation={m: v['metrics'] for m, v in regular.items()}, proxy={m: v['metrics'] for m, v in proxy.items()})
                with (run / 'train.log').open('a') as f: f.write(json.dumps(record)+'\n')
                print(json.dumps(record), flush=True)
            result = dict(variant=variant, seed=seed, score=best, path=str(run))
            (run / 'complete.json').write_text(json.dumps(result)); runs.append(result)
            del model, net
    chosen = max(runs, key=lambda r: r['score'])
    (out / 'selection.json').write_text(json.dumps(dict(chosen=chosen, runs=runs, criterion='validation_only'), indent=2))
    net = Ranker(chosen['variant']=='semantic').to(device)
    saved = torch.load(Path(chosen['path'])/'best.pt', map_location=device, weights_only=False)
    net.load_state_dict(saved['online'])
    reports = {}
    for split in ('val', 'seen_test', 'test'):
        cache = GPUCache(cache_root / split, device)
        valid = torch.from_numpy(np.load(cache_root / split / 'view_valid.npy')).to(device)
        for mode in ('fixed0', 'random'):
            report = evaluate(net, cache, valid, unseen if split=='test' else seen, mode)
            report.update(selected_run=chosen, checkpoint_epoch=saved['epoch'])
            (out / f'evaluation_{split}_{mode}_best.json').write_text(json.dumps(report, indent=2))
            reports[f'{split}_{mode}'] = report['metrics']
            print('V7_EVALUATION', split, mode, json.dumps(report['metrics']), flush=True)
        del cache
    (out / 'summary.json').write_text(json.dumps(dict(selection=chosen, metrics=reports), indent=2))
    print('V7_TRAIN_EVAL_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
