"""Isolated, causal ETRI adaptations; not original-dataset reproductions."""
import argparse
import copy
import hashlib
import importlib.util
import itertools
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from rl import fusion_weighted_common_v1 as common
from rl import weighted_fusion_policy_variants_v1 as variants

spec = importlib.util.spec_from_file_location('official_mvselect', HERE / 'upstream/MVSelect/src/models/mvselect.py')
upstream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upstream)
DEVICE = torch.device('cuda:0')
SEEDS = common.TRAIN_SEEDS
BANK = common.SEEN_CLASSES
ANCHOR = ROOT / 'work_dir/weighted_fusion_policy_variant_comparison_v1/new_split_271/v5'
TRUE_UNSEEN = common.TRUE_UNSEEN
PSEUDO_UNSEEN = common.PSEUDO_UNSEEN
EVAL_SEEDS = common.EVAL_SEEDS
DATA_CACHE = variants.NEW_CACHE
GATE_CHECKPOINT = variants.NEW_GATE_CHECKPOINT
OUTPUT_ROOT = HERE
ANCHOR_NAME = 'v5'
TRAIN_EPISODES = 2900
EVAL_MOVES = (1, 2)


def dump(p, x):
    common.dump(Path(p), x)


def mask(raw, ids, path):
    out = raw.valid[ids] & raw.reachable[ids, path[:, -1]]
    return out.scatter(1, path, False)


class MVSelect(nn.Module):
    def __init__(self):
        super().__init__()
        self.selector = upstream.CamSelect(4, 512, aggregation='max')

    def forward(self, raw, ids, path):
        # Gather observed descriptors ONLY. Reproduce official zero-masked max.
        obs = torch.zeros((len(ids), 4, 512), device=DEVICE)
        obs.scatter_(1, path[..., None].expand(-1, -1, 512), raw.z[ids[:, None], path].float())
        visited = torch.zeros((len(ids), 4), device=DEVICE).scatter_(1, path, 1.)
        m = self.selector
        feature = m.feat_branch(obs.amax(1)[..., None, None]).flatten(1)
        q = m.value_head(feature + m.emb_branch(visited @ m.cam_emb))
        return q.masked_fill(~mask(raw, ids, path), -1e9), None


class MissingFrameA2C(nn.Module):
    """Paper equations (1)/(2), independent camera mfLSTMs, actor/critic.

    2-D 17-joint distances replace unavailable 3-D NTU skeletons. A selected
    cached clip is one acquisition block, NOT real elapsed robot time.
    """
    def __init__(self):
        super().__init__()
        self.hidden = 100
        self.cells = nn.ModuleList([nn.LSTMCell(136, 100) for _ in range(4)])
        self.u = nn.Parameter(torch.full((4, 136), .01))
        self.v = nn.Parameter(torch.zeros(4, 136))
        self.null = nn.Parameter(torch.zeros(4, 136))
        self.heads = nn.ModuleList([nn.Linear(100, 55) for _ in range(4)])
        self.actor = nn.Sequential(nn.Linear(404, 512), nn.LeakyReLU(.2), nn.Linear(512, 128), nn.LeakyReLU(.2), nn.Linear(128, 4))
        self.critic = nn.Sequential(nn.Linear(404, 512), nn.LeakyReLU(.2), nn.Linear(512, 128), nn.LeakyReLU(.2), nn.Linear(128, 1))

    def encode(self, raw, ids, path, dropout=False):
        n = len(ids)
        h = [torch.zeros(n, 100, device=DEVICE) for _ in range(4)]
        c = [torch.zeros_like(h[0]) for _ in range(4)]
        last = self.null[None].expand(n, -1, -1)
        delta = torch.zeros(n, 4, device=DEVICE)
        rows = torch.arange(n, device=DEVICE)
        # Only selected cameras' sequences enter this function.
        selected = raw._pairdist[ids[:, None], path]
        for k in range(path.shape[1]):
            camera = path[:, k]
            observed = F.one_hot(camera, 4).bool()
            for t in range(13):
                keep = observed
                if dropout:
                    keep = keep & (torch.rand(n, 4, device=DEVICE) > .5)
                delta = torch.where(keep, 0., delta + 1.)
                fill = torch.exp(-F.relu(delta[..., None] * self.u + self.v))
                estimate = fill * last + (1. - fill) * self.null
                real = selected[:, k, t, None, :].expand(-1, 4, -1)
                x = torch.where(keep[..., None], real, estimate)
                last = torch.where(keep[..., None], real, last)
                for view in range(4):
                    h[view], c[view] = self.cells[view](x[:, view], (h[view], c[view]))
        return torch.cat([*h, delta / 39.], -1), h

    def forward(self, raw, ids, path):
        state, _ = self.encode(raw, ids, path)
        return self.actor(state).masked_fill(~mask(raw, ids, path), -1e9), self.critic(state).squeeze(-1)


def attach(raw):
    p = raw.pose.reshape(-1, 4, 13, 17, 2).float()
    i, j = torch.triu_indices(17, 17, 1, device=DEVICE)
    raw._pairdist = (p[..., i, :] - p[..., j, :]).norm(dim=-1)


@torch.no_grad()
def utility(raw, ids, path, bank):
    # Equal-logit CE, matching frozen-recognizer/post-hoc-fusion anchor protocol.
    z = raw.logits[ids[:, None], path].float().mean(1).index_select(-1, bank)
    return F.log_softmax(z, -1).gather(1, torch.searchsorted(bank, raw.labels[ids])[:, None]).squeeze(1)


@torch.no_grad()
def rollout(model, raw, ids, start, moves):
    path = start[:, None]
    for _ in range(moves):
        q, _ = model(raw, ids, path)
        assert bool(mask(raw, ids, path).any(-1).all())
        path = torch.cat([path, q.argmax(-1)[:, None]], -1)
    return path


@torch.no_grad()
def validate(model, raw):
    bank = common.class_bank(PSEUDO_UNSEEN, DEVICE)
    ids = ((raw.valid.sum(-1) == 4) & torch.isin(raw.labels, bank)).nonzero().flatten()
    start = torch.zeros_like(ids)
    path = rollout(model, raw, ids, start, 2)
    policy = common.path_metrics(raw, ids, path, bank, 'equal_mean', None)['correct'].float()
    random_acc = []
    for a, b in itertools.combinations([1, 2, 3], 2):
        p = torch.tensor([0, a, b], device=DEVICE).expand(len(ids), -1)
        random_acc.append(common.path_metrics(raw, ids, p, bank, 'equal_mean', None)['correct'].float())
    return float((policy - torch.stack(random_acc).mean(0)).mean())


def train(name, seed, raw, val, args):
    out = OUTPUT_ROOT / f'seed_{seed}'
    if (out / 'complete.json').exists():
        return json.loads((out / 'complete.json').read_text())
    out.mkdir(parents=True, exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = (MVSelect() if name == 'mvselect' else MissingFrameA2C()).to(DEVICE)
    bank = common.class_bank(BANK, DEVICE)
    pool = ((raw.valid.sum(-1) == 4) & torch.isin(raw.labels, bank)).nonzero().flatten()
    assert len(pool) == TRAIN_EPISODES, f'anchor mismatch {len(pool)} != {TRAIN_EPISODES}'
    # Important: historical v5 trains all 50 seen classes, including the named
    # pseudo validation classes. It is NOT class-disjoint pseudo-unseen training.
    dump(out / 'audit.json', {'training_episodes': len(pool), 'classes': BANK,
         'validation_classes_overlap_training': PSEUDO_UNSEEN,
         'true_unseen_training_overlap': sorted(set(BANK) & set(TRUE_UNSEEN)),
         'no_future_features': True, 'training_fusion': 'equal_mean', 'final_fusion': list(common.METHODS)})
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    target = copy.deepcopy(model).eval() if name == 'mvselect' else None
    best, best_epoch = -1e9, 0
    # Auxiliary missing-frame classification initializes mfLSTM only. It does
    # NOT replace, fine-tune or supply logits to the final frozen VPOCLIP.
    start_epoch = 1
    if (out / 'last.pt').exists():
        saved = torch.load(out / 'last.pt', map_location=DEVICE, weights_only=False)
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        if target is not None: target.load_state_dict(saved['target'])
        best, best_epoch = saved['best_score'], saved['best_epoch']
        start_epoch = saved['epoch'] + 1
        torch.set_rng_state(saved['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
    if name != 'mvselect' and start_epoch == 1:
        for epoch in range(args.pretrain):
            for _ in range(args.updates):
                ids = pool[torch.randint(len(pool), (args.batch,), device=DEVICE)]
                path = torch.randint(4, (len(ids), 1), device=DEVICE)
                _, h = model.encode(raw, ids, path, dropout=True)
                logits = torch.stack([head(x) for head, x in zip(model.heads, h)], 1)
                selected = logits[torch.arange(len(ids), device=DEVICE), path[:, 0]].index_select(-1, bank)
                loss = F.cross_entropy(selected, torch.searchsorted(bank, raw.labels[ids]))
                optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
            print(json.dumps({'model': name, 'seed': seed, 'pretrain_epoch': epoch + 1, 'loss': float(loss.detach())}), flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        begun = time.monotonic()
        model.train()
        losses = []
        for _ in range(args.updates):
            ids = pool[torch.randint(len(pool), (args.batch,), device=DEVICE)]
            path = torch.randint(4, (len(ids), 1), device=DEVICE)
            previous = utility(raw, ids, path, bank)
            terms, rewards, values, logps, entropies = [], [], [], [], []
            for step in range(2):
                q, value = model(raw, ids, path)
                if name == 'mvselect':
                    eps = max(.05, 1. - epoch / max(args.epochs * .8, 1.))
                    explore = torch.multinomial(mask(raw, ids, path).float(), 1).squeeze(1)
                    action = torch.where(torch.rand(len(ids), device=DEVICE) < eps, explore, q.argmax(-1))
                else:
                    dist = torch.distributions.Categorical(logits=q)
                    action = dist.sample()
                    logps.append(dist.log_prob(action)); entropies.append(dist.entropy()); values.append(value)
                path = torch.cat([path, action[:, None]], -1)
                current = utility(raw, ids, path, bank)
                if name == 'mvselect':
                    with torch.no_grad():
                        future = target(raw, ids, path)[0].amax(-1) if step == 0 else 0.
                        td = current - previous + .99 * future
                    terms.append(F.smooth_l1_loss(q.gather(1, action[:, None]).squeeze(1), td))
                else:
                    rewards.append(current)
                previous = current
            if name != 'mvselect':
                ret = torch.zeros_like(rewards[0])
                for step in reversed(range(2)):
                    ret = rewards[step] + .9 * ret
                    advantage = ret - values[step]
                    terms.append(-(logps[step] * advantage.detach()).mean() + .5 * advantage.square().mean() - .01 * entropies[step].mean())
            loss = torch.stack(terms).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError((name, seed, epoch))
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
            if target is not None:
                with torch.no_grad():
                    for p, t in zip(model.parameters(), target.parameters()):
                        t.lerp_(p, .01)
            losses.append(float(loss.detach()))
        model.eval()
        score = validate(model, val)
        improved = score > best
        if improved: best, best_epoch = score, epoch
        payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'seed': seed,
                   'best_score': best, 'best_epoch': best_epoch, 'torch_rng': torch.get_rng_state(),
                   'cuda_rng': torch.cuda.get_rng_state_all(), 'target': target.state_dict() if target else None}
        torch.save(payload, out / 'last.pt')
        if improved: torch.save(payload, out / 'best.pt')
        row = {'model': name, 'seed': seed, 'epoch': epoch, 'loss': float(np.mean(losses)), 'validation_gain': score,
               'best_score': best, 'seconds': time.monotonic() - begun}
        with (out / 'train.jsonl').open('a') as f: f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    result = {'seed': seed, 'score': best, 'epoch': best_epoch, 'checkpoint': str(out / 'best.pt')}
    dump(out / 'complete.json', result)
    return result


def ci(x):
    # Seed averaging first: 30 random starts are not 30 independent test sets.
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2: x = x.mean(0)
    rng = np.random.default_rng(7201)
    means = x[rng.integers(len(x), size=(5000, len(x)))].mean(1)
    return [float(v) for v in np.quantile(means, [.025, .975])]


@torch.no_grad()
def evaluate(name, selected, raws):
    model = (MVSelect() if name == 'mvselect' else MissingFrameA2C()).to(DEVICE)
    model.load_state_dict(torch.load(selected['checkpoint'], map_location=DEVICE, weights_only=False)['model'])
    model.eval()
    gate = variants.load_gate(GATE_CHECKPOINT)
    report = []
    for group, raw, classes in raws:
        bank = common.class_bank(classes, DEVICE)
        for protocol in ['fixed0', 'fixed1', 'fixed2', 'fixed3', 'random_start']:
            for moves in EVAL_MOVES:
                # Reuse exact anchor episode IDs, initial views AND random paths.
                detail = ANCHOR / group / protocol / 'details'
                base = np.load(detail / f'{protocol}_moves_{moves}_policy.npz')
                rand = np.load(detail / f'{protocol}_moves_{moves}_random.npz')
                assert np.array_equal(base['episode_indices'], rand['episode_indices'])
                assert np.array_equal(base['paths'][:, :, 0], rand['paths'][:, :, 0])
                ids = torch.as_tensor(base['episode_indices'], device=DEVICE).long()
                labels = torch.searchsorted(bank, raw.labels[ids]).cpu().numpy()
                predictions = {m: [] for m in common.METHODS}
                paths = []
                # Fixed protocols have identical input in every eval seed.
                count = 30 if protocol == 'random_start' else 1
                for s in range(count):
                    starts = torch.as_tensor(base['paths'][s, :, 0], device=DEVICE).long()
                    path = rollout(model, raw, ids, starts, moves)
                    paths.append(path.cpu().numpy())
                    for method in common.METHODS:
                        predictions[method].append(common.path_metrics(raw, ids, path, bank, method, gate)['prediction'].cpu().numpy())
                out = OUTPUT_ROOT / f"seed_{selected['seed']}" / 'evaluation' / group
                out.mkdir(parents=True, exist_ok=True)
                details = {'episode_indices': base['episode_indices'], 'labels': labels, 'paths': np.repeat(paths, len(EVAL_SEEDS), axis=0) if count == 1 else np.asarray(paths), 'eval_seeds': np.asarray(EVAL_SEEDS)}
                for method in common.METHODS:
                    pred = np.asarray(predictions[method])
                    if count == 1: pred = np.repeat(pred, len(EVAL_SEEDS), axis=0)
                    correct = pred == labels
                    random_correct = rand[f'prediction_{method}'] == labels
                    anchor_correct = base[f'prediction_{method}'] == labels
                    details[f'prediction_{method}'] = pred
                    report.append({'group': group, 'protocol': protocol, 'moves': moves, 'fusion': method, 'n': len(ids),
                        'top1': float(correct.mean()), 'ci95': ci(correct), 'random_top1': float(random_correct.mean()),
                        'anchor_name': ANCHOR_NAME, 'anchor_top1': float(anchor_correct.mean()),
                        'delta_random': float((correct.astype(float) - random_correct).mean()),
                        'delta_random_ci95': ci(correct.astype(float) - random_correct),
                        'delta_anchor': float((correct.astype(float) - anchor_correct).mean()),
                        'delta_anchor_ci95': ci(correct.astype(float) - anchor_correct)})
                np.savez_compressed(out / f'{protocol}_moves_{moves}.npz', **details)
                print(json.dumps({'evaluated': name, 'seed': selected['seed'], 'group': group, 'protocol': protocol, 'moves': moves}), flush=True)
    dump(HERE / name / f"seed_{selected['seed']}" / 'evaluation.json', report)
    return report


def main():
    global BANK, TRUE_UNSEEN, PSEUDO_UNSEEN, EVAL_SEEDS, DATA_CACHE
    global GATE_CHECKPOINT, OUTPUT_ROOT, ANCHOR, ANCHOR_NAME, TRAIN_EPISODES, EVAL_MOVES
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--updates', type=int, default=64)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--pretrain', type=int, default=10)
    p.add_argument('--model', choices=['mvselect', 'mflstm_a2c'], required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    p.add_argument('--cache-root', type=Path, default=DATA_CACHE)
    p.add_argument('--output-root', type=Path, default=None)
    p.add_argument('--anchor-root', type=Path, default=ANCHOR)
    p.add_argument('--anchor-name', default=ANCHOR_NAME)
    p.add_argument('--gate-checkpoint', type=Path, default=GATE_CHECKPOINT)
    p.add_argument('--true-unseen', type=int, nargs='+', default=TRUE_UNSEEN)
    p.add_argument('--seen-classes', type=int, nargs='+', default=BANK)
    p.add_argument('--pseudo-unseen', type=int, nargs='+', default=PSEUDO_UNSEEN)
    p.add_argument('--train-episodes', type=int, default=TRAIN_EPISODES)
    p.add_argument('--eval-seeds', type=int, nargs='+', default=EVAL_SEEDS)
    p.add_argument('--eval-moves', type=int, nargs='+', default=list(EVAL_MOVES))
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    BANK = sorted(set(args.seen_classes))
    TRUE_UNSEEN = sorted(set(args.true_unseen))
    PSEUDO_UNSEEN = sorted(set(args.pseudo_unseen))
    EVAL_SEEDS = list(args.eval_seeds)
    DATA_CACHE = args.cache_root
    GATE_CHECKPOINT = args.gate_checkpoint
    OUTPUT_ROOT = args.output_root or (HERE / args.model)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    ANCHOR = args.anchor_root
    ANCHOR_NAME = args.anchor_name
    TRAIN_EPISODES = args.train_episodes
    EVAL_MOVES = tuple(args.eval_moves)
    dump(OUTPUT_ROOT / 'config.json', {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    })
    train_raw = common.load_raw(DATA_CACHE, 'dqn_train', DEVICE)
    val_raw = common.load_raw(DATA_CACHE, 'val', DEVICE)
    if args.model == 'mflstm_a2c': attach(train_raw); attach(val_raw)
    results = [train(args.model, s, train_raw, val_raw, args) for s in args.seeds]
    selected = max(results, key=lambda x: x['score'])
    dump(OUTPUT_ROOT / 'selection.json', {'chosen': selected, 'runs': results, 'criterion': 'validation fixed0 3-view equal-fusion gain; no true unseen selection', 'anchor_name': ANCHOR_NAME})
    del train_raw, val_raw
    torch.cuda.empty_cache()
    unseen = common.load_raw(DATA_CACHE, 'test', DEVICE)
    seen = common.load_raw(DATA_CACHE, 'seen_test', DEVICE)
    if args.model == 'mflstm_a2c': attach(unseen); attach(seen)
    all_results = {}
    for result in results:
        all_results[str(result['seed'])] = evaluate(args.model, result, [('true_unseen_5way', unseen, TRUE_UNSEEN), ('seen', seen, BANK)])
    dump(OUTPUT_ROOT / 'summary.json', {'selection': selected, 'by_train_seed': all_results, 'anchor_name': ANCHOR_NAME, 'cache_root': str(DATA_CACHE), 'train_episodes': TRAIN_EPISODES})
    print(args.model + '_TRAIN_EVAL_COMPLETE', flush=True)


if __name__ == '__main__': main()
