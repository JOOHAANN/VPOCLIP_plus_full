"""Re-train epochs 1-4 of v8-v11 and retain every early checkpoint.

This is isolated from completed experiments. For v8/v9/v11, the seed used for
each early epoch is selected by that epoch's validation score. For v10, the
three same-epoch members are averaged, matching its ensemble definition.
Unseen test labels are read only after this validation-based choice.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .causal_rank_v6 import causal_state, evaluate, task_banks
from .causal_rank_v7 import utility
from .four_ideas_models import Ensemble, decision_loss, make_model
from .run_four_ideas import SEEDS, VERSIONS, validation, verify_v6
from .train_rl_fast import GPUCache


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'work_dir/active_view_four_ideas_early_epoch_probe_new50_5'
EPOCHS = 4


def atomic_torch_save(obj, path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)


def dump(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def main():
    verify_v6()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / 'logs/rl_candidate_rank_v6_new50_5/config.yaml').read_text())
    cfg['early_probe'] = dict(versions=VERSIONS, seeds=SEEDS, epochs=EPOCHS,
                              batch_size=8192, updates_per_epoch=48,
                              checkpoint_policy='every_epoch_1_to_4',
                              seed_selection='validation_only_per_epoch',
                              v10='same_epoch_mean_of_three_members')
    (OUT / 'config_resolved.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    device = torch.device(cfg['runtime']['device'])
    cache_root = Path(cfg['cache']['root'])
    train, val = [GPUCache(cache_root / x, device) for x in ('dqn_train', 'val')]
    present, vp = [torch.from_numpy(np.load(cache_root / x / 'view_valid.npy')).to(device) for x in ('dqn_train', 'val')]
    pool = cfg['v6']['policy_train_classes']; hold = cfg['v6']['policy_holdout_classes']
    seen = cfg['evaluation']['seen_class_ids']; unseen = cfg['evaluation']['unseen_class_ids']
    flat = (present & torch.isin(train.labels, torch.tensor(pool, device=device))[:, None]).flatten().nonzero().flatten()
    episodes, starts = (flat // 4).repeat(4), (flat % 4).repeat(4)
    labels = train.labels[episodes]
    freq = torch.bincount(labels, minlength=55).float()
    uniform = 1 / freq[labels]; uniform /= uniform.sum()
    proxy_eps = torch.isin(val.labels, torch.tensor(hold, device=device)).nonzero().flatten().repeat(4)
    proxy_bank = task_banks(val, proxy_eps, hold, torch.Generator(device=device).manual_seed(90117), 0)
    all_runs = {}
    for version in VERSIONS:
        version_out = OUT / version
        version_out.mkdir(exist_ok=True)
        run_records = {}
        for seed in SEEDS:
            run_out = version_out / f'seed_{seed}'
            run_out.mkdir(exist_ok=True)
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            gen = torch.Generator(device=device).manual_seed(seed)
            net = make_model(version).to(device)
            compiled = torch.compile(net, mode='reduce-overhead')
            opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=.03)
            records = []
            for epoch in range(1, EPOCHS + 1):
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
                    opt.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 2.); opt.step()
                    losses.append(loss.detach())
                score, regular, proxy = validation(net, val, vp, seen, hold, proxy_eps, proxy_bank)
                record = dict(version=version, seed=seed, epoch=epoch,
                              loss=torch.stack(losses).mean().item(), selective=int(selective.sum()),
                              seconds=time.monotonic() - started, selection=score,
                              validation=regular, proxy=proxy)
                records.append(record)
                atomic_torch_save(dict(online=net.state_dict(), optimizer=opt.state_dict(),
                                       generator=gen.get_state(), cpu_rng=torch.get_rng_state(),
                                       cuda_rng=torch.cuda.get_rng_state(), epoch=epoch,
                                       version=version, seed=seed, record=record),
                                  run_out / f'epoch_{epoch:02d}.pt')
                with (run_out / 'train.log').open('a') as f:
                    f.write(json.dumps(record) + '\n')
                print(json.dumps(dict(version=version, seed=seed, epoch=epoch,
                                      loss=record['loss'], selection=score)), flush=True)
            dump(run_out / 'complete.json', dict(version=version, seed=seed, epochs=EPOCHS,
                                                 checkpoints=[f'epoch_{i:02d}.pt' for i in range(1, EPOCHS + 1)]))
            run_records[str(seed)] = records
            del compiled, net, opt
        all_runs[version] = run_records

        test_cache = GPUCache(cache_root / 'test', device)
        test_present = torch.from_numpy(np.load(cache_root / 'test/view_valid.npy')).to(device)
        epoch_results = {}
        for epoch in range(1, EPOCHS + 1):
            members = []
            for seed in SEEDS:
                net = make_model(version).to(device)
                saved = torch.load(version_out / f'seed_{seed}' / f'epoch_{epoch:02d}.pt', map_location=device, weights_only=False)
                net.load_state_dict(saved['online']); net.eval(); members.append(net)
            if version == 'v10':
                selected_net = Ensemble(members).to(device)
                chosen = 'same_epoch_mean_of_three_members'
            else:
                chosen_seed = max(SEEDS, key=lambda seed: all_runs[version][str(seed)][epoch - 1]['selection'])
                selected_net = members[SEEDS.index(chosen_seed)]
                chosen = dict(seed=chosen_seed, validation_score=all_runs[version][str(chosen_seed)][epoch - 1]['selection'])
            selected_net.eval()
            result = {}
            for protocol in ('fixed0', 'random'):
                report = evaluate(selected_net, test_cache, test_present, unseen, protocol)
                result[protocol] = report['metrics']
            epoch_results[str(epoch)] = dict(chosen=chosen, metrics=result)
            print('EARLY_EVALUATION', version, epoch, json.dumps(epoch_results[str(epoch)]), flush=True)
            del selected_net, members, test_cache
            test_cache = GPUCache(cache_root / 'test', device)
        del test_cache
        dump(version_out / 'early_test_summary.json', dict(version=version, epochs=epoch_results,
                                                          test_classes=unseen, test_episodes=414,
                                                          selection='validation_only_per_epoch'))
    dump(OUT / 'summary.json', dict(versions=all_runs, early_test_summaries={
        v: json.loads((OUT / v / 'early_test_summary.json').read_text()) for v in VERSIONS
    }))
    verify_v6()
    print('EARLY_EPOCH_PROBE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
