"""Synthetic sequential replay with persistent camera identities.

Camera slots are remapped between recordings; each subject starts a new
stream. Recording order is shuffled without consulting action labels.
Movement remains the cache's angular proxy, not measured path length.
"""
import json
from pathlib import Path
import numpy as np
import torch
from . import experiment as b

original_load = b.load_raw
original_pool = b.build_transition_pool
original_report = b.write_results_markdown


def load(*args):
    raw, audit = original_load(*args)
    meta = json.loads((args[0] / args[2] / 'metadata.json').read_text())
    ranks = np.load(args[0] / args[2] / 'rank_to_original.npy')
    raw.camera_ids = []
    raw.subjects = []
    for i, ep in enumerate(meta['episodes']):
        source = {v['original_view_index']: v['camera'] for v in ep['views']}
        raw.camera_ids.append([source.get(int(x), '') for x in ranks[i]])
        raw.subjects.append(ep['subject'])
    return raw, audit


def streams(raw, episodes, seed):
    rng = np.random.default_rng(seed)
    groups = {}
    for e in episodes:
        groups.setdefault(raw.subjects[int(e)], []).append(int(e))
    return [rng.permutation(v).tolist() for _, v in sorted(groups.items())]


def assign(raw, ep, a, c, left, right):
    costs = raw.cost[ep].detach().cpu().numpy()
    reach = raw.reachable[ep].detach().cpu().numpy()
    def edge(x, y):
        return 0.0 if x == y else float(costs[x,y]) if reach[x,y] else float('inf')
    first, second = edge(a,left)+edge(c,right), edge(a,right)+edge(c,left)
    return (left,right) if first <= second else (right,left)


def pool(raw, fused, penalty, device):
    p = original_pool(raw, fused, penalty, device)
    episodes = b.eligible_episodes(
        raw, b.POLICY_TRAIN_CLASSES
    ).nonzero().flatten().tolist()
    successors = {}
    for seq in streams(raw, episodes, 731):
        successors.update(zip(seq[:-1], seq[1:]))
    ne = p.episode.cpu().numpy().copy()
    na = p.start_a.cpu().numpy().copy(); nb = p.start_b.cpu().numpy().copy()
    ns = p.action.cpu().numpy().copy(); done = p.done.cpu().numpy().copy()
    e = p.episode.cpu().numpy(); aa = p.start_a.cpu().numpy(); bb = p.start_b.cpu().numpy()
    selected = p.selected.cpu().numpy(); action = p.action.cpu().numpy()
    for k in np.flatnonzero(~p.stage1.cpu().numpy()):
        old = int(e[k]); nxt = successors.get(old)
        ns[k] = -1
        if nxt is None:
            continue
        a,c = assign(raw, old, int(aa[k]), int(bb[k]), int(selected[k]), int(action[k]))
        cams = raw.camera_ids[nxt]
        if raw.camera_ids[old][a] not in cams or raw.camera_ids[old][c] not in cams:
            continue
        ne[k] = nxt; na[k] = cams.index(raw.camera_ids[old][a]); nb[k] = cams.index(raw.camera_ids[old][c]); done[k] = 0
    for key, value in [('next_episode',ne),('next_a',na),('next_b',nb),('next_selected',ns)]:
        setattr(p,key,torch.as_tensor(value,device=device,dtype=torch.long))
    p.done = torch.as_tensor(done,device=device)
    assert ((p.next_selected[p.stage1]) == p.action[p.stage1]).all()
    return p


@torch.inference_mode()
def replay(raw, fused, model, variant, seed, bank, penalty=0):
    episodes = b.eligible_episodes(raw,bank).nonzero().flatten()
    # Batch all counterfactual start pairs for efficient causal lookup.
    starts = torch.cartesian_prod(torch.arange(4,device=raw.device),torch.arange(4,device=raw.device))
    ee = episodes.repeat_interleave(16); aa = starts[:,0].repeat(len(episodes)); bb = starts[:,1].repeat(len(episodes))
    lookup = {}
    if model is not None:
        out = b.dqn_rollout(model,raw,ee,aa,bb,variant)
        for e,a,c,l,r in zip(*(out[k].tolist() for k in ['episodes','start_a','start_b','target_a','target_b'])):
            lookup[e,a,c] = (l,r)
    rng = np.random.default_rng(seed)
    rows = {k:[] for k in ['episodes','start_a','start_b','target_a','target_b']}
    for seq in streams(raw,episodes.tolist(),seed):
        a,c = rng.choice(4,2,replace=False).tolist()
        for step,e in enumerate(seq):
            if step:
                cams = raw.camera_ids[e]
                a,c = cams.index(prev_a),cams.index(prev_b)
            if model is not None:
                l,r = lookup[e,a,c]
            else:
                tt = lambda x: torch.tensor([x],device=raw.device)
                _,valid = b.assignment_matrix(raw,tt(e),tt(a),tt(c))
                legal = [pair for pair in b.PAIR_LIST if valid[0,pair[0],pair[1]]]
                if variant == 'random_pair':
                    l,r = legal[int(rng.integers(len(legal)))]
                else:
                    cams = sorted(raw.camera_ids[e]); offset = 1 if variant=='cyclic_adjacent_pair' else 2
                    wanted = tuple(sorted([raw.camera_ids[e].index(cams[step%4]),raw.camera_ids[e].index(cams[(step+offset)%4])]))
                    l,r = wanted if wanted in legal else legal[0]
            for key,value in zip(rows,[e,a,c,l,r]): rows[key].append(value)
            x,y = assign(raw,e,a,c,l,r)
            prev_a,prev_b = raw.camera_ids[e][x],raw.camera_ids[e][y]
    out = {k:torch.tensor(v,device=raw.device) for k,v in rows.items()}
    metrics = b.pair_metrics(raw,fused,out,bank)
    ids = torch.as_tensor(b.PAIR_INDEX,device=raw.device)[out['target_a'],out['target_b']]
    margin = b.pair_margin(fused[out['episodes'],ids],raw.labels[out['episodes']],bank)
    metrics['mean_reward'] = float(margin.mean())-penalty*metrics['mean_movement_cost']
    return metrics


def validation(model,raw,fused,episodes,a,c,variant,bank,penalty):
    result = replay(raw,fused,model,variant,731,bank,penalty)
    random = replay(raw,fused,None,'random_pair',731,bank,penalty)
    ratio = result['mean_movement_cost']/max(random['mean_movement_cost'],1e-8)
    result['movement_ratio_to_random'] = ratio
    # Select by accuracy subject to the requested validation movement budget.
    result['mean_reward'] = result['accuracy'] if ratio <= .5 else -ratio
    return result


def evaluation(raw,fused,policies,variants,seed,bank):
    methods = {v:replay(raw,fused,policies[v],v,seed,bank) for v in variants}
    for v in ['random_pair','cyclic_pair','cyclic_adjacent_pair']:
        methods[v] = replay(raw,fused,None,v,seed,bank)
    return {'seed':seed,'bank':bank,'methods':methods,'protocol':'subject-wise synthetic sequential replay'}


def report(root,metadata,summary):
    metadata['robot_protocol'] = 'persistent camera identities; shuffled recordings within each subject; reset only at subject boundary'
    metadata['training_transition'] = 'pair completion bootstraps into the next recording at assigned camera positions'
    metadata['movement_target'] = 'validation mean cost <= 0.5 * random; angular proxy, not meters'
    metadata['sequence_limit'] = 'synthetic ordering; no real cross-action timestamps'
    b.dump_json(root/'config_resolved.json',metadata)
    lines = ['# Sequential replay results','','Synthetic subject-wise streams; 30 evaluation seeds. Movement is an angular proxy.','', '|Method|Accuracy|95% CI|Move cost|Ratio to random|','|---|---:|---:|---:|---:|']
    ref = summary['random_pair']['mean_movement_cost']['mean']
    for v,s in summary.items():
        a=s['accuracy']; m=s['mean_movement_cost']['mean']
        lines.append(f"|{v}|{100*a['mean']:.2f}%|[{100*a['ci95_low']:.2f}, {100*a['ci95_high']:.2f}]|{m:.4f}|{m/ref:.3f}|")
    (root/'RESULTS.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    b.load_raw=load; b.build_transition_pool=pool
    b.evaluate_policy_once=validation; b.test_seed_evaluation=evaluation
    b.write_results_markdown=report
    b.main()
