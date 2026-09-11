"""Full-information terminal action regression and pairwise candidate ranking."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .train_rl_fast import GPUCache, _take_state
from .networks import DuelingQNetwork

NUM_VIEWS = 4


def targets(logits, labels, classes, present, tie_threshold=0.03):
    """Targets use training labels only; state construction remains label-free."""
    classes = torch.as_tensor(classes, device=logits.device)
    fused = (logits[:, :, None, :] + logits[:, None, :, :]) * .5
    scores = fused[..., classes]
    local = (labels[:, None] == classes[None]).long().argmax(-1)
    true = scores.gather(-1, local[:,None,None,None].expand(-1,4,4,1)).squeeze(-1)
    wrong = scores.clone()
    wrong.scatter_(-1, local[:,None,None,None].expand(-1,4,4,1), -torch.inf)
    margin = true - wrong.max(-1).values
    correct = scores.argmax(-1).eq(local[:,None,None])
    utility = 2 * correct.float() + margin / (20 + margin.abs())
    valid = present[:,:,None] & present[:,None,:] & ~torch.eye(NUM_VIEWS,device=logits.device,dtype=torch.bool)[None]
    contexts = present.flatten().nonzero().flatten()
    u, mask = utility.reshape(-1,4)[contexts], valid.reshape(-1,4)[contexts]
    best = u.masked_fill(~mask,-torch.inf).max(-1).values
    worst = u.masked_fill(~mask,torch.inf).min(-1).values
    gap = best-worst
    regret = ((u-best[:,None]) / gap.clamp_min(.1)[:,None]).clamp(-1,0).masked_fill(~mask,0)
    # Do not force a preference when the measured utilities are effectively tied.
    pairs = mask[:,:,None] & mask[:,None,:] & ((u[:,:,None]-u[:,None,:]) > tie_threshold)
    corr = correct.reshape(-1,4)[contexts]
    informative = (corr & mask).any(-1) & ((~corr) & mask).any(-1)
    # Treat modest but real utility gaps as informative; otherwise most
    # candidate ranking signal is discarded before training.
    informative |= gap > tie_threshold
    return contexts, mask, regret, pairs, informative, u


@torch.inference_mode()
def evaluate(net, cache, present, classes, protocol, seed=20260909):
    """Same starts for all baselines; only valid real views can be selected."""
    n = cache.num_episodes
    device = cache.device
    idx = torch.arange(n,device=device)
    rng = np.random.default_rng(seed)
    counts = present.sum(-1).cpu().numpy()
    start_np = np.zeros(n,dtype=np.int64) if protocol=='fixed0' else np.array([rng.integers(k) for k in counts])
    start = torch.as_tensor(start_np,device=device)
    mask = present.clone()
    mask[idx,start] = False
    action = torch.empty(n,dtype=torch.long,device=device)
    net.eval()
    for b in range(0,n,512):
        e=idx[b:b+512]
        state=cache.build_states(e,start[e])
        q=net(state).float().masked_fill(~mask[e],-torch.inf)
        action[e]=q.argmax(-1)
    rand = torch.tensor([rng.choice(np.flatnonzero(m)) for m in mask.cpu().numpy()],device=device)
    fixed = torch.tensor([1 if m[1] else np.flatnonzero(m)[0] for m in mask.cpu().numpy()],device=device)
    fused=(cache.logits[idx,start,None,:]+cache.logits)*.5
    cs=torch.as_tensor(classes,device=device)
    scores=fused[...,cs]
    target=(cache.labels[:,None]==cs).long().argmax(-1)
    pred=cs[scores.argmax(-1)]
    correct=pred.eq(cache.labels[:,None])
    true=fused.gather(-1,cache.labels[:,None,None].expand(-1,4,1)).squeeze(-1)
    # Keep historical oracle definition and expose actual accuracy ceiling separately.
    oracle=true.masked_fill(~mask,-torch.inf).argmax(-1)
    choices={'single':start,'fixed':fixed,'random':rand,'policy':action,'oracle':oracle}
    metrics={}
    for name,a in choices.items():
        s=cache.logits[idx,start][:,cs] if name=='single' else scores[idx,a]
        p=s.softmax(-1)
        ok=cs[s.argmax(-1)].eq(cache.labels)
        top=cs[s.topk(min(5,len(classes)),-1).indices].eq(cache.labels[:,None]).any(-1)
        metrics[name]={'top1':ok.float().mean().item(),'top5':top.float().mean().item(),
                       'mean_entropy':(-(p*p.clamp_min(1e-12).log()).sum(-1)/np.log(len(classes))).mean().item(),
                       'mean_movement_cost':cache.cost[idx,start,a].mean().item(),
                       'by_view_count':{str(k):ok[present.sum(-1)==k].float().mean().item() for k in (2,3,4) if (present.sum(-1)==k).any()}}
    expected=(correct.float()*mask).sum(-1)/mask.sum(-1)
    metrics['random_exact']={'top1':expected.mean().item(),'by_view_count':{str(k):expected[present.sum(-1)==k].mean().item() for k in (2,3,4) if (present.sum(-1)==k).any()}}
    ceiling=(correct&mask).any(-1).float().mean().item()
    assert mask[idx,action].all()
    return {'protocol':protocol,'episodes':n,'candidate_classes':classes,'metrics':metrics,
            'classification_oracle_top1':ceiling,'initial_views':start.cpu().tolist(),
            'selected':{k:v.cpu().tolist() for k,v in choices.items()},
            'view_counts':present.sum(-1).cpu().tolist(),'labels':cache.labels.cpu().tolist()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);args=p.parse_args()
    cfg=yaml.safe_load(args.config.read_text());out=Path(cfg['train']['output_dir']);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(cfg['runtime']['seed']);np.random.seed(cfg['runtime']['seed'])
    # Full precision avoids erasing small action gaps with bfloat16 rounding.
    torch.set_float32_matmul_precision('highest')
    device=torch.device(cfg['runtime']['device']);root=Path(cfg['cache']['root'])
    train=GPUCache(root/'dqn_train',device); val=GPUCache(root/'val',device)
    present=torch.from_numpy(np.load(root/'dqn_train/view_valid.npy')).to(device)
    vp=torch.from_numpy(np.load(root/'val/view_valid.npy')).to(device)
    seen=cfg['evaluation']['seen_class_ids'];assert not set(train.labels.cpu().tolist())&set(cfg['evaluation']['unseen_class_ids'])
    ranking=cfg.get('ranking',{})
    tie_threshold=float(ranking.get('near_tie_utility_threshold',0.03))
    context,mask,reward,pairs,informative,utility=targets(
        train.logits,train.labels,seen,present,tie_threshold=tie_threshold)
    states=train.build_states(context//NUM_VIEWS,context%NUM_VIEWS)
    assert torch.equal(states['action_mask'],mask)
    candidate_count=mask.sum(-1)
    two_view_weight=float(ranking.get('two_view_loss_weight',0.1))
    priority_mixture=float(ranking.get('priority_mixture',0.5))
    weights=torch.where(candidate_count<=1,torch.tensor(two_view_weight,device=device),torch.ones_like(candidate_count,dtype=torch.float32))
    uniform=torch.ones(len(context),device=device)/len(context)
    priority=(informative & (candidate_count>1)).float()
    priority_dist=priority/priority.sum() if priority.sum()>0 else uniform
    prob=(1.0-priority_mixture)*uniform+priority_mixture*priority_dist
    # A compact auditable training-only table with every starting and destination slot.
    np.savez_compressed(out/'training_action_table.npz',contexts=context.cpu().numpy(),valid=mask.cpu().numpy(),
                        utility=utility.cpu().numpy(),target_regret=reward.cpu().numpy(),rank_pairs=pairs.cpu().numpy(),
                        sample_weight=weights.cpu().numpy(),sampling_probability=prob.cpu().numpy())
    # Keep an auditable table with every valid start/candidate action.
    metadata=json.loads((root/'dqn_train'/'metadata.json').read_text())
    with (out/'training_action_table.jsonl').open('w',encoding='utf-8') as table:
        for row_id,ctx in enumerate(context.cpu().tolist()):
            episode,start=divmod(int(ctx),NUM_VIEWS)
            views=metadata['episodes'][episode]['views']
            candidates=[int(x) for x in mask[row_id].nonzero(as_tuple=False).flatten().cpu().tolist()]
            table.write(json.dumps({
                'episode_id':metadata['episodes'][episode].get('episode_id',episode),
                'base_sample':metadata['episodes'][episode].get('base_sample'),
                'start_slot':start,
                'start_camera':views[start].get('camera') if start<len(views) else None,
                'candidate_slots':candidates,
                'candidate_cameras':[views[x].get('camera') for x in candidates],
                'utility':[float(utility[row_id,x]) if x in candidates else None for x in range(NUM_VIEWS)],
                'target_regret':[float(reward[row_id,x]) if x in candidates else None for x in range(NUM_VIEWS)],
                'distinct_pairs':[[int(a),int(b)] for a,b in pairs[row_id].nonzero(as_tuple=False).cpu().tolist()],
                'informative':bool(informative[row_id]),
                'sample_weight':float(weights[row_id]),
            },ensure_ascii=False)+'\n')
    net=DuelingQNetwork(4,cfg['state']).to(device)
    optimizer=torch.optim.AdamW(net.parameters(),lr=cfg['agent']['learning_rate'],weight_decay=cfg['agent']['weight_decay'])
    compiled=torch.compile(net,mode='reduce-overhead')
    batch=cfg['fast']['batch_size'];updates=cfg['fast']['updates_per_epoch'];best=-float('inf')
    (out/'config_resolved.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    print('TRAIN_START',json.dumps({'contexts':len(context),'informative':int(informative.sum()),
        'informative_fraction':float(informative.float().mean()),'two_view_contexts':int((candidate_count<=1).sum()),
        'two_view_weight':two_view_weight,'tie_threshold':tie_threshold,'batch':batch,
        'updates_per_epoch':updates}),flush=True)
    with (out/'train.log').open('w',buffering=1) as log:
        for epoch in range(1,cfg['train']['epochs']+1):
            net.train();begun=time.monotonic();losses=[]
            for _ in range(updates):
                ids=torch.multinomial(prob,batch,replacement=True)
                q=compiled(_take_state(states,ids)).float()
                reg=(F.smooth_l1_loss(q,reward[ids],reduction='none')*mask[ids]).sum(-1)/mask[ids].sum(-1)
                diff=q[:,:,None]-q[:,None,:]
                required=(0.25 + reward[ids,:,None]-reward[ids,None,:]).clamp_min(0)
                eligible=pairs[ids]
                rank=(F.relu(required-diff)*eligible).sum((1,2))/eligible.sum((1,2)).clamp_min(1)
                rank_weight=float(ranking.get('rank_loss_weight',2.0))
                loss=((reg+rank_weight*rank)*weights[ids]).mean()
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),10.);optimizer.step();losses.append(loss.detach())
            torch.cuda.synchronize();seconds=time.monotonic()-begun
            validations={mode:evaluate(net,val,vp,seen,mode) for mode in ('fixed0','random')}
            score=sum(v['metrics']['policy']['top1'] for v in validations.values())/2
            row={'epoch':epoch,'loss':torch.stack(losses).mean().item(),'update_seconds':seconds,'sample_updates_per_second':batch*updates/seconds,
                 'validation':{k:v['metrics'] for k,v in validations.items()},'selection_score':score}
            print(json.dumps(row),flush=True);log.write(json.dumps(row)+'\n')
            checkpoint={'online':net.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'config':cfg,'selection_score':score}
            torch.save(checkpoint,out/'last.pt')
            if score>best:
                best=score;torch.save(checkpoint,out/'best.pt')
    del train,states,val
    for split in ('val','seen_test','test'):
        cache=GPUCache(root/split,device);pv=torch.from_numpy(np.load(root/split/'view_valid.npy')).to(device)
        classes=cfg['evaluation']['unseen_class_ids'] if split=='test' else seen
        for ck in ('best','last'):
            net.load_state_dict(torch.load(out/(ck+'.pt'),map_location=device,weights_only=False)['online'])
            for mode in ('fixed0','random'):
                result=evaluate(net,cache,pv,classes,mode)
                result['coverage']=json.loads((root/split/'coverage.json').read_text())
                (out/f'evaluation_{split}_{mode}_{ck}.json').write_text(json.dumps(result,indent=2))
                print('EVAL',split,mode,ck,json.dumps(result['metrics']),flush=True)
        del cache
    print('RANK_TRAIN_EVAL_COMPLETE',flush=True)


if __name__=='__main__':main()
