"""Causal second-view selection: current observation and known geometry only.

Future-view logits are allowed only in offline training targets and evaluation.
Policy holdout actions were seen by VPOCLIP: this is a policy-transfer proxy,
not a claim of strict recognizer zero-shot validation.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import yaml

from .train_rl_fast import GPUCache


def class_mask(classes, n, device):
    mask=torch.zeros((n,55),dtype=torch.bool,device=device)
    mask[:,classes]=True
    return mask


def task_banks(cache, episodes, pool, generator, full_fraction=.25):
    """Seen-only training/proxy tasks; true label is included by task construction."""
    allowed=class_mask(pool,len(episodes),cache.device)
    labels=cache.labels[episodes]
    assert allowed[torch.arange(len(episodes),device=cache.device),labels].all()
    noise=torch.rand(allowed.shape,device=cache.device,generator=generator).masked_fill(~allowed,-1.)
    noise.scatter_(1,labels[:,None],2.)
    subset=torch.zeros_like(allowed).scatter_(1,noise.topk(5,-1).indices,True)
    full=torch.rand((len(episodes),1),device=cache.device,generator=generator)<full_fraction
    return torch.where(full,allowed,subset)


def causal_state(cache, episodes, starts, bank):
    """No access to labels, prototypes, or unvisited visual observations."""
    current=cache.logits[episodes,starts]
    masked=current.masked_fill(~bank,-torch.inf)
    top=masked.topk(5,-1).values
    gaps=((top-top[:,:1])/20).clamp(-5,0)
    p=masked.softmax(-1);pt=(masked/10).softmax(-1)
    ent=lambda x: -(x*x.clamp_min(1e-12).log()).sum(-1,keepdim=True)/bank.sum(-1,keepdim=True).float().log()
    evidence=torch.cat((gaps,p.topk(5,-1).values,pt.topk(5,-1).values,
                        ent(p),ent(pt),(top[:,:1]-top[:,1:2]).tanh(),bank.sum(-1,keepdim=True)/55),-1)
    geometry=cache.geometry[episodes,:,:2]
    source=cache.geometry[episodes,starts,:2]
    sin_delta=geometry[...,0]*source[:,None,1]-geometry[...,1]*source[:,None,0]
    cos_delta=(geometry*source[:,None]).sum(-1)
    reachable=cache.reachable[episodes,starts].clone()
    reachable[torch.arange(len(episodes),device=cache.device),starts]=False
    candidate=torch.cat((geometry,torch.stack((sin_delta,cos_delta),-1),
                         source[:,None].expand(-1,4,-1),reachable[...,None].float()),-1)
    return {'z':cache.z[episodes,starts], 'pose':cache.pose[episodes,starts].clamp(-2,2),
            'object':cache.object_map[episodes,starts].clamp_min(0).log1p().clamp_max(5)/math.log(11),
            'quality':cache.quality[episodes,starts].clamp(0,1),
            'evidence':evidence,'candidate':candidate,'mask':reachable}


class CausalRanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.z=nn.Sequential(nn.Linear(512,128),nn.LayerNorm(128),nn.GELU())
        self.pose=nn.Sequential(nn.Linear(442,64),nn.LayerNorm(64),nn.GELU())
        self.obj=nn.Sequential(nn.Linear(1800,64),nn.LayerNorm(64),nn.GELU())
        self.context=nn.Sequential(nn.Linear(128+64+64+4+19,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(.15),nn.Linear(256,128),nn.GELU())
        self.geometry=nn.Sequential(nn.Linear(7,64),nn.GELU(),nn.Linear(64,128),nn.GELU())
        self.score=nn.Sequential(nn.Linear(384,128),nn.GELU(),nn.Linear(128,1))

    def forward(self,s):
        h=self.context(torch.cat((self.z(s['z']),self.pose(s['pose']),self.obj(s['object']),s['quality'],s['evidence']),-1))
        g=self.geometry(s['candidate'])
        h=h[:,None].expand(-1,4,-1)
        return self.score(torch.cat((h,g,h*g),-1)).squeeze(-1)


def supervision(cache, episodes, starts, bank, valid):
    """Correct candidate sets are tied; never invent a unique winner among them."""
    scores=(cache.logits[episodes,starts,None,:]+cache.logits[episodes])*.5
    scores=scores.masked_fill(~bank[:,None,:],-torch.inf)
    correct=scores.argmax(-1).eq(cache.labels[episodes,None]) & valid
    has_correct=correct.any(-1)
    selective=has_correct & ((~correct)&valid).any(-1)
    # The target is a set, not an arbitrarily chosen highest-confidence action.
    target=torch.where(has_correct[:,None],correct.float()/correct.sum(-1,keepdim=True).clamp_min(1),
                       valid.float()/valid.sum(-1,keepdim=True))
    return target, correct, selective


def selection_loss(q, target, correct, selective, valid):
    safe=q.masked_fill(~valid,-1e4)
    logp=F.log_softmax(safe,-1)
    ce=-(target*logp).sum(-1)
    # BCE encourages calibrated action quality while the categorical term ranks.
    bce=(F.binary_cross_entropy_with_logits(q,correct.float(),reduction='none')*valid).sum(-1)/valid.sum(-1)
    weight=torch.where(selective,1.,.05)*torch.where(valid.sum(-1)==1,.1,1.)
    return ((ce+.25*bce)*weight).sum()/weight.sum().clamp_min(1e-6)


@torch.inference_mode()
def evaluate(net,cache,present,classes,protocol,episodes=None,bank=None,seed=20260909):
    episodes=torch.arange(cache.num_episodes,device=cache.device) if episodes is None else episodes
    n=len(episodes);rows=torch.arange(n,device=cache.device)
    real=present[episodes]
    rng=np.random.default_rng(seed)
    starts=torch.tensor([0 if protocol=='fixed0' else rng.choice(np.flatnonzero(v)) for v in real.cpu().numpy()],device=cache.device)
    bank=class_mask(classes,n,cache.device) if bank is None else bank
    valid=real.clone();valid[rows,starts]=False
    actions=torch.empty(n,dtype=torch.long,device=cache.device)
    net.eval()
    for b in range(0,n,512):
        state=causal_state(cache,episodes[b:b+512],starts[b:b+512],bank[b:b+512])
        assert torch.equal(state['mask'],valid[b:b+512])
        actions[b:b+512]=net(state).masked_fill(~state['mask'],-torch.inf).argmax(-1)
    rand=torch.tensor([rng.choice(np.flatnonzero(v)) for v in valid.cpu().numpy()],device=cache.device)
    fixed=torch.tensor([1 if v[1] else np.flatnonzero(v)[0] for v in valid.cpu().numpy()],device=cache.device)
    fused=(cache.logits[episodes,starts,None,:]+cache.logits[episodes])*.5
    score=fused.masked_fill(~bank[:,None],-torch.inf)
    labels=cache.labels[episodes]
    true=score.gather(-1,labels[:,None,None].expand(-1,4,1)).squeeze(-1)
    oracle=true.masked_fill(~valid,-torch.inf).argmax(-1)
    correct=score.argmax(-1).eq(labels[:,None])
    expected=(correct*valid).sum(-1)/valid.sum(-1)
    choices={'single':starts,'fixed':fixed,'random':rand,'policy':actions,'oracle':oracle}
    metrics={};policy_correct=None
    for name,a in choices.items():
        s=cache.logits[episodes,starts].masked_fill(~bank,-torch.inf) if name=='single' else score[rows,a]
        ok=s.argmax(-1).eq(labels);p=s.softmax(-1)
        metrics[name]={'top1':ok.float().mean().item(),
                       'top5':s.topk(5,-1).indices.eq(labels[:,None]).any(-1).float().mean().item(),
                       'mean_entropy':(-(p*p.clamp_min(1e-12).log()).sum(-1)/bank.sum(-1).float().log()).mean().item(),
                       'mean_movement_cost':cache.cost[episodes,starts,a].mean().item(),
                       'by_view_count':{str(k):ok[real.sum(-1)==k].float().mean().item() for k in (2,3,4) if (real.sum(-1)==k).any()}}
        if name=='policy':policy_correct=ok
    metrics['random_exact']={'top1':expected.mean().item()}
    per_class={str(int(c)): {'count':int((labels==c).sum()),'policy':policy_correct[labels==c].float().mean().item(),
                           'random_exact':expected[labels==c].mean().item()} for c in labels.unique()}
    # Equal weight per action class; two-view episodes cannot inflate selector gains.
    can_choose=real.sum(-1)>2
    gains=[(policy_correct[(labels==c)&can_choose].float()-expected[(labels==c)&can_choose]).mean().item()
           for c in labels.unique() if ((labels==c)&can_choose).any()]
    return {'protocol':protocol,'episodes':n,'candidate_classes':classes,'metrics':metrics,'per_class':per_class,
            'macro_choice_gain':float(np.mean(gains)) if gains else 0.,
            'classification_oracle_top1':(correct&valid).any(-1).float().mean().item(),
            'initial_views':starts.cpu().tolist(),'selected':{k:v.cpu().tolist() for k,v in choices.items()},
            'labels':labels.cpu().tolist(),'episode_indices':episodes.cpu().tolist(),'view_counts':real.sum(-1).cpu().tolist()}


def train_seed(cfg,seed):
    torch.set_num_threads(4);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision('highest')
    device=torch.device(cfg['runtime']['device']);root=Path(cfg['cache']['root'])
    out=Path(cfg['train']['output_dir'])/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return json.loads((out/'complete.json').read_text())
    train=GPUCache(root/'dqn_train',device);val=GPUCache(root/'val',device)
    present=torch.from_numpy(np.load(root/'dqn_train/view_valid.npy')).to(device)
    vp=torch.from_numpy(np.load(root/'val/view_valid.npy')).to(device)
    pool=cfg['v6']['policy_train_classes'];hold=cfg['v6']['policy_holdout_classes'];seen=cfg['evaluation']['seen_class_ids']
    unseen=cfg['evaluation']['unseen_class_ids']
    assert not (set(pool)&set(hold) or (set(pool)|set(hold))&set(unseen))
    eligible=present & torch.isin(train.labels,torch.tensor(pool,device=device))[:,None]
    contexts=eligible.flatten().nonzero().flatten();episodes,starts=contexts//4,contexts%4
    generator=torch.Generator(device=device).manual_seed(seed)
    # Finite, reproducible tasks for efficient GPU-resident sampling.
    episodes=episodes.repeat(4);starts=starts.repeat(4)
    banks=task_banks(train,episodes,pool,generator)
    state=causal_state(train,episodes,starts,banks)
    target,correct,selective=supervision(train,episodes,starts,banks,state['mask'])
    labs=train.labels[episodes];freq=torch.bincount(labs,minlength=55).float()
    uniform=1/freq[labs];uniform/=uniform.sum()
    priority=uniform*selective.float();priority=priority/priority.sum() if priority.sum()>0 else uniform
    probability=.5*uniform+.5*priority
    np.savez_compressed(out/'training_targets.npz',episodes=episodes.cpu().numpy(),starts=starts.cpu().numpy(),
                        task_classes=banks.cpu().numpy(),valid=state['mask'].cpu().numpy(),
                        correct=correct.cpu().numpy(),target=target.cpu().numpy(),selective=selective.cpu().numpy())
    proxy_eps=torch.isin(val.labels,torch.tensor(hold,device=device)).nonzero().flatten()
    assert len(proxy_eps)>0
    proxy_gen=torch.Generator(device=device).manual_seed(90117)
    proxy_bank=task_banks(val,proxy_eps,hold,proxy_gen,full_fraction=0)
    net=CausalRanker().to(device);compiled=torch.compile(net,mode='reduce-overhead')
    optimizer=torch.optim.AdamW(net.parameters(),lr=cfg['v6']['learning_rate'],weight_decay=.01)
    epochs=cfg['train']['epochs'];scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=1e-5)
    batch=cfg['fast']['batch_size'];updates=cfg['fast']['updates_per_epoch'];best=-float('inf');start_epoch=1
    if (out/'last.pt').exists():
        saved=torch.load(out/'last.pt',map_location=device,weights_only=False);net.load_state_dict(saved['online'])
        optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        generator.set_state(saved['sampler_state']);torch.set_rng_state(saved['cpu_rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
        start_epoch=saved['epoch']+1;best=saved['best_score']
    audit={'seed':seed,'contexts':len(episodes),'selective':int(selective.sum()),'proxy_episodes':len(proxy_eps),
           'policy_train_classes':pool,'policy_holdout_classes':hold,'unseen_classes':unseen,
           'proxy_note':'Policy-held-out seen classes; VPO was trained on them. Not strict recognizer ZSL.',
           'causal_inputs_only':True,'candidate_feature_dim':7,'batch':batch,'updates_per_epoch':updates}
    (out/'audit.json').write_text(json.dumps(audit,indent=2));(out/'config_resolved.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    print('V6_TRAIN_START',json.dumps(audit),flush=True)
    with (out/'train.log').open('a',buffering=1) as log:
        for epoch in range(start_epoch,epochs+1):
            net.train();begun=time.monotonic();losses=[]
            for _ in range(updates):
                ids=torch.multinomial(probability,batch,replacement=True,generator=generator)
                s={k:v[ids] for k,v in state.items()}
                q=compiled(s)
                loss=selection_loss(q,target[ids],correct[ids],selective[ids],s['mask'])
                optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step();losses.append(loss.detach())
            torch.cuda.synchronize();seconds=time.monotonic()-begun;scheduler.step()
            regular={mode:evaluate(net,val,vp,seen,mode) for mode in ('fixed0','random')}
            proxy={mode:evaluate(net,val,vp,hold,mode,proxy_eps,proxy_bank) for mode in ('fixed0','random')}
            proxy_gain=np.mean([v['macro_choice_gain'] for v in proxy.values()]);seen_gain=np.mean([v['macro_choice_gain'] for v in regular.values()])
            selection=float(.75*proxy_gain+.25*seen_gain)
            record={'epoch':epoch,'seed':seed,'loss':torch.stack(losses).mean().item(),'update_seconds':seconds,
                    'draws_per_second':batch*updates/seconds,'proxy_gain':float(proxy_gain),'seen_gain':float(seen_gain),
                    'selection_score':selection,'validation':{k:v['metrics'] for k,v in regular.items()},
                    'proxy_5way':{k:v['metrics'] for k,v in proxy.items()}}
            print(json.dumps(record),flush=True);log.write(json.dumps(record)+'\n')
            improved=selection>best
            if improved:best=selection
            saved={'online':net.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                   'epoch':epoch,'best_score':best,'selection_score':selection,'config':cfg,'sampler_state':generator.get_state(),
                   'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()}
            torch.save(saved,out/'last.pt')
            if improved:torch.save(saved,out/'best.pt')
    result={'seed':seed,'best_selection_score':best,'output':str(out)}
    (out/'complete.json').write_text(json.dumps(result))
    return result


def final_evaluation(cfg,summaries):
    # Choose the run by validation only, before loading any test cache.
    chosen=max(summaries,key=lambda x:x['best_selection_score'])
    root=Path(cfg['cache']['root']);out=Path(cfg['train']['output_dir']);device=torch.device(cfg['runtime']['device'])
    (out/'selection.json').write_text(json.dumps({'chosen':chosen,'runs':summaries,'criterion':'validation_only'},indent=2))
    reports={}
    for split in ('val','seen_test','test'):
        cache=GPUCache(root/split,device);present=torch.from_numpy(np.load(root/split/'view_valid.npy')).to(device)
        classes=cfg['evaluation']['unseen_class_ids'] if split=='test' else cfg['evaluation']['seen_class_ids']
        for ck in ('best','last'):
            net=CausalRanker().to(device);saved=torch.load(Path(chosen['output'])/(ck+'.pt'),map_location=device,weights_only=False);net.load_state_dict(saved['online'])
            for mode in ('fixed0','random'):
                result=evaluate(net,cache,present,classes,mode)
                result.update(coverage=json.loads((root/split/'coverage.json').read_text()),checkpoint_epoch=saved['epoch'],selected_seed=chosen['seed'])
                path=out/f'evaluation_{split}_{mode}_{ck}.json';path.write_text(json.dumps(result,indent=2))
                reports[f'{split}_{mode}_{ck}']=result['metrics']
                print('V6_EVALUATION',split,mode,ck,json.dumps(result['metrics']),flush=True)
        del cache
    (out/'summary.json').write_text(json.dumps({'selection':chosen,'metrics':reports},indent=2))
    print('V6_TRAIN_EVAL_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);args=p.parse_args()
    cfg=yaml.safe_load(args.config.read_text());results=[]
    for seed in cfg['v6']['seeds']:
        results.append(train_seed(cfg,seed));torch.cuda.empty_cache()
    final_evaluation(cfg,results)


if __name__=='__main__':main()
