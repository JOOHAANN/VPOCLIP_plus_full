"""Preserve v5 and launch the audited causal v6 experiment."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import yaml


def telemetry(path,stop):
    with path.open('a',buffering=1) as f:
        while not stop.is_set():
            p=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True)
            if p.returncode==0:f.write(json.dumps({'time':time.time(),'gpu':p.stdout.strip()})+'\n')
            stop.wait(2)


def main():
    root=Path(__file__).resolve().parents[1]
    logs=root/'logs/rl_candidate_rank_v6_new50_5';logs.mkdir(exist_ok=True)
    config_path=logs/'config.yaml'
    if not config_path.exists():
        cfg=yaml.safe_load((root/'work_dir/active_view_candidate_rank_v5_new50_5/config_resolved.yaml').read_text())
        cfg['ranking']['reuse_cache']=cfg['cache']['root']
        cfg['ranking']['merge_group_labels']=[48,49]
        cfg['cache']['root']=str(root/'data/rl_candidate_rank_v6_new50_5/cache')
        cfg['train']['output_dir']=str(root/'work_dir/active_view_candidate_rank_v6_new50_5')
        cfg['train']['epochs']=50
        cfg['fast'].update(batch_size=8192,updates_per_epoch=64)
        cfg['state']['candidate_dim']=7
        seen=cfg['evaluation']['seen_class_ids']
        hold=sorted(np.random.default_rng(20260909).choice(seen,10,replace=False).tolist())
        cfg['v6']={'policy_holdout_classes':hold,'policy_train_classes':sorted(set(seen)-set(hold)),
                   'seeds':[20260909,20260910,20260911],'learning_rate':.0003,
                   'causal_inputs_only':True,'selection':'0.75 proxy5way_macro_gain + 0.25 seen50way_macro_gain',
                   'priority_mixture':.5,'task_full_fraction':.25,'two_view_weight':.1,
                   'holdout_note':'Policy-held-out seen classes, not recognizer-unseen classes.'}
        cfg['_config_path']=str(config_path)
        # v5 ranking hyperparameters are not used by this independent trainer.
        cfg['ranking']['priority_mixture']=.5
        cfg['ranking']['rank_loss_weight']=None
        cfg['ranking']['utility']='correct-candidate set; no confidence-only unique winner'
        config_path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    cfg=yaml.safe_load(config_path.read_text())
    out=Path(cfg['train']['output_dir']);out.mkdir(exist_ok=True)
    stop=threading.Event();thread=threading.Thread(target=telemetry,args=(logs/'gpu_telemetry.jsonl',stop),daemon=True);thread.start()
    try:
        for split in ('dqn_train','val','seen_test','test'):
            print('V6_CACHE',split,flush=True)
            subprocess.run([sys.executable,'-u','-m','rl.build_offline_cache','--config',str(config_path),'--split',split],cwd=root,check=True)
            coverage=json.loads((Path(cfg['cache']['root'])/split/'coverage.json').read_text())
            assert not coverage['missing_classes'],coverage
        print('V6_ALL_CACHE_COVERAGE_PASS',flush=True)
        # Archive the implementation used by this run for reproducibility.
        for name in ('causal_rank_v6.py','run_causal_v6.py'):
            (out/name).write_bytes((root/'rl'/name).read_bytes())
        subprocess.run([sys.executable,'-u','-m','rl.causal_rank_v6','--config',str(config_path)],cwd=root,check=True)
        print('V6_PIPELINE_COMPLETE',flush=True)
    finally:
        stop.set();thread.join(timeout=5)


if __name__=='__main__':main()
