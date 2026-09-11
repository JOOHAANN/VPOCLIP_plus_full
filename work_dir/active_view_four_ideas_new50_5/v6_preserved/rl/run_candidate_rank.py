"""Launch isolated variable-view ranking caches, training and both start protocols."""
import json
from pathlib import Path
import subprocess
import sys
import yaml


def main():
    root=Path(__file__).resolve().parents[1]
    path=root/'rl/handoff_configs/new50_5_candidate_rank.yaml'
    if not path.exists():
        cfg=yaml.safe_load((root/'rl/handoff_configs/new50_5_offline_nocost.yaml').read_text())
        old=cfg['cache']['root']
        cfg['cache']['root']=str(root/'data/rl_candidate_rank_new50_5/cache')
        cfg['train']['output_dir']=str(root/'work_dir/active_view_candidate_rank_new50_5')
        cfg['ranking']={'variable_views':True,'merge_group_labels':[48],'reuse_cache':old,
                        'two_view_loss_weight':.1,'priority_mixture':.5,'rank_loss_weight':2.,
                        'utility':'2*correct + margin/(20+abs(margin))',
                        'near_tie_utility_threshold':.03,'initial_protocols':['fixed0','random']}
        cfg['_config_path']=str(path)
        path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    cfg=yaml.safe_load(path.read_text())
    for split in ('dqn_train','val','seen_test','test'):
        subprocess.run([sys.executable,'-u','-m','rl.build_offline_cache','--config',str(path),'--split',split],cwd=root,check=True)
    subprocess.run([sys.executable,'-u','-m','rl.train_candidate_rank','--config',str(path)],cwd=root,check=True)
    print('CANDIDATE_RANK_PIPELINE_COMPLETE',flush=True)


if __name__=='__main__':main()
