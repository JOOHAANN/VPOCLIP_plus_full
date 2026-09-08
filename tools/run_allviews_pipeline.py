"""All-camera retraining with canonical subjects, isolated outputs and stage logs.

--prepare inventories data and writes configs; default also runs the pipeline.
"""
import argparse
import collections
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml

WS = Path('/home/youhan/ws')
VPO = WS / 'VPOCLIP_plus_full'
X3D = WS / 'X3D_full'
CTR = WS / 'CTR-GCN_17_full'
YOLO = WS / 'yolov5_full'
RGB = WS / 'ETRI-Activity3D-RGB'
MANIFEST = WS / 'ETRI_subject_split_45_25_10_20.json'
RUN = VPO / 'logs/allviews_20260907'
CACHE = X3D / 'data/etri_rgb_allviews_cs_45_25_10_20'
POSE = CTR / 'data/etri_coco17/allviews_cs45'
NPZ = POSE / 'ETRI_55_CS_rtmpose_coco17_13.npz'
XOUT = X3D / 'outputs/etri_allviews_cs45_zsl50_5'
COUT = CTR / 'work_dir/etri_coco17/allviews_cs45_zsl50_5'
FEATURES = VPO / 'data/allviews_cs45_features'
ZDATA = VPO / 'data/allviews_cs45_zsl50_5'
CDATA = VPO / 'data/allviews_cs45_closed55'
PY = sys.executable
UNSEEN = [9, 10, 11, 17, 49]
OUTPUT_PREFIX = 'allviews_'
RUN_NAME = 'allviews_20260907'
ZSL_ONLY = False
REUSE_OBJECTS = False
OBJECTS = FEATURES / 'object'
CLASS_SPLIT_PATH = None
X3D_MAX_ITER = None


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def stage(name, repo, argv):
    marker = RUN / (name + '.done')
    if marker.exists():
        print('SKIP completed', name, flush=True)
        return
    command = [PY, *map(str, argv)]
    print('START', name, command, flush=True)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(repo), PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='0',
               PATH=str(Path(PY).parent) + ':' + env['PATH'])
    with (RUN / (name + '.log')).open('a') as log:
        result = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{name} failed ({result.returncode}); see {RUN / (name + ".log")}')
    marker.touch()
    print('DONE', name, flush=True)


def config(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(content, sort_keys=False))


def prepare():
    RUN.mkdir(parents=True, exist_ok=True)
    groups = json.loads(MANIFEST.read_text())['subject_groups']
    sets = [set(groups[k]) for k in ('vpo_train','dqn_train','val','test')]
    assert [len(s) for s in sets] == [45,25,10,20]
    assert len(set.union(*sets)) == sum(map(len,sets))
    owner = {p:s for s, people in groups.items() for p in people}
    counts = {s:collections.Counter() for s in groups}
    seen_train = 0
    for p in RGB.rglob('*.mp4'):
        m = re.fullmatch(r'A(\d{3})_(P\d{3})_G\d{3}_(C\d{3})\.mp4',p.name)
        if not m: continue
        action, person, camera = m.groups()
        assert 1 <= int(action) <= 55 and person in owner
        counts[owner[person]][camera] += 1
        if owner[person]=='vpo_train' and int(action)-1 not in UNSEEN:
            seen_train += 1
    for c in counts.values():
        assert set(c) == {f'C{i:03}' for i in range(1,9)}, c
    supervised = sum(sum(counts[s].values()) for s in ('vpo_train','val','test'))
    need_gib = supervised * (3*13*160*160*2 + 2*(13*192*6*6*2+2*64*13*17*4)) / 2**30 + 35
    available = shutil.disk_usage(WS).free/2**30
    # Cache allocation is only checked before the first launch, since files are resumable.
    if not (CACHE/'metadata.json').exists() and available < need_gib:
        raise RuntimeError(f'Need {need_gib:.1f} GiB, available {available:.1f} GiB')
    inventory = dict(camera_counts=counts,seen_train=seen_train,required_GiB=need_gib,
                     available_GiB=available,subject_manifest=str(MANIFEST))
    save(RUN/'inventory.json',inventory)
    print(json.dumps(inventory,indent=2),flush=True)
    xc = yaml.safe_load((X3D/'configs/x3d-s_etri_rgb_cs_45_25_10_20_zsl50_5_182.yaml').read_text())
    xc['OUTPUT_DIR']=str(XOUT)
    for s in ('TRAIN','TEST'):
        xc['DATASETS'][s]['DATA_DIR']=str(CACHE)
        xc['DATASETS'][s]['ANNOTATION_DIR']=str(CACHE)
    config(RUN/'x3d.yaml',xc)
    cc = yaml.safe_load((CTR/'config/etri-coco17/ctrgcn_joint_coco17_13.yaml').read_text())
    cc['work_dir']=str(COUT)
    for s in ('train_feeder_args','test_feeder_args'): cc[s]['data_path']=str(NPZ)
    for s in ('train_feeder_args','test_feeder_args'): cc[s]['exclude_classes']=UNSEEN
    cc.update(batch_size=512,test_batch_size=512,num_worker=8,prefetch_factor=2,save_interval=1)
    config(RUN/'ctrgcn.yaml',cc)
    for name, data in [('final_aug',ZDATA),('final_aug_entropy_single_h2',ZDATA),
                       ('final55_aug',CDATA),('final55_aug_entropy_h5_full300_score',CDATA)]:
        closed=name.startswith('final55')
        if ZSL_ONLY and closed: continue
        work=VPO/'work_dir'/(OUTPUT_PREFIX+name)
        cfg={'base_config':str(VPO/f'config_{name}.yaml'),
             'runtime':{'device':'cuda:0'},
             'data':{'train':{'data_dir':str(data)},'val':{'data_dir':str(data)},
                     'text':{'xlsx':str(VPO/'ntu55_global_descriptions.xlsx'),
                             'cache_output':str(work/'text.pt')},
                     'dataloader':{'num_workers':12,'persistent_workers':True}},
             'outputs':{'work_dir':str(work),'auto_run_dir':True,'run_name':RUN_NAME}}
        # Resolved absolute paths prevent the generated config directory changing semantics.
        cfg['train']={'init_checkpoint':None}
        config(RUN/f'{name}.yaml',cfg)
    return seen_train


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--shared-only',action='store_true')
    parser.add_argument('--wait-extractor-pid',type=int)
    args=parser.parse_args()
    seen_train=prepare()
    if args.prepare: return
    if args.wait_extractor_pid:
        print('WAIT existing shared extractor', args.wait_extractor_pid, flush=True)
        proc = Path(f'/proc/{args.wait_extractor_pid}')
        while proc.exists():
            status = (proc/'stat').read_text().split(') ',1)[-1].split()[0]
            if status == 'Z': break
            time.sleep(15)
    stage('01_rgb_cache',X3D,['tools/build_etri_rgb_tensor.py','--rgb-root',RGB,
        '--output-dir',CACHE,'--camera','all','--subject-manifest',MANIFEST,
        '--seed','20260906','--workers','20'])
    for s in ('train','val','test'):
        valid=np.load(CACHE/f'{s}_valid.npy')
        if not valid.all():
            raise RuntimeError(f'{s}: {(~valid).sum()} failed videos. Review decode failures before alignment.')
    stage('02_rtmpose',CTR,['tools/extract_rtmpose_parallel.py','--rgb-root',RGB,
        '--split-dir',CACHE,'--output-dir',POSE])
    if args.shared_only:
        print('SHARED PREPROCESSING COMPLETE; old 50/5 and closed55 PAUSED by user', flush=True)
        return
    with np.load(NPZ,allow_pickle=True) as z:
        for s in ('train','val','test','dqn'):
            assert np.array_equal(z[f'{s}_sample_name'],np.load(CACHE/f'{s}_sample_names.npy'))
            assert np.isfinite(z[f'x_{s}']).all()
    # Retain approximately 120 passes through the enlarged seen-class training set.
    # Reserve memory for the pre-existing RL jobs; evaluate uses up to 2x batch.
    free_mb=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    batch=320 if free_mb>93000 else 256 if free_mb>80000 else 224 if free_mb>70000 else 160
    iterations=X3D_MAX_ITER if X3D_MAX_ITER is not None else math.ceil(seen_train/batch)*120
    save(RUN/'x3d_schedule.json',dict(batch=batch,iterations=iterations,
                                  effective_epochs=iterations*batch/seen_train,
                                  warmup_iterations=400,lr_steps=[int(iterations*.5),int(iterations*.83)],
                                  selection='seen-validation top1',unseen=UNSEEN))
    stage('03_x3d_train',X3D,['tools/train_etri_x3d.py','--config',RUN/'x3d.yaml',
        '--data-dir',CACHE,'--output-dir',XOUT,'--pretrained',X3D/'pretrained/x3d_s.pyth',
        '--exclude-classes',*UNSEEN,'--workers','8','--prefetch-factor','2',
        '--batch-size',batch,'--max-iter',iterations,'--warmup-iterations','400',
        '--lr-steps',int(iterations*.5),int(iterations*.83),'--save-step','1000','--eval-step','1000','--resume'])
    stage('04_ctrgcn_train',CTR,['main.py','--config',RUN/'ctrgcn.yaml'])
    log=(COUT/'log.txt').read_text()
    epochs=re.findall(r'Epoch number:\s*(\d+)',log)
    if not epochs: raise RuntimeError('CTR-GCN best validation epoch missing')
    weights=list(COUT.glob(f'runs-{epochs[-1]}-*.pt'))
    assert len(weights)==1
    save(RUN/'selected_backbones.json',dict(x3d=str(XOUT/'model_best.pth'),pose=str(weights[0]),yolo=str(YOLO/'best.pt')))
    for s in ('train','val','test'):
        stage(f'05_x3d_{s}',X3D,['tools/extract_x3d_features.py','--config',RUN/'x3d.yaml',
            '--checkpoint',XOUT/'model_best.pth','--data-dir',CACHE,'--split',s,
            '--output',FEATURES/'x3d'/f'{s}_video.npy','--batch-size','512','--num-workers','8',
            '--tensor-resize-size','182','--spatial-size','6','--save-sidecars'])
        if not REUSE_OBJECTS:
            stage(f'06_yolo_{s}',YOLO,['extract_frame7_objects.py','--data-dir',CACHE,
                '--weights',YOLO/'best.pt','--split',s,'--output',OBJECTS/f'{s}_object.npy',
                '--batch-size','256','--device','0'])
        stage(f'07_pose_{s}',VPO,['tools/extract_current_ctrgcn_features.py','--ctrgcn-root',CTR,
            '--config',RUN/'ctrgcn.yaml','--weights',weights[0],'--npz',NPZ,'--split',s,
            '--output-dir',FEATURES/'pose','--batch-size','1024'])
    if not REUSE_OBJECTS:
        stage('08_object_maps',VPO,['tools/build_object_rs_maps.py','--paths',
            *[OBJECTS/f'{s}_object.npy' for s in ('train','val','test')],
            '--backup-dir',OBJECTS/'backups','--overwrite'])
    stage('09_assemble',VPO,['tools/assemble_current_45_25_10_20.py','--cache-dir',CACHE,
        '--x3d-dir',FEATURES/'x3d','--pose-dir',FEATURES/'pose','--object-dir',OBJECTS,
        '--zsl-dir',ZDATA,'--closed-dir',CDATA,'--manifest',MANIFEST,
        '--unseen-classes',*UNSEEN,*(['--zsl-only'] if ZSL_ONLY else []),
        *(['--unseen-recordings-manifest',CLASS_SPLIT_PATH] if CLASS_SPLIT_PATH else [])])
    for name,parent in [('final_aug',None),('final_aug_entropy_single_h2','final_aug'),
                        ('final55_aug',None),('final55_aug_entropy_h5_full300_score','final55_aug')]:
        if ZSL_ONLY and name.startswith('final55'): continue
        argv=['train.py','--config',RUN/f'{name}.yaml']
        if parent:
            latest=json.loads((VPO/'work_dir'/(OUTPUT_PREFIX+parent)/'latest_run.json').read_text())
            argv+=['--init-checkpoint',latest['last_model']]
        stage('10_'+name,VPO,argv)
    print('ALL CAMERA TRAINING COMPLETE',flush=True)


if __name__=='__main__':
    main()
