"""Live stage progress and GPU telemetry; never changes training processes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time


def tail(path, size=12000):
    if not path.exists():
        return ''
    with path.open('rb') as f:
        f.seek(max(0, path.stat().st_size-size))
        return f.read().decode('utf-8', errors='replace')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    while True:
        stamp = datetime.now(timezone.utc).isoformat()
        log = tail(args.run_dir/'pipeline.log')
        stages = re.findall(r'^START (\S+)', log, flags=re.M)
        stage = stages[-1] if stages else 'waiting'
        try:
            gpu = subprocess.check_output(['nvidia-smi',
                '--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw',
                '--format=csv,noheader'], text=True, timeout=5).strip()
        except (OSError, subprocess.SubprocessError) as exc:
            gpu = str(exc)
        record = dict(time=stamp, stage=stage, gpu=gpu)
        with (args.run_dir/'gpu_telemetry.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        print('\033[2J\033[H', end='')
        print(stamp, '\nRun:', args.run_dir, '\nStage:', stage)
        print('GPU: index, utilization, memory used/total, power\n'+gpu)
        print('\nPipeline:\n'+'\n'.join(log.splitlines()[-3:]))
        if stage != 'waiting':
            lines = tail(args.run_dir/(stage+'.log')).replace('\r','\n').splitlines()
            print('\nStage progress:\n'+'\n'.join(lines[-5:]))
        print('', flush=True)
        if args.once:
            return
        time.sleep(15)


if __name__ == '__main__':
    main()
