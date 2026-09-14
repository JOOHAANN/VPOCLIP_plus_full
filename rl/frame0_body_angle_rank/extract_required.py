"""Extract indexed skeleton CSVs in full, validating each member's CRC."""
import csv
import json
import shutil
import zipfile
from pathlib import Path

root = Path('/home/youhan/ws')
archive = root / 'ETRI-Activity3D-Skeleton-archives/ETRI-Activity3D_CSV.zip'
with (root / 'ETRI-Activity3D-RGB/low_view_camera_angles_split55.csv').open() as f:
    rows = list(csv.DictReader(f))
with zipfile.ZipFile(archive) as z:
    members = {Path(i.filename).name: i for i in z.infolist() if not i.is_dir()}
    targets = {r['sample_id'] + '.csv': Path(r['skeleton_path']) for r in rows}
    missing = sorted(set(targets) - set(members))
    if missing:
        raise RuntimeError(f'Missing {len(missing)} CSVs: {missing[:10]}')
    needed = sum(members[n].file_size for n in targets)
    free = shutil.disk_usage(root).free
    print(json.dumps(dict(files=len(targets), extracted_bytes=needed, free_bytes=free)), flush=True)
    if needed + 8 * 1024**3 > free:
        raise RuntimeError('Insufficient space including 8 GiB cache reserve')
    for k, (name, target) in enumerate(targets.items(), 1):
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.csv.extracting')
        with z.open(members[name]) as src, temporary.open('wb') as dst:
            shutil.copyfileobj(src, dst)
        temporary.replace(target)
        if k % 1000 == 0:
            print(f'EXTRACTED {k}/{len(targets)}', flush=True)
print('EXTRACTION_COMPLETE_CRC_CHECKED', flush=True)
