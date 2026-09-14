"""Align per-episode target columns after excluding unavailable episodes."""
import json
import numpy as np
from .build_ranked_cache import DEFAULT_OUTPUT_CACHE, DEFAULT_SOURCE_CACHE, SPLITS

for split in SPLITS:
    dest = DEFAULT_OUTPUT_CACHE / split
    meta = json.loads((dest / 'metadata.json').read_text())
    source = json.loads((DEFAULT_SOURCE_CACHE / split / 'metadata.json').read_text())
    lookup = {e['base_sample']: i for i, e in enumerate(source['episodes'])}
    assert len(lookup) == len(source['episodes'])
    indices = [lookup[e['base_sample']] for e in meta['episodes']]
    for name in ('labels', 'target_columns'):
        values = np.load(DEFAULT_SOURCE_CACHE / split / (name + '.npy'))[indices]
        np.save(dest / (name + '.npy'), values)
    print(split, len(indices), flush=True)
    reachable = np.load(dest / 'reachable.npy')
    for j in range(4):
        reachable[:, j, j] = True
    np.save(dest / 'reachable.npy', reachable)
