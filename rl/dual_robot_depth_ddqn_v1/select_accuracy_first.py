"""Record the validation-only ranking and verify restored winning checkpoint."""
import json
from pathlib import Path


def main():
    source = Path('work_dir/strict45_5_sequential_reward_search')
    output = Path('work_dir/strict45_5_sequential_accuracy_first')
    rows = []
    for path in source.glob('*/models/*/*/train.log'):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record['validation'].get('episodes', 0):
                rows.append({'configuration':path.parents[3].name,'epoch':record['epoch'],
                             'validation':record['validation']})
    rows.sort(key=lambda r:(-round(r['validation']['accuracy'],6),r['validation']['mean_movement_cost']))
    chosen = json.loads((output/'selection_full_depth19.json').read_text())['chosen']
    logs = [json.loads(line) for line in (output/'models/full_depth19/seed_20260909/train.log').read_text().splitlines()]
    restored = next(row for row in logs if row['epoch']==chosen['best_epoch'])
    assert rows[0]['configuration']=='ce8_move0.1_sep0'
    assert chosen['best_epoch']==rows[0]['epoch']
    assert abs(restored['validation']['accuracy']-rows[0]['validation']['accuracy']) < 1e-7
    (output/'pseudo_accuracy_ranking.json').write_text(json.dumps(rows,indent=2))
    (output/'selected_reward.json').write_text(json.dumps({'ce':8.,'movement':.1,'separation':0.,
        'selected':chosen,'validation':restored['validation'],
        'selection_rule':'pseudo accuracy first; movement only breaks accuracy ties; no movement feasibility cutoff'},indent=2))
    print('Verified restored winner:',chosen['best_epoch'],restored['validation']['accuracy'])


if __name__=='__main__':
    main()
