"""Strict 45/5/5 evaluation with persistent robot positions across recordings."""
import json
from . import experiment as b
from . import sequential as s


def load(*args):
    raw, audit = s.original_load(*args)
    meta = json.loads((args[0] / args[2] / 'metadata.json').read_text())
    raw.camera_ids = []
    raw.subjects = []
    for ep in meta['episodes']:
        cameras = [''] * 4
        for view in ep['views']:
            cameras[int(view['angle_rank'])] = view['camera']
        raw.camera_ids.append(cameras)
        raw.subjects.append(ep['subject'])
    assert len(raw.camera_ids) == raw.num_episodes
    for i in b.eligible_episodes(raw, range(55)).nonzero().flatten().tolist():
        assert set(raw.camera_ids[i]) == {'C001', 'C003', 'C005', 'C007'}
    audit['sequential_camera_mapping'] = 'metadata angle_rank to physical camera; verified all complete episodes'
    return raw, audit


def replay(raw, fused, model, variant, seed, bank, penalty=0):
    return s.replay(raw, fused, model, variant, seed, bank, penalty)


def validation(model, raw, fused, episodes, a, c, variant, bank, penalty):
    result = replay(raw, fused, model, variant, 731, bank, penalty)
    reference = replay(raw, fused, None, 'random_pair', 731, bank, penalty)
    ratio = result['mean_movement_cost'] / max(reference['mean_movement_cost'], 1e-8)
    result['movement_ratio_to_random'] = ratio
    result['physical_reward'] = result['mean_reward']
    result['mean_reward'] = result['accuracy'] if ratio <= 0.4 else -ratio
    return result


def report(root, metadata, summary):
    metadata.update(
        robot_protocol='persistent physical camera identities; shuffle within subject; initialize only at subject boundary',
        training_transition='selected robot positions map by camera identity into next sample; bootstrap across samples',
        checkpoint_selection='pseudo-unseen accuracy subject to move cost <= 0.4 * random; infeasible checkpoints ranked by movement ratio',
        sequence_limit='synthetic subject-wise ordering, not recorded chronological robot navigation',
    )
    b.dump_json(root / 'config_resolved.json', metadata)
    lines = ['# Strict 45/5/5 sequential results', '',
             f"Final unseen classes: {b.FINAL_UNSEEN}. Pseudo-unseen selection classes: {b.PSEUDO_UNSEEN}.",
             f"{len(metadata['eval_seeds'])} evaluation seeds; CI describes variation in stream order and initial positions, not independent training runs.",
             'Movement is the summed two-robot normalized angular proxy, not meters. Current positions remain selectable.', '',
             '|Model|Accuracy (%)|95% CI (%)|Move cost|Move 95% CI|Ratio to random|',
             '|---|---:|---:|---:|---:|---:|']
    ref = summary['random_pair']['mean_movement_cost']['mean']
    for name, values in summary.items():
        a = values['accuracy']; m = values['mean_movement_cost']
        lines.append(f"|{name}|{100*a['mean']:.2f}|[{100*a['ci95_low']:.2f}, {100*a['ci95_high']:.2f}]|{m['mean']:.4f}|[{m['ci95_low']:.4f}, {m['ci95_high']:.4f}]|{m['mean']/ref:.3f}|")
    lines += ['', 'Only subject boundaries reset positions. Camera identity, rather than body-relative rank, is carried between samples.',
              'Training uses 45 seen classes. Checkpoint selection uses the five pseudo-unseen classes only.',
              'Reward coefficients match the preceding independent-start comparison (margin minus 0.25 times movement).',
              'No reward coefficient sweep is performed in this comparison.']
    (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    b.load_raw = load
    b.build_transition_pool = s.pool
    b.evaluate_policy_once = validation
    b.test_seed_evaluation = s.evaluation
    b.write_results_markdown = report
    b.main()
