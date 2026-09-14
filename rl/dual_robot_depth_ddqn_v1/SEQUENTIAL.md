# Persistent-position sequential experiment

Entry point: `python -m rl.dual_robot_depth_ddqn_v1.sequential`.
Output: `work_dir/dual_robot_sequential_v2_lambda025`.

Full, Human-only, and Object-only use temporal Conv1d inputs, respectively
human 13x6 and object 13x19. Human orientation and object relative inverse
depth are first-frame values repeated across the window in the source cache.

Each subject is a separate synthetic stream. Recordings are shuffled with a
seed, without using action labels for ordering. Robot positions persist by
camera identity and are remapped through rank_to_original for each recording.
Current positions are legal targets, with zero self-movement cost. The two
final target positions must be distinct. Minimum-cost feasible assignment
determines which physical robot reaches which target.

Training enumerates counterfactual transitions for every start pair. After
pair selection the Bellman target continues to the next recording at the
assigned positions; streams terminate at subject boundaries. Replay minibatch
order need not follow stream order. Evaluation executes each method's own
position history over shared seeded streams. No test-time parameter updates.

The initial run uses movement penalty 0.25 and three training seeds. Checkpoint
selection maximizes validation accuracy among checkpoints whose mean movement
is at most half the validation random baseline. If none satisfies the budget,
the lowest ratio wins and must be reported as infeasible. Final evaluation has
30 stream/start seeds, not 30 independently trained networks.

Movement is the cache's normalized angular proxy; measured meter distances
and real cross-action temporal continuity are unavailable. This experiment
must be described as synthetic sequential replay, not real continuous robot
navigation. Real-world camera calibration is still required to validate
physical travel distance. Accuracy exceeding random/cyclic is an empirical
objective and is not guaranteed by the movement constraint.
