"""Sequential Full-DQN control with a non-adjacent target-pair constraint.

This control keeps the Full DQN input and training protocol unchanged, but
restricts the two final target views to pairs whose linear rank distance is at
least two: (0, 2), (0, 3), and (1, 3).  Thus (3, 0) is allowed, while the
adjacent pairs (0, 1), (1, 2), and (2, 3) are masked.  The current robot
positions remain legal starts and may be selected at zero movement cost.
"""

from __future__ import annotations

import torch

from . import sequential as seq


VARIANT = "full_depth19_nonadjacent"
_original_build_state = seq.b.build_state


def _constrain_state(raw, episodes, start_a, start_b, selected, variant):
    state = _original_build_state(
        raw, episodes, start_a, start_b, selected, variant
    )
    if variant != VARIANT:
        return state

    # The ordinary state mask allows every feasible second partner.  Replace
    # it with the non-adjacent partner set.  At stage zero all four slots
    # remain possible, but only if at least one non-adjacent partner is
    # feasible.
    _, pair_feasible = seq.b.assignment_matrix(
        raw, episodes, start_a, start_b
    )
    rows = torch.arange(len(episodes), device=raw.device)
    ranks = torch.arange(seq.b.NUM_VIEWS, device=raw.device)
    nonadjacent = (ranks[:, None] - ranks[None, :]).abs() >= 2
    present = selected >= 0
    allowed = torch.zeros_like(state["mask"], dtype=torch.bool)

    stage0 = ~present
    if stage0.any():
        stage0_rows = rows[stage0]
        stage0_pairs = pair_feasible[stage0_rows]
        allowed[stage0] = (stage0_pairs & nonadjacent[None]).any(dim=-1)

    stage1 = present
    if stage1.any():
        selected_stage1 = selected[stage1]
        stage1_rows = rows[stage1]
        stage1_pairs = pair_feasible[stage1_rows] & nonadjacent[None]
        allowed[stage1] = stage1_pairs[
            torch.arange(len(stage1_rows), device=raw.device), selected_stage1
        ]

    state["mask"] = state["mask"] & allowed
    return state


def _filter_transition_pool(pool):
    """Keep only transitions whose terminal pair has rank gap at least two."""

    stage1 = pool.stage1
    terminal_pair = (pool.selected - pool.action).abs() >= 2

    # A stage-1 transition is retained only if its selected first action has
    # a feasible opposite partner for the same two robot starts.
    _, pair_feasible = seq.b.assignment_matrix(
        seq_pool_raw,
        pool.episode,
        pool.start_a,
        pool.start_b,
    )
    ranks = torch.arange(seq.b.NUM_VIEWS, device=pool.action.device)
    nonadjacent = (ranks[:, None] - ranks[None, :]).abs() >= 2
    stage1_rows = torch.arange(pool.size, device=pool.action.device)[stage1]
    stage1_pairs = pair_feasible[stage1_rows] & nonadjacent[None]
    stage1_keep = stage1_pairs[
        torch.arange(len(stage1_rows), device=pool.action.device), pool.action[stage1]
    ].any(dim=-1)
    keep = torch.zeros_like(stage1)
    keep[stage1] = stage1_keep
    keep[~stage1] = terminal_pair[~stage1]

    fields = (
        "episode",
        "start_a",
        "start_b",
        "selected",
        "action",
        "reward",
        "done",
        "stage1",
        "next_episode",
        "next_a",
        "next_b",
        "next_selected",
    )
    for name in fields:
        if hasattr(pool, name):
            setattr(pool, name, getattr(pool, name)[keep])
    return pool


seq_pool_raw = None


def pool(raw, fused, penalty, device):
    global seq_pool_raw
    seq_pool_raw = raw
    result = seq.pool(raw, fused, penalty, device)
    return _filter_transition_pool(result)


def report(root, metadata, summary):
    metadata["pair_constraint"] = (
        "only target pairs with linear rank distance >= 2: "
        "(0,2), (0,3), and (1,3); adjacent pairs are masked"
    )
    metadata["pair_constraint_training"] = (
        "the transition pool and both DQN decision-stage masks use the same constraint"
    )
    seq.report(root, metadata, summary)


if __name__ == "__main__":
    seq.b.load_raw = seq.load
    seq.b.build_state = _constrain_state
    seq.b.build_transition_pool = pool
    seq.b.evaluate_policy_once = seq.validation
    seq.b.test_seed_evaluation = seq.evaluation
    seq.b.write_results_markdown = report
    seq.b.main()
