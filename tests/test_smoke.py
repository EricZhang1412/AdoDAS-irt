"""Minimal unit tests — no GPU, no real features. Run with pytest."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_dass21_grouping():
    from adodas.utils.dass21 import DASS21_GROUP_ITEMS, DASS21_GROUP_CUTOFFS, item_to_group_vector

    # Each subscale has exactly 7 items, jointly cover all 21.
    assert all(len(v) == 7 for v in DASS21_GROUP_ITEMS.values())
    flat = sum(DASS21_GROUP_ITEMS.values(), [])
    assert set(flat) == set(range(21))

    # Standard DASS-21 cutoffs.
    assert DASS21_GROUP_CUTOFFS["D"] == 10.0
    assert DASS21_GROUP_CUTOFFS["A"] == 8.0
    assert DASS21_GROUP_CUTOFFS["S"] == 15.0

    g = item_to_group_vector()
    assert len(g) == 21 and all(0 <= x < 3 for x in g)


def test_sigmoid_logit_roundtrip():
    from adodas.models.heads import logit, sigmoid

    p = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    back = sigmoid(logit(p))
    assert np.allclose(back, p, atol=1e-4)


def test_coral_monotonic_decoding():
    from adodas.models.heads import argmax_from_cumlogits, monotonic_class_probs

    rng = np.random.default_rng(0)
    z = rng.standard_normal((50, 21, 3)).astype(np.float32)
    probs = monotonic_class_probs(z)
    # Simplex constraint.
    assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-4)
    # Decoded class in [0, 3].
    cls = argmax_from_cumlogits(z)
    assert cls.shape == (50, 21)
    assert cls.min() >= 0 and cls.max() <= 3


def test_soft_dass21_all_high():
    """All-1 z-logits should drive every subscale above its cutoff."""
    from adodas.models.heads import soft_dass21_from_expected, expected_score_from_cumlogits

    z = np.full((4, 21, 3), 10.0, dtype=np.float32)  # very confident "score >= 3"
    e_y = expected_score_from_cumlogits(z)
    soft = soft_dass21_from_expected(e_y, temperature=1.0)
    assert (soft > 0.95).all()


def test_threshold_to_int_monotone():
    from adodas.stack.thresholds import threshold_to_int

    score = np.array([0.1, 1.0, 2.0, 2.9, 3.0])
    t = np.array([0.5, 1.5, 2.5])
    out = threshold_to_int(score, t)
    assert out.tolist() == [0, 1, 2, 3, 3]


def test_threshold_optimisation_recovers_signal():
    """If continuous scores are noisy label + noise, threshold opt should
    recover near-optimal thresholds and beat naive 0.5/1.5/2.5."""
    from adodas.stack.thresholds import apply_thresholds, optimise_per_item
    from adodas.utils.metrics import quadratic_weighted_kappa

    rng = np.random.default_rng(42)
    n = 400
    y = rng.integers(0, 4, size=(n,))
    # Shift the score distribution so naive thresholds are sub-optimal.
    score = y.astype(np.float32) * 0.7 + 0.4 + rng.normal(0, 0.2, size=n).astype(np.float32)
    t, qwk = optimise_per_item(score, y, init=(0.5, 1.5, 2.5))
    assert qwk > 0.5
    # Should beat naive thresholds.
    naive = apply_thresholds(score[:, None], np.array([[0.5, 1.5, 2.5]], dtype=np.float32))[:, 0]
    qwk_naive = quadratic_weighted_kappa(y, naive)
    assert qwk >= qwk_naive - 1e-6


def test_nnls_blend_simple():
    from adodas.stack.meta_blend import blend_a2_nnls

    rng = np.random.default_rng(0)
    n, e = 50, 3
    y = rng.uniform(0, 3, size=(n, 21)).astype(np.float32)
    # Two informative experts + one noise expert.
    oof = np.zeros((n, 21, e), dtype=np.float32)
    oof[..., 0] = y + rng.normal(0, 0.1, size=y.shape).astype(np.float32)
    oof[..., 1] = y + rng.normal(0, 0.3, size=y.shape).astype(np.float32)
    oof[..., 2] = rng.normal(0, 1.0, size=y.shape).astype(np.float32)
    res = blend_a2_nnls(oof, y)
    # Expert 0 should dominate.
    assert res.weights[:, 0].mean() > res.weights[:, 2].mean()


def test_fold_disjoint():
    from adodas.data.folds import assign_folds, verify_disjoint
    from adodas.data.labels import ID_COLS
    from adodas.utils.dass21 import ITEM_COLS

    rng = np.random.default_rng(0)
    n = 50
    labels = pd.DataFrame({
        "anon_school": [f"s{i // 10:02d}" for i in range(n)],
        "anon_class": [f"c{(i // 5) % 4:02d}" for i in range(n)],
        "anon_pid": [f"p{i:04d}" for i in range(n)],
    })
    for c in ITEM_COLS:
        labels[c] = rng.integers(0, 4, size=n)
    folds = assign_folds(labels, n_folds=5, seed=42)
    verify_disjoint(labels, folds)
    assert len(folds) == n
    assert set(folds.tolist()) == {0, 1, 2, 3, 4}


def test_submission_writer_shape():
    from adodas.data.labels import ID_COLS
    from adodas.submit.build_submission import write_a1_submission, write_a2_submission
    from adodas.utils.dass21 import A1_COLS, ITEM_COLS
    import tempfile

    n = 8
    ids = pd.DataFrame({
        "anon_school": [f"s{i:02d}" for i in range(n)],
        "anon_class": [f"c{i % 3:02d}" for i in range(n)],
        "anon_pid": [f"p{i:04d}" for i in range(n)],
    })
    a1 = np.full((n, 3), 0.5, dtype=np.float32)
    a2 = np.zeros((n, 21), dtype=np.int32)

    with tempfile.TemporaryDirectory() as td:
        p1 = write_a1_submission(ids, a1, td)
        p2 = write_a2_submission(ids, a2, td)
        df1 = pd.read_csv(p1)
        df2 = pd.read_csv(p2)
    expected_a1 = ID_COLS + ["p_D", "p_A", "p_S"]
    expected_a2 = ID_COLS + ITEM_COLS
    assert list(df1.columns) == expected_a1
    assert list(df2.columns) == expected_a2
    assert (df2[ITEM_COLS].values == 0).all()
    assert (df1[["p_D", "p_A", "p_S"]].values == 0.5).all()


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
