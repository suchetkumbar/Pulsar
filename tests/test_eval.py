import numpy as np
import pandas as pd
import pytest

from pulsar.eval.bootstrap import cluster_bootstrap, paired_bootstrap_diff
from pulsar.eval.metrics import bland_altman, fairness_summary, hr_metrics, ita_slope, subject_level


def _table(rng, n_subj=12, n_win=20, err_by_group=None):
    err_by_group = err_by_group or {"A": 2.0, "B": 6.0}
    rows = []
    for g, e in err_by_group.items():
        for s in range(n_subj):
            ita = 40 if g == "A" else -10
            for _ in range(n_win):
                true = rng.uniform(60, 100)
                rows.append(
                    {
                        "subject": f"{g}{s}",
                        "group": g,
                        "ita": ita + rng.normal(0, 3),
                        "hr_true": true,
                        "hr_pred": true + rng.normal(0, e),
                    }
                )
    return pd.DataFrame(rows)


def test_hr_metrics_basic():
    m = hr_metrics(np.array([70, 80, 90.0]), np.array([72, 80, 87.0]))
    assert m["mae"] == pytest.approx(5 / 3)
    assert m["acc5"] == 1.0 and m["n"] == 3


def test_subject_level_weights_subjects_equally():
    df = pd.DataFrame({"subject": ["a"] * 9 + ["b"], "hr_true": [70] * 10, "hr_pred": [71] * 9 + [80]})
    assert subject_level(df)["mae"].iloc[0] == pytest.approx((1 + 10) / 2)


def test_fairness_and_slope(rng):
    df = _table(rng)
    fs = fairness_summary(df, "group")
    assert fs["worst_group"] == "B" and fs["gap"] > 2.0
    assert ita_slope(df) < 0  # error grows as ITA decreases


def test_bootstrap_ci_contains_point_and_narrows(rng):
    df = _table(rng)

    def stat(d):
        return subject_level(d)["mae"].iloc[0]

    ci = cluster_bootstrap(df, stat, group_col="group", n_boot=200)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    small = cluster_bootstrap(_table(rng, n_subj=4), stat, group_col="group", n_boot=200)
    assert (small["hi"] - small["lo"]) > (ci["hi"] - ci["lo"])


def test_paired_bootstrap_detects_improvement(rng):
    a = _table(rng, err_by_group={"A": 2.0, "B": 3.0})
    b = a.copy()
    b["hr_pred"] = b["hr_true"] + (a["hr_pred"] - a["hr_true"]) * 2.0  # B is twice as bad

    def worst(d):
        return subject_level(d, "group")["mae"].max()

    d = paired_bootstrap_diff(a, b, worst, group_col="group", n_boot=200)
    assert d["hi"] < 0  # A better than B, CI excludes 0


def test_bland_altman():
    ba = bland_altman(np.array([71, 72, 73.0]), np.array([70, 70, 70.0]))
    assert ba["bias"] == pytest.approx(2.0) and ba["loa_lo"] < 2 < ba["loa_hi"]
