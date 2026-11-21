import numpy as np
import pandas as pd
import pytest

from pulsar.eval import selective


def test_risk_coverage_and_aurc_reward_good_ranking(rng):
    err = rng.exponential(3.0, 500)
    good = -err + rng.normal(0, 0.1, 500)  # confidence ranks errors well
    bad = rng.normal(size=500)  # uninformative
    cov, risk = selective.risk_coverage(err, good)
    assert cov[-1] == 1.0 and risk[-1] == pytest.approx(err.mean())
    assert selective.aurc(err, good) < selective.aurc(err, bad)


def test_threshold_for_target_risk(rng):
    err = rng.exponential(3.0, 1000)
    conf = -err
    tau = selective.threshold_for_target_risk(err, conf, 2.0)
    assert np.abs(err[conf >= tau]).mean() <= 2.0 + 1e-9


def test_coverage_parity_detects_disparate_abstention(rng):
    rows = []
    for g, conf_shift in (("light", 0.2), ("dark", -0.2)):
        for s in range(10):
            for _ in range(30):
                rows.append(
                    {
                        "group": g,
                        "subject": f"{g}{s}",
                        "abs_err": rng.exponential(3),
                        "conf": np.clip(0.6 + conf_shift + rng.normal(0, 0.1), 0, 1),
                    }
                )
    df = pd.DataFrame(rows)
    tab = selective.selective_by_group(df, "group", tau=0.6)
    assert tab.loc["light", "coverage"] > tab.loc["dark", "coverage"]
    assert selective.coverage_parity_ratio(tab) < 0.5


def test_ece_perfect_and_overconfident(rng):
    conf = rng.uniform(0, 1, 20000)
    correct = rng.uniform(0, 1, 20000) < conf
    assert selective.ece(conf, correct) < 0.02
    assert selective.ece(np.full(1000, 0.95), np.zeros(1000)) == pytest.approx(0.95)


def test_fit_temperature_recovers_scale():
    bpm = np.arange(40, 201, 1.0)
    rng = np.random.default_rng(0)
    hr = rng.uniform(60, 120, 400)
    pred = hr + rng.normal(0, 4.0, 400)  # true spread 4 bpm
    # logits of a model that is overconfident by x3 (sharpness of 4/3 bpm instead of 4)
    logits = -((bpm[None] - pred[:, None]) ** 2) / (2 * (4.0 / 3) ** 2)
    t = selective.fit_temperature(logits, hr, bpm)
    assert t > 3.0  # needs strong softening
    _, c1 = selective.confidence_from_logits(logits, bpm)
    _, ct = selective.confidence_from_logits(logits, bpm, t)
    acc = np.abs(pred - hr) <= 5
    assert selective.ece(ct, acc) < selective.ece(c1, acc)
