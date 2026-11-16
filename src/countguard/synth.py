"""Synthetic kidney-transplant-like cohort.

Shape follows the preprint's 10-variable cube (race 6 x sex 2 x 7 binary
immunosuppressant classes x graft rejection = 3,072 atomic cells). Every value
is simulated: medications come from a small latent-regimen model and
rejection from a logistic model. Nothing here is derived from patient data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MISSING = "cumulus__none"  # Cumulus' marker for a missing value

RACES = [
    "White",
    "Black or African American",
    "Asian",
    "American Indian or Alaska Native",
    "Native Hawaiian or Other Pacific Islander",
    MISSING,
]
RACE_P = [0.56, 0.19, 0.07, 0.02, 0.01, 0.15]
SEXES = ["female", "male"]
SEX_P = [0.42, 0.58]

MEDS = ["cni", "antimetabolite", "steroid", "mtor", "belatacept", "atg", "il2ra"]
MED_LABELS = {
    "cni": "calcineurin inhibitor",
    "antimetabolite": "antimetabolite",
    "steroid": "corticosteroid",
    "mtor": "mTOR inhibitor",
    "belatacept": "costimulation blocker",
    "atg": "lymphocyte-depleting induction",
    "il2ra": "IL-2 receptor antagonist",
}
YESNO = ["no", "yes"]

KIDNEY_VARS = ["race", "sex", *MEDS, "rejection"]
SUBCUBE_8 = ["race", "sex", "cni", "antimetabolite", "steroid", "mtor", "atg", "rejection"]
SUBCUBE_6 = ["race", "sex", "cni", "antimetabolite", "steroid", "rejection"]

LEVELS = {"race": RACES, "sex": SEXES, "rejection": YESNO, **{m: YESNO for m in MEDS}}

# Maintenance regimens: (probability, set of maintenance classes)
_REGIMENS = [
    (0.80, {"cni", "antimetabolite", "steroid"}),  # standard triple therapy
    (0.08, {"cni", "antimetabolite"}),  # steroid-free
    (0.035, {"mtor", "antimetabolite", "steroid"}),  # mTOR conversion
    (0.03, {"cni", "mtor", "steroid"}),  # CNI + mTOR
    (0.025, {"belatacept", "antimetabolite", "steroid"}),
    (0.03, {"cni", "steroid"}),  # dual therapy
]
# Induction: (probability, class or None)
_INDUCTION = [(0.65, "atg"), (0.30, "il2ra"), (0.05, None)]
_P_IN, _P_OUT = 0.98, 0.005  # adherence to / contamination of the latent regimen
# Tuned so the n=285 10-variable cube has roughly the preprint's shape
# (~25-29k non-empty cube rows, ~6k published, ~50 single-patient cells).


def kidney_cohort(n: int = 285, seed: int = 0) -> pd.DataFrame:
    """One row per (synthetic) patient, categorical string columns + patient_ref."""
    rng = np.random.default_rng(seed)
    race = rng.choice(len(RACES), size=n, p=RACE_P)
    sex = rng.choice(2, size=n, p=SEX_P)

    reg = rng.choice(len(_REGIMENS), size=n, p=[p for p, _ in _REGIMENS])
    ind = rng.choice(len(_INDUCTION), size=n, p=[p for p, _ in _INDUCTION])
    med = {}
    for m in MEDS:
        if m in ("atg", "il2ra"):
            planned = np.array([_INDUCTION[i][1] == m for i in ind])
        else:
            planned = np.array([m in _REGIMENS[r][1] for r in reg])
        p = np.where(planned, _P_IN, _P_OUT)
        med[m] = (rng.random(n) < p).astype(int)

    # Rejection (logistic). Coefficients are illustrative, not estimated.
    logit = (
        -1.55
        + 0.45 * (race == 1)
        + 0.15 * (sex == 1)
        - 0.70 * med["cni"]
        + 0.35 * (1 - med["steroid"])
        - 0.30 * med["atg"]
        + 0.60 * med["belatacept"]
        + 0.25 * med["mtor"]
        + 0.40
    )
    rej = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame(
        {
            "patient_ref": [f"Patient/syn-{seed}-{i:06d}" for i in range(n)],
            "race": np.array(RACES, dtype=object)[race],
            "sex": np.array(SEXES, dtype=object)[sex],
            **{m: np.array(YESNO, dtype=object)[med[m]] for m in MEDS},
            "rejection": np.array(YESNO, dtype=object)[rej],
        }
    )
    return df
