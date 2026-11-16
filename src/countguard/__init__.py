"""CountGuard: audit and protect aggregate count-cube releases."""

from .attack import AttackResult, InfeasibleRelease, run_attack
from .cube import Cube, Schema
from .cumulus import read_table, release_to_table, write_table
from .mechanisms import DiscreteGaussian, Generalize, Pipeline, PrivacyBudgetError, Release, Round, Suppress
from .metrics import Privacy, accuracy_loss, assess, make_queries
from .synth import KIDNEY_VARS, SUBCUBE_6, SUBCUBE_8, kidney_cohort

__all__ = [
    "AttackResult",
    "Cube",
    "DiscreteGaussian",
    "Generalize",
    "InfeasibleRelease",
    "Pipeline",
    "Privacy",
    "PrivacyBudgetError",
    "Release",
    "Round",
    "Schema",
    "Suppress",
    "accuracy_loss",
    "assess",
    "kidney_cohort",
    "make_queries",
    "read_table",
    "release_to_table",
    "run_attack",
    "write_table",
    "KIDNEY_VARS",
    "SUBCUBE_6",
    "SUBCUBE_8",
]
__version__ = "0.1.0"
