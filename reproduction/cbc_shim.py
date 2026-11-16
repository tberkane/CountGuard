"""PuLP 3.3.2 bundles an x86-64-only CBC binary, which cannot run on Apple silicon
without Rosetta. Point PuLP's default CBC at the same solver (COIN-OR CBC, arm64
build from conda-forge) without touching the original research code."""

import os
import shutil
import sys

import pulp

_CBC = shutil.which("cbc") or os.path.join(sys.prefix, "bin", "cbc")


def install():
    if not os.path.exists(_CBC):
        return False
    _orig = pulp.COIN_CMD

    def PULP_CBC_CMD(*args, **kw):
        kw.setdefault("path", _CBC)
        return _orig(*args, **kw)

    pulp.PULP_CBC_CMD = PULP_CBC_CMD
    pulp.apis.PULP_CBC_CMD = PULP_CBC_CMD
    return True
