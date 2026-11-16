# Contributing

Thanks for your interest in CountGuard. Issues and pull requests are welcome.

## Development setup

```bash
git clone https://github.com/tberkane/CountGuard && cd CountGuard
python -m venv .venv && source .venv/bin/activate     # Python ≥ 3.12
pip install -e ".[dev,docs]"
```

## Checks (the same ones CI runs)

```bash
ruff check . && ruff format --check .   # lint and formatting
mypy                                     # type checks (src/countguard)
pytest --cov                             # tests with coverage
mkdocs build --strict                    # documentation
```

## Guidelines

- **Synthetic data only.** Never commit, paste or attach real patient data, including in issues.
  Use `countguard synth` or `countguard.synth.kidney_cohort` for examples.
- **Soundness first.** A change to a mechanism or to the attack needs a test showing the released
  ranges, or the proven bounds, still contain the truth. Property tests in
  `tests/test_properties.py` are the natural place.
- **Public API** changes need type hints, docstrings and a CHANGELOG entry under "Unreleased".
- **The original research code** in `reproduction/original/` stays unmodified.
- Every pull request is reviewed before merging.

## Reporting security issues

Please follow [SECURITY.md](SECURITY.md) instead of opening a public issue.
