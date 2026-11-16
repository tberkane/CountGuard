# Security policy

CountGuard is a privacy tool, so a bug can expose patients. Please report suspected
vulnerabilities privately, not in a public issue.

## How to report

Use GitHub's private vulnerability reporting: on the repository page, open **Security →
Report a vulnerability**. Include the CountGuard version, a minimal example (synthetic data
only, never real patient data), and what you expected to happen.

You can expect an acknowledgement within 5 business days and an assessment within 15 business
days. Fixes are released as a patch version with a CHANGELOG entry and a GitHub security
advisory. Reporters are credited unless they prefer otherwise.

## What counts as a vulnerability

- the attack under-reports risk: a released table lets an attacker learn more than the audit says
  (for example, bounds that do not contain the truth, or a missed exact recovery);
- privacy accounting spends more than the stated budget, or noise is drawn from a weaker source
  than documented;
- a protected release leaks information outside the mechanism's guarantee (for example through
  table formatting, ordering or metadata);
- any path by which CountGuard reads or writes data it should not.

## Known limitations (not vulnerabilities)

- `DiscreteGaussian(seed=...)` and the `--seed` CLI option use NumPy's generator so benchmarks are
  reproducible. They are documented as unsuitable for real releases; use the default (OpenDP's
  secure sampler).
- The audit simulates the attacks described in the documentation. It is not a proof that no other
  attack exists.

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | yes |
