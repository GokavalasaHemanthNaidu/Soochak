# Contributing to SOOCHAK

## Branch Protection
- **Never push directly to `main`.** All changes must go through a Pull Request.
- PRs require CI to pass (pytest 75% coverage, ruff, Bandit, Safety).
- At least one review is required before merge.
- `.env` and `*.pem` files are blocked by CI; if found, build fails immediately.
