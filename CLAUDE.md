# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Proximal is a Python library for solving constrained linear least squares problems:
```
minimize 0.5 ||mat @ x - vec||^2
subject to x >= 0, sum(x) = 1
```

The library exports two functions from `proximal_lq`:
- `proj_simplex(vec, rad=1)` - Projects a vector onto the probability simplex
- `prox_gradient(mat, vec, eps_rel=1e-6, max_iter=1000)` - Proximal gradient descent solver

## Development Commands

```bash
make install    # Install uv, create .venv, install dependencies
make test       # Run pytest with coverage
make fmt        # Format code with ruff (format + check --fix)
make deptry     # Check for missing/unused dependencies
make marimo     # Start Marimo notebook server
make book       # Build documentation
```

## Project Structure

- `src/proximal_lq/` - Main library code (proximal.py contains core algorithms)
- `tests/` - Test suite using pytest
- `book/marimo/` - Interactive Marimo notebooks for documentation
- `.rhiza/` - Rhiza framework config (template-managed, do not edit directly)

## Tech Stack

- **Package manager**: uv
- **Testing**: pytest with coverage
- **Linting/Formatting**: ruff (line length 120, Google-style docstrings)
- **Build system**: hatchling
- **Runtime dependency**: NumPy >=2.0.0, Python >=3.11

## Workflow

1. Run `make install` to set up the environment
2. Write code in `src/` and tests in `tests/`
3. Run `make test` to verify changes
4. Run `make fmt` before committing
5. Run `make deptry` to check dependencies
