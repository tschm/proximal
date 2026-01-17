# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Comprehensive test suite with parametrized tests for edge cases
- Type annotations for all public functions
- Performance benchmark tests
- CHANGELOG.md for tracking changes
- PyPI classifiers for better discoverability
- docs/DEVELOPMENT.md with algorithm documentation
- Portfolio optimization example in examples/
- Security (bandit) rules enabled in ruff configuration

### Changed

- Improved docstrings with examples and algorithm references
- Enhanced module documentation in __init__.py
- Updated SECURITY.md with actual reporting procedures

## [0.0.0] - 2024-01-01

### Added

- Initial implementation of `proj_simplex` function for simplex projection
- Initial implementation of `prox_gradient` function for constrained optimization
- Basic test coverage with CLA data validation
- Marimo notebook demonstration
- CI/CD workflows for testing and releases

[Unreleased]: https://github.com/tschm/proximal/compare/v0.0.0...HEAD
[0.0.0]: https://github.com/tschm/proximal/releases/tag/v0.0.0
