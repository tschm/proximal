# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.x.x   | :white_check_mark: |

As this project is currently in initial development (version 0.x.x), all releases receive security updates.

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

1. **Do NOT** create a public GitHub issue for security vulnerabilities
2. Email the maintainer directly at thomas.schmelzer@gmail.com
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes (optional)

### What to Expect

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days with assessment
- **Resolution Timeline**: Depends on severity
  - Critical: Within 7 days
  - High: Within 14 days
  - Medium/Low: Within 30 days

### After Reporting

- We will acknowledge receipt of your report
- We will investigate and assess the vulnerability
- We will keep you informed of our progress
- Once fixed, we will publicly acknowledge your contribution (if desired)

### Code Scanning
- **CodeQL**: Automated code scanning for Python and GitHub Actions
- **Bandit**: Python security linter integrated in CI and pre-commit
- **Secret Scanning**: GitHub secret scanning enabled on this repository
- **Fuzzing**: ClusterFuzzLite exercises Atheris-based fuzz targets on pull requests and scheduled batch runs

When using `proximal-lq`:

1. **Input Validation**: Always validate input data before passing to library functions
2. **Dependency Updates**: Keep the library updated to the latest version
3. **Environment**: Use virtual environments to isolate dependencies

## Known Security Considerations

This library:

- Does not handle sensitive data directly
- Does not make network requests
- Does not execute arbitrary code
- Only processes numerical arrays via NumPy

The primary attack surface is malformed input arrays. The library validates inputs to prevent common numerical issues.
