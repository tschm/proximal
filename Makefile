## Makefile (repo-owned)
# Keep this file small. It can be edited without breaking template sync.

# Override template default: notebooks live in book/marimo, matching the folder
# that .github/workflows/rhiza_marimo.yml discovers. The rhiza.mk default is
# docs/notebooks, which does not exist here, so the marimo/book targets would
# silently skip. Must be set above the include to beat rhiza.mk's ?= default.
MARIMO_FOLDER = book/marimo

# Override template default: include mkdocstrings plugin for API docs
MKDOCS_EXTRA_PACKAGES = --with 'mkdocstrings[python]'

# Always include the Rhiza API (template-managed)
include .rhiza/rhiza.mk

# Optional: developer-local extensions (not committed)
-include local.mk
