#!/bin/bash
# One-word local launcher for Wavelength.
#
# Installed to ~/.local/bin/wavelength-app by scripts/install-launcher.sh.
# From any terminal, `wavelength-app` cds into the project, activates the
# venv (creating and installing it on first run), then serves the UI.
#
# The project location is baked in at install time:
WAVELENGTH_DIR="__WAVELENGTH_DIR__"

set -e

if [ ! -d "$WAVELENGTH_DIR" ]; then
    echo "Wavelength isn't where the launcher expects it:"
    echo "  $WAVELENGTH_DIR"
    echo "Re-run scripts/install-launcher.sh from the project to fix the path."
    exit 1
fi

cd "$WAVELENGTH_DIR"

# Create the venv on first run.
if [ ! -f ".venv/bin/activate" ]; then
    echo "First run — setting up the environment (one time)…"
    python3.12 -m venv .venv 2>/dev/null \
        || python3.11 -m venv .venv 2>/dev/null \
        || python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# Install/upgrade the package if the CLI is missing or --update was passed.
if ! command -v wavelength >/dev/null 2>&1 || [ "$1" = "--update" ]; then
    if [ "$1" = "--update" ]; then
        echo "Updating to the latest version…"
        git pull --ff-only || echo "(couldn't git pull — continuing with local code)"
        shift
    fi
    echo "Installing Wavelength…"
    pip install -e . >/dev/null
fi

exec wavelength serve "$@"
