#!/bin/sh
# First boot: install the separation + labeling models into the /data
# volume (one-time, ~10 minutes, survives container upgrades). Then serve
# in public mode.
set -e

CONFIG_DIR="${HOME}/.config/wavelength"
if [ ! -f "${CONFIG_DIR}/config.toml" ]; then
    mkdir -p "${CONFIG_DIR}"
    # Servers don't need source archiving (disk) and always use bandit.
    printf 'engine = "bandit"\narchive_sources = false\n' \
        > "${CONFIG_DIR}/config.toml"
fi

echo "[wavelength] ensuring models are installed (first boot takes ~10 min)"
wavelength setup bandit
wavelength setup clap || echo "[wavelength] CLAP setup failed; serving without labels"

exec wavelength serve --public --host 0.0.0.0 --port 8317 --no-browser
