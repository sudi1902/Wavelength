#!/bin/bash
# Installs the `wavelength-app` one-word launcher into ~/.local/bin.
#
# Run once from the project root:
#     bash scripts/install-launcher.sh
#
# Afterwards, `wavelength-app` starts the app from any terminal.
# `wavelength-app --update` pulls the latest code first.

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="$HOME/.local/bin"
TARGET="$BIN_DIR/wavelength-app"

mkdir -p "$BIN_DIR"

# Bake the project path into the launcher so it works from anywhere.
sed "s|__WAVELENGTH_DIR__|$PROJECT_DIR|" \
    "$PROJECT_DIR/deploy/wavelength-app.sh" > "$TARGET"
chmod +x "$TARGET"

echo "Installed: $TARGET"
echo "           (points at $PROJECT_DIR)"

# Make sure ~/.local/bin is on PATH for the user's shell.
add_path_line='export PATH="$HOME/.local/bin:$PATH"'
ensure_path() {
    local rc="$1"
    # Create the file if absent (create=1), or only touch it if it exists.
    local create="$2"
    [ -f "$rc" ] || [ "$create" = "create" ] || return 0
    if ! grep -qF "$add_path_line" "$rc" 2>/dev/null; then
        printf '\n# Added by Wavelength launcher installer\n%s\n' \
            "$add_path_line" >> "$rc"
        echo "Added ~/.local/bin to PATH in $rc"
    fi
}

case ":$PATH:" in
    *":$BIN_DIR:"*)
        already_on_path=1 ;;
    *)
        already_on_path=0
        # zsh is the macOS default — always ensure .zshrc (creating it if
        # needed). Also cover bash if those files are already present.
        ensure_path "$HOME/.zshrc" create
        ensure_path "$HOME/.bashrc"
        ensure_path "$HOME/.bash_profile"
        ;;
esac

echo
echo "Done. Start the app any time with:"
echo "    wavelength-app"
echo
if [ "$already_on_path" = "0" ]; then
    echo "Open a NEW terminal window first (so the PATH change takes effect),"
    echo "or run this once in your current window:"
    echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
fi
