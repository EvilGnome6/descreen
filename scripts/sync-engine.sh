#!/bin/bash
# Build a fresh venv in app Resources (venvs aren't relocatable — create in place).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/opt/homebrew/bin/python3}"
if [ ! -x "$PYTHON" ]; then
  PYTHON="$(command -v python3)"
fi

DEST="$ROOT/app/Descreen/Resources/engine"
rm -rf "$DEST"
mkdir -p "$DEST"

cp "$ROOT/engine/descreen.py" "$DEST/descreen.py"
cp "$ROOT/engine/requirements.txt" "$DEST/requirements.txt"

"$PYTHON" -m venv "$DEST/.venv"
"$DEST/.venv/bin/pip" install -q --upgrade pip
"$DEST/.venv/bin/pip" install -q -r "$DEST/requirements.txt"

cat > "$DEST/descreen-engine" << 'EOF'
#!/bin/bash
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec "$ROOT/.venv/bin/python" "$ROOT/descreen.py" "$@"
EOF
chmod +x "$DEST/descreen-engine" "$DEST/descreen.py"

# Dev CLI convenience (local venv at engine/.venv)
"$ROOT/engine/setup.sh"

echo "Synced engine -> app/Descreen/Resources/engine/"
