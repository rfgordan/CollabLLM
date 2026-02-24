#!/usr/bin/env bash
set -euo pipefail

# cd /workspace

# 1. Install uv if missing
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
source $HOME/.local/bin/env

# 2. Clone or update repo
# if [ ! -d CollabLLM-repro ]; then
#   git clone https://github.com/rfgordan/CollabLLM-repro
# else
#   cd CollabLLM-repro
#   git pull --rebase
#   cd ..
# fi

cd CollabLLM

# 3. Sync deps
uv sync
source .venv/bin/activate

# Fix triton backend executable permissions (RunPod installs without execute bits)
chmod -R +x .venv/lib/python*/site-packages/triton/backends/nvidia/bin/ 2>/dev/null || true

# 4. Optional: start shell or training
# exec "$@"
