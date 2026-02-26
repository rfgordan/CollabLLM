#!/usr/bin/env bash
set -euo pipefail

# cd /workspace

# 1. Install uv if missing
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
source $HOME/.local/bin/env

# Persist uv to PATH for future shells
if ! grep -q '.local/bin/env' ~/.bashrc; then
    echo 'source $HOME/.local/bin/env' >> ~/.bashrc
fi

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

# Fix executable permissions for venv binaries missing execute bits (RunPod issue)
find .venv -type f -path "*/bin/*" ! -perm -111 -exec chmod +x {} +

# 4. Optional: start shell or training
# exec "$@"
