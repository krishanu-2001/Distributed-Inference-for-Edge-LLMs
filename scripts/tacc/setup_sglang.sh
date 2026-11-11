#!/bin/bash
# One-time setup on a Lonestar6 login node: SGLang venv + Qwen2.5-0.5B-Instruct.
# Follows the conventions of $SCRATCH/llm/setup_vllm.sh (python/3.12.11, torch cu129).
set -euo pipefail
source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
module load python/3.12.11

BASE=${LLM_BASE:-/scratch/11018/krishanu21saini/llm}
VENV=${SGLANG_VENV:-$BASE/venv-sglang}
export HF_HOME=$BASE/hf
export XDG_CACHE_HOME=$BASE/cache
export PIP_CACHE_DIR=$BASE/cache/pip
export TMPDIR=/tmp/sglang-pip-$USER; mkdir -p "$TMPDIR"
CU=https://download.pytorch.org/whl/cu129
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}

if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install -q --upgrade pip setuptools wheel

echo "=== [1/3] sglang (torch from cu129 index) ==="
"$VENV/bin/python" -m pip install --extra-index-url "$CU" "sglang[all]" "huggingface_hub[hf_transfer]"

echo "=== [2/3] simulator deps ==="
"$VENV/bin/python" -m pip install aiohttp pyyaml pandas pyarrow

echo "=== [3/3] model: $MODEL ==="
HF_HUB_ENABLE_HF_TRANSFER=1 "$VENV/bin/python" -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$MODEL', allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.tiktoken']))"

echo "=== VERIFY ==="
"$VENV/bin/python" -c "import torch, sglang; print('torch', torch.__version__, 'cuda', torch.version.cuda, '| sglang', sglang.__version__)"
echo "=== DONE ==="
