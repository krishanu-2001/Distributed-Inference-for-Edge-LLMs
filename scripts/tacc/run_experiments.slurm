#!/bin/bash
#SBATCH -J dist-sglang
#SBATCH -A IRI26037
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -o runs/tacc/job-%j.txt
#SBATCH --mail-user=ks59833@eid.utexas.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#
# Decentralized cache-aware routing on 4 real SGLang servers (Qwen2.5-0.5B-Instruct)
# on one Lonestar6 gpu-a100 node (3x A100 40GB; servers 0 and 3 share GPU 0).
# Each server's KV cache is capped with --max-total-tokens to create cache pressure.
# Requests are sent one at a time (batching-free, as in the paper), so servers that
# share a GPU never compete for it.
#
# One-time setup on a login node:  bash scripts/tacc/setup_sglang.sh
#   python -m src.gen_sglang_gsp --num-groups 128 --prompts-per-group 32 --output-len 32 --out data/sglang_gsp_big.csv
#   python -m src.gen_sglang_gsp --num-groups 128 --prompts-per-group 32 --output-len 32 \
#       --group-distribution zipf --zipf-alpha 1.1 --out data/sglang_gsp_big_zipf.csv
#   python -m src.abcd     # data/requests.csv (MMLU)
# Submit from the repo root:  mkdir -p runs/tacc && sbatch scripts/tacc/run_experiments.slurm
#
# The job exits when the matrix finishes (no idle billing). Runs that would not fit
# in the remaining time are skipped and listed in the summary.

set -uo pipefail
REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$REPO"

BASE=${LLM_BASE:-/scratch/11018/krishanu21saini/llm}
VENV=${SGLANG_VENV:-$BASE/venv-sglang}
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
PY=$VENV/bin/python
NUM_NODES=4
SGLANG_PORT0=30000
JOB_SECONDS=${JOB_SECONDS:-$((4 * 3600))}
# SMOKE=1: 30 requests per run, to check the stack end-to-end on gpu-a100-dev:
#   SMOKE=1 JOB_SECONDS=1800 sbatch -p gpu-a100-dev -t 00:30:00 scripts/tacc/run_experiments.slurm
SMOKE=${SMOKE:-0}
LIMIT_ARGS=(); [ "$SMOKE" = 1 ] && LIMIT_ARGS=(--limit 30)
# RUNS=gsp or RUNS=mmlu runs only that half of the matrix (default: all)
RUNS=${RUNS:-all}
START=$(date +%s)
DEADLINE=$((START + JOB_SECONDS - 600))   # keep 10 minutes for shutdown + summary

# Environment, following vllm-kit/serve.sh (proven on these nodes)
source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
module load python/3.12.11
module load cuda/12.8 2>/dev/null || true             # FlashInfer JIT needs a real nvcc
# SGLang's JIT kernels are C++20 (<concepts>); the default gcc 9.4 can't build them.
# Use gcc 13.2 by path (its module swaps MODULEPATH and drops cuda/12.8).
GCC=/scratch/tacc/apps/gcc/13.2.0
export CUDA_HOME=/opt/apps/cuda/12.8 PATH="$GCC/bin:/opt/apps/cuda/12.8/bin:$VENV/bin:$PATH"
export LD_LIBRARY_PATH="$GCC/lib64:$GCC/lib:${LD_LIBRARY_PATH:-}"
export CC=$GCC/bin/gcc CXX=$GCC/bin/g++ CUDAHOSTCXX=$GCC/bin/g++ NVCC_CCBIN=$GCC/bin/g++
export HF_HOME=$BASE/hf HF_HUB_OFFLINE=1
export FLASHINFER_WORKSPACE_BASE=$BASE/cache XDG_CACHE_HOME=$BASE/cache \
       TORCHINDUCTOR_CACHE_DIR=$BASE/cache/inductor TRITON_CACHE_DIR=$BASE/cache/triton \
       TVM_FFI_CACHE_DIR=$BASE/cache/tvm-ffi
mkdir -p "$BASE/cache"

OUT=$REPO/runs/tacc/job_${SLURM_JOB_ID:-local}
mkdir -p "$OUT"
NGPU=$(nvidia-smi -L | wc -l)
echo "node $(hostname)  gpus=$NGPU  out=$OUT  start=$(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$PY" -c "import torch, sglang; print('torch', torch.__version__, '| sglang', sglang.__version__)"
echo "host compiler: $(g++ --version | head -1)"

SERVER_PIDS=()

stop_servers() {
  [ ${#SERVER_PIDS[@]} -eq 0 ] && return
  kill "${SERVER_PIDS[@]}" 2>/dev/null
  wait "${SERVER_PIDS[@]}" 2>/dev/null
  SERVER_PIDS=()
  pkill -f "sglang.launch_server" 2>/dev/null; sleep 5
}
trap stop_servers EXIT

start_servers() {  # $1 = KV cap (tokens), $2 = tag for log names
  local cap=$1 tag=$2
  stop_servers
  echo "[$(date +%T)] starting $NUM_NODES SGLang servers, --max-total-tokens $cap"
  for i in $(seq 0 $((NUM_NODES - 1))); do
    CUDA_VISIBLE_DEVICES=$((i % NGPU)) "$PY" -m sglang.launch_server \
      --model-path "$MODEL" --host 127.0.0.1 --port $((SGLANG_PORT0 + i)) \
      --max-total-tokens "$cap" --mem-fraction-static 0.3 --context-length 4096 \
      --random-seed 0 --log-level warning \
      > "$OUT/sglang_${tag}_node$i.log" 2>&1 &
    SERVER_PIDS+=($!)
  done
  for i in $(seq 0 $((NUM_NODES - 1))); do
    local ok=0
    for _ in $(seq 1 300); do   # up to 25 min (a cold kernel cache takes ~15 min)
      curl -sf "http://127.0.0.1:$((SGLANG_PORT0 + i))/health" >/dev/null && { ok=1; break; }
      sleep 5
    done
    if [ $ok = 0 ]; then
      echo "server $i did not come up; tail of its log:"; tail -30 "$OUT/sglang_${tag}_node$i.log"
      return 1
    fi
  done
  echo "[$(date +%T)] servers ready"
  # Check the response carries what the simulator records
  curl -s "http://127.0.0.1:$SGLANG_PORT0/generate" -H 'Content-Type: application/json' \
    -d '{"input_ids":[1,2,3,4,5,6,7,8],"sampling_params":{"max_new_tokens":4,"temperature":0,"ignore_eos":true}}' \
    | "$PY" -c "import json,sys; d=json.load(sys.stdin); d=d[0] if isinstance(d,list) else d; print('generate check: keys', sorted(d), '| meta', {k: d['meta_info'].get(k) for k in ('prompt_tokens','completion_tokens','cached_tokens')})"
}

flush_caches() {
  for i in $(seq 0 $((NUM_NODES - 1))); do
    curl -sf -X POST "http://127.0.0.1:$((SGLANG_PORT0 + i))/flush_cache" >/dev/null \
      || curl -sf "http://127.0.0.1:$((SGLANG_PORT0 + i))/flush_cache" >/dev/null || true
  done
  sleep 2
}

RUN_DIRS=()
SKIPPED=()
run_one() {  # name config requests policy est_minutes
  local name=$1 cfg=$2 req=$3 policy=$4 est=$5
  local remaining=$((DEADLINE - $(date +%s)))
  [ "$SMOKE" = 1 ] && est=1
  if [ $remaining -lt $((est * 60)) ]; then
    echo "[$(date +%T)] SKIP $name: needs ~${est} min, ${remaining}s left"
    SKIPPED+=("$name"); return
  fi
  flush_caches
  echo "[$(date +%T)] RUN $name ($policy, $req)"
  timeout $remaining "$PY" -m src.main --config "$cfg" --requests "$req" --policy "$policy" \
    --exit --run-dir "$OUT/$name" "${LIMIT_ARGS[@]}" > "$OUT/$name.stdout" 2>&1
  echo "[$(date +%T)] done $name exit=$?  errors=$(grep -c Traceback "$OUT/$name.stdout")"
  RUN_DIRS+=("$OUT/$name")
}

# --- Experiment matrix (most important first) ---
if [[ $RUNS =~ all|gsp ]] && start_servers 32768 gsp; then
  run_one gsp_uniform_routed config/tacc/sglang_gsp_big.yaml data/sglang_gsp_big.csv      cache_aware 25
  run_one gsp_uniform_self   config/tacc/sglang_gsp_big.yaml data/sglang_gsp_big.csv      self        25
  run_one gsp_zipf_routed    config/tacc/sglang_gsp_big.yaml data/sglang_gsp_big_zipf.csv cache_aware 25
  run_one gsp_zipf_self      config/tacc/sglang_gsp_big.yaml data/sglang_gsp_big_zipf.csv self        25
fi
if [[ $RUNS =~ all|mmlu ]] && start_servers 6144 mmlu; then
  run_one mmlu_routed config/tacc/sglang_mmlu.yaml data/requests.csv cache_aware 30
  run_one mmlu_self   config/tacc/sglang_mmlu.yaml data/requests.csv self        30
fi
stop_servers

echo; echo "=== SUMMARY ($(date)) ==="
[ ${#SKIPPED[@]} -gt 0 ] && echo "skipped (out of time): ${SKIPPED[*]}"
if [ ${#RUN_DIRS[@]} -gt 0 ]; then
  "$PY" -m src.summarize "${RUN_DIRS[@]}" | tee "$OUT/summary.txt"
fi
echo "elapsed $(( ($(date +%s) - START) / 60 )) min"
