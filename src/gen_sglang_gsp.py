"""Generate SGLang's "generated-shared-prefix" (gsp) benchmark workload as a CSV.

Mirrors sglang.benchmark.datasets.generated_shared_prefix
(`python -m sglang.bench_serving --dataset-name generated-shared-prefix`):
`num_groups` random system prompts, each shared by `prompts_per_group`
requests that append a random question; lengths are sampled uniformly in
[len * range_ratio, len]; groups are uniform or Zipf-distributed; requests are
shuffled unless --ordered.

Differences from SGLang: prompts are random words ("w<id>"), one word per
token, so lengths are exact under our whitespace mock tokenizer and no HF
tokenizer is needed.

Output columns: text, subject (= prefix group), output_len, target_node
(placeholder; nodes are assigned at run time by src/request_loader.py).

    python -m src.gen_sglang_gsp --out data/sglang_gsp.csv
    python -m src.gen_sglang_gsp --group-distribution zipf --zipf-alpha 1.1 \
        --out data/sglang_gsp_zipf.csv
"""

import argparse
import os
import random

import numpy as np
import pandas as pd

VOCAB_SIZE = 32000


def compute_random_lens(rng, full_len: int, range_ratio: float, num: int) -> np.ndarray:
    """Same sampling as sglang's compute_random_lens."""
    if full_len <= 0:
        return np.zeros(num, dtype=int)
    return rng.randint(max(int(full_len * range_ratio), 1), full_len + 1, size=num)


def gen_prompt(rng, length: int) -> str:
    return " ".join(f"w{t}" for t in rng.randint(0, VOCAB_SIZE, size=length))


def zipf_group_probs(num_groups: int, alpha: float) -> np.ndarray:
    weights = 1.0 / np.arange(1, num_groups + 1, dtype=np.float64) ** alpha
    return weights / weights.sum()


def generate(
    num_groups=64, prompts_per_group=16, system_prompt_len=2048,
    question_len=128, output_len=256, range_ratio=1.0, seed=1,
    group_distribution="uniform", zipf_alpha=None, ordered=False,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    sys_lens = compute_random_lens(rng, system_prompt_len, range_ratio, num_groups)
    q_lens = compute_random_lens(
        rng, question_len, range_ratio, num_groups * prompts_per_group
    ).reshape(num_groups, prompts_per_group)
    out_lens = compute_random_lens(
        rng, output_len, range_ratio, num_groups * prompts_per_group
    ).reshape(num_groups, prompts_per_group)

    system_prompts = [gen_prompt(rng, int(n)) for n in sys_lens]
    questions = [
        [gen_prompt(rng, int(q_lens[g, p])) for p in range(prompts_per_group)]
        for g in range(num_groups)
    ]

    total = num_groups * prompts_per_group
    if group_distribution == "uniform":
        assignment = np.repeat(np.arange(num_groups), prompts_per_group)
    elif group_distribution == "zipf":
        if zipf_alpha is None or zipf_alpha <= 0:
            raise ValueError("zipf requires --zipf-alpha > 0")
        # Isolated RNG, as in sglang, so the prompt pool matches uniform mode
        assignment = np.random.default_rng(seed).choice(
            num_groups, size=total, p=zipf_group_probs(num_groups, zipf_alpha)
        )
    else:
        raise ValueError(f"unknown group_distribution {group_distribution!r}")

    rows = []
    for slot, g in enumerate(assignment):
        src_g, src_p = divmod(slot, prompts_per_group)
        rows.append({
            "text": f"{system_prompts[g]}\n\n{questions[src_g][src_p]}",
            "subject": f"gsp_group_{int(g):03d}",
            "output_len": int(out_lens[src_g, src_p]),
            "target_node": 0,
        })
    if not ordered:
        random.Random(seed).shuffle(rows)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--num-groups", type=int, default=64)
    p.add_argument("--prompts-per-group", type=int, default=16)
    p.add_argument("--system-prompt-len", type=int, default=2048)
    p.add_argument("--question-len", type=int, default=128)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--range-ratio", type=float, default=1.0)
    p.add_argument("--group-distribution", choices=["uniform", "zipf"], default="uniform")
    p.add_argument("--zipf-alpha", type=float, default=None)
    p.add_argument("--ordered", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="data/sglang_gsp.csv")
    a = p.parse_args()

    df = generate(
        a.num_groups, a.prompts_per_group, a.system_prompt_len, a.question_len,
        a.output_len, a.range_ratio, a.seed, a.group_distribution, a.zipf_alpha,
        a.ordered,
    )
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    lens = df["text"].str.split().str.len()
    print(f"Wrote {len(df)} requests to {a.out}")
    print(f"  groups: {df['subject'].nunique()} ({a.group_distribution})")
    print(f"  input tokens: mean {lens.mean():.0f}, total {lens.sum()}")
    print(f"  output tokens: mean {df['output_len'].mean():.0f}")


if __name__ == "__main__":
    main()
