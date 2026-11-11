"""Summarize one or more run directories into a comparison table.

    python -m src.summarize runs/run_A runs/run_B ...

Failed requests are counted separately and excluded from latency/hit means,
so the error column must be checked alongside them.
"""

import sys

import pandas as pd
import yaml


def summarize(run_dir: str) -> dict:
    df = pd.read_csv(f"{run_dir}/results.csv", sep="|")
    try:
        with open(f"{run_dir}/config.yaml") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        cfg = {}
    ok = df[df["error"].isna()]
    share = ok["processed_by"].value_counts(normalize=True)
    return {
        "run": run_dir.rstrip("/").split("/")[-1],
        "requests": cfg.get("requests", {}).get("path", "?").split("/")[-1],
        "policy": cfg.get("router", {}).get("policy", "?"),
        "assign": cfg.get("requests", {}).get("node_assignment", "?"),
        "n": len(df),
        "errors": int(df["error"].notna().sum()),
        "hit_%": round(100 * ok["cache_hit_ratio"].mean(), 1),
        # what the routing node predicted for the node it picked (paper: "Exp. Hit")
        "exp_hit_%": round(100 * (ok["route_expected_tokens"] / ok["input_tokens"]).mean(), 1)
        if "route_expected_tokens" in ok and ok["route_expected_tokens"].notna().any() else None,
        "fwd_%": round(100 * ok["routed_to"].notna().mean(), 1),
        "hot_%": round(100 * share.max(), 1) if len(share) else None,
        "lat_mean_s": round(ok["total_time_s"].mean(), 4),
        "lat_p90_s": round(ok["total_time_s"].quantile(0.9), 4),
        "infer_mean_s": round(ok["inference_time_s"].mean(), 4),
        "uncached_tok": int(ok["uncached_tokens"].fillna(0).sum()) if "uncached_tokens" in ok else None,
    }


def main(dirs: list[str]):
    table = pd.DataFrame([summarize(d) for d in dirs])
    print(table.to_string(index=False))
    return table


if __name__ == "__main__":
    main(sys.argv[1:])
