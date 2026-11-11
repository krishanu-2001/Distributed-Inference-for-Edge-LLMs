import pandas as pd
import pytest

from src.config import RequestsConfig, load_config
from src.gen_sglang_gsp import generate
from src.request_loader import assign_nodes, load_requests


def frame(n=12):
    return pd.DataFrame({
        "text": [f"q{i}" for i in range(n)],
        "subject": [f"topic{i % 3}" for i in range(n)],
        "target_node": [(i * 7) % 4 for i in range(n)],
    })


def test_round_robin_is_default_and_even():
    df = frame(12)
    nodes = assign_nodes(df, 4, RequestsConfig())
    assert nodes.tolist() == [0, 1, 2, 3] * 3


def test_original_keeps_csv_column():
    df = frame()
    assert assign_nodes(df, 4, RequestsConfig(node_assignment="original")).tolist() == df["target_node"].tolist()


def test_hash_topic_keeps_each_topic_on_one_node():
    df = frame(30)
    nodes = assign_nodes(df, 4, RequestsConfig(node_assignment="hash_topic"))
    assert df.assign(n=nodes).groupby("subject")["n"].nunique().max() == 1


def test_random_is_seeded():
    df = frame(50)
    cfg = RequestsConfig(node_assignment="random", seed=7)
    assert assign_nodes(df, 4, cfg).tolist() == assign_nodes(df, 4, cfg).tolist()


def test_weighted_respects_zero_weight_and_validates():
    df = frame(200)
    nodes = assign_nodes(df, 4, RequestsConfig(node_assignment="weighted", node_weights=[1, 1, 0, 2]))
    assert 2 not in set(nodes)
    with pytest.raises(ValueError):
        assign_nodes(df, 4, RequestsConfig(node_assignment="weighted", node_weights=[1, 1]))
    with pytest.raises(ValueError):
        assign_nodes(df, 4, RequestsConfig(node_assignment="bogus"))


def test_load_requests_and_config(tmp_path):
    csv = tmp_path / "r.csv"
    frame(10).drop(columns=["subject"]).to_csv(csv, index=False)
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(f"requests:\n  path: {csv}\n  node_assignment: round_robin\n")
    cfg = load_config(str(cfg_file))
    df = load_requests(cfg.requests, 4, limit=6)
    assert len(df) == 6 and set(df["subject"]) == {"all"}
    assert df["target_node"].tolist() == [0, 1, 2, 3, 0, 1]


def test_gsp_generator_matches_sglang_shape():
    df = generate(num_groups=4, prompts_per_group=5, system_prompt_len=50,
                  question_len=10, output_len=8, seed=3)
    assert len(df) == 20 and df["subject"].nunique() == 4
    assert df.groupby("subject").size().tolist() == [5] * 4
    lens = df["text"].str.split().str.len()
    assert (lens == 60).all() and (df["output_len"] == 8).all()
    # every prompt in a group shares the exact system prompt
    for _, g in df.groupby("subject"):
        assert g["text"].str.split().str[:50].map(tuple).nunique() == 1
    # deterministic for a seed
    assert generate(num_groups=4, prompts_per_group=5, system_prompt_len=50,
                    question_len=10, output_len=8, seed=3).equals(df)


def test_gsp_zipf_skews_toward_group_zero():
    df = generate(num_groups=8, prompts_per_group=50, system_prompt_len=5,
                  question_len=2, output_len=1, group_distribution="zipf", zipf_alpha=1.2)
    counts = df["subject"].value_counts()
    assert counts.idxmax() == "gsp_group_000"
