"""Convert MMLU.parquet to a 5-shot CSV for distributed inference."""

import pandas as pd


ANSWER_LETTERS = ["A", "B", "C", "D"]


def format_question(row: pd.Series) -> str:
    """Format a single MMLU question with lettered choices."""
    lines = [row["question"]]
    for letter, choice in zip(ANSWER_LETTERS, row["choices"]):
        lines.append(f"{letter}. {choice}")
    return "\n".join(lines)


def format_example(row: pd.Series) -> str:
    """Format a question + its correct answer (for few-shot context)."""
    return format_question(row) + f"\nAnswer: {ANSWER_LETTERS[row['answer']]}"


def build_prompt(few_shot_examples: list[str], query_row: pd.Series) -> str:
    """Build a full 5-shot prompt for a query."""
    parts = few_shot_examples + [format_question(query_row) + "\nAnswer:"]
    return "\n\n".join(parts)


def main():
    df = pd.read_parquet("src/MMLU.parquet")

    print(f"Loaded {len(df)} rows, {df['subject'].nunique()} subjects")
    print(f"Rows per subject (min/max): {df['subject'].value_counts().min()} / {df['subject'].value_counts().max()}")

    rows = []
    node_counter = 0
    num_nodes = 4

    for subject, group in df.groupby("subject"):
        group = group.reset_index(drop=True)

        # First 5 rows are few-shot examples
        shot_rows = group.iloc[:5]
        query_rows = group.iloc[5:]

        few_shot_examples = [format_example(row) for _, row in shot_rows.iterrows()]

        if len(query_rows) == 0:
            continue

        for _, row in query_rows.iterrows():
            prompt = build_prompt(few_shot_examples, row)
            rows.append({
                "text": prompt,
                "target_node": node_counter % num_nodes,
                "subject": subject,
            })
            node_counter += 1

    out = pd.DataFrame(rows)
    out = out.sample(frac=1, random_state=42).reset_index(drop=True)
    out.to_csv("data/requests.csv", index=False)

    print(f"\nOutput: {len(out)} queries written to data/requests.csv")
    print(f"Subjects with <6 rows (skipped queries): {(df['subject'].value_counts() < 6).sum()}")
    print(f"Target node distribution:\n{out['target_node'].value_counts().sort_index()}")
    print(f"\n--- Sample prompt (first row, truncated) ---")
    print(out.iloc[0]["text"][:500])


if __name__ == "__main__":
    main()
