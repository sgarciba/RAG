

# 1. Load parquet file
from pathlib import Path
import pandas as pd
import json
from evaluation.eval_pipeline import evaluation_pipeline


with open("./evaluation/eval_dataset.json", "r") as f:
    data = json.load(f)
questions_df = pd.DataFrame(data["questions"])
results_df = pd.read_parquet("./evaluation/retrieval_results.parquet")

print(questions_df.shape)
print(results_df.shape)


# ==========================
# 1. Recall@2
# ==========================

merge_df = questions_df.merge(
    results_df,
    how="left",
    left_on="id",
    right_on="question_id"
)

recall_df = (
    merge_df
    .groupby(["question_id", "chunking_method"])
    .agg(
        true_source_docs=("true_source_docs", "first"),
        pred_source_docs=("retrieved_source", list)
    )
    .reset_index()
)

recall_df["hit"] = recall_df.apply(
    lambda r: any(Path(p).name in r["true_source_docs"] for p in r["pred_source_docs"]),
    axis=1
)

recall_at_2 = recall_df.groupby("chunking_method")["hit"].mean()

print("\nRecall@2 by chunking method:")
print(recall_at_2)

#.to_parquet("./evaluation/recall_at_2.parquet")

