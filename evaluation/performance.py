

# 1. Load parquet file
from pathlib import Path
import pandas as pd
import json
from evaluation.faithfulness import compute_faithfulness

# ==========================
# 1. Recall@2
# ==========================

def compute_recall_at_2(merge_df):
    """Checks, for each question, whether the correct source document was among the top 2 retrieved results.
    Returns the average hit rate per chunking method.
    """

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

    return recall_df.groupby("chunking_method")["hit"].mean()


# ==========================
# 2. MRR@2
# ==========================

def compute_mrr_at_2(merge_df):
    """Scores how high up the correct document appeared in the top 2 results (1st place scores higher than 2nd).
    Returns the average of these scores per chunking method.
    """

    merge_df = merge_df.copy()
    merge_df["is_relevant"] = merge_df.apply(
        lambda r: Path(r["retrieved_source"]).name in r["true_source_docs"],
        axis=1
    )

    def reciprocal_rank(group):
        relevant_ranks = group.loc[group["is_relevant"], "rank"]
        return 1 / relevant_ranks.min() if not relevant_ranks.empty else 0

    mrr_df = (
        merge_df
        .groupby(["question_id", "chunking_method"])
        .apply(reciprocal_rank, include_groups=False)
        .reset_index(name="reciprocal_rank")
    )

    return mrr_df.groupby("chunking_method")["reciprocal_rank"].mean()


# ==========================
# 3. Faithfulness
# ==========================

def compute_faithfulness(faithfulness_df):
    """Averages the faithfulness scores (how well answers stick to the source text) across questions.
    Returns one average score per chunking method.
    """
    return faithfulness_df.groupby("chunking_method")["faithfulness"].mean()


if __name__ == "__main__":

    with open("./evaluation/eval_dataset.json", "r") as f:
        data = json.load(f)
    questions_df = pd.DataFrame(data["questions"])
    results_df = pd.read_parquet("./evaluation/retrieval_results.parquet")
    faithfulness_df = pd.read_parquet("./evaluation/faithfulness_results.parquet")

    print(questions_df.shape)
    print(results_df.shape)

    merge_df = questions_df.merge(
        results_df,
        how="left",
        left_on="id",
        right_on="question_id"
    )

    recall_at_2 = compute_recall_at_2(merge_df)
    print("\nRecall@2 by chunking method:")
    print(recall_at_2)

    mrr_at_2 = compute_mrr_at_2(merge_df)
    print("\nMRR@2 by chunking method:")
    print(mrr_at_2)

    faithfulness_at_2 = compute_faithfulness(faithfulness_df)
    print("\nFaithfulness by chunking method:")
    print(faithfulness_at_2)

    # ==========================
    # 4. Summary
    # ==========================

    summary_df = pd.DataFrame({
        "Recall@2": recall_at_2,
        "MRR@2": mrr_at_2,
        "Faithfulness": faithfulness_at_2,
    })

    print("\nSummary of metrics by chunking method:")
    print(summary_df)
