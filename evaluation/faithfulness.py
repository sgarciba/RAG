import json
import pandas as pd
from credentials import OPENAI_API_KEY
from openai import OpenAI
from config import CHUNKING_METHODS
from pipeline import (
    doc_loading_and_chunking,
    vector_database_setup,
    query_processing,
    vector_search,
    context_augmentation,
    generate_response,
)

client = OpenAI(api_key=OPENAI_API_KEY)


def _strip_code_fence(text):
    """Removes the ```code fence``` markers that sometimes wrap a model's JSON reply.
    Returns the cleaned text ready to be parsed.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    return text.strip()


def decompose_into_claims(answer):
    """Asks the model to break an answer down into a list of small, individual factual statements.
    Returns that list so each claim can be checked separately.
    """
    prompt = f"""Break the following answer into a list of individual, atomic factual claims.
Return ONLY a JSON array of strings, one claim per element, with no extra commentary.

Answer:
{answer}
"""
    response = client.responses.create(model="gpt-5.6-luna", input=prompt)
    claims = json.loads(_strip_code_fence(response.output_text))
    return claims


def is_claim_supported(claim, context):
    """Asks the model whether a single claim can be backed up by the given context text.
    Returns True if the model answers "Yes", otherwise False.
    """
    prompt = f"""Context:
{context}

Claim: {claim}

Can this claim be inferred from the context above? Answer with only "Yes" or "No".
"""
    response = client.responses.create(model="gpt-5.6-luna", input=prompt)
    return response.output_text.strip().lower().startswith("yes")


def compute_faithfulness(answer, context):
    """Breaks an answer into claims and checks how many are actually supported by the context.
    Returns the fraction of supported claims, e.g. 0.75 means 3 out of 4 claims checked out.
    """
    claims = decompose_into_claims(answer)
    if not claims:
        return None
    supported = sum(is_claim_supported(claim, context) for claim in claims)
    return supported / len(claims)


if __name__ == "__main__":

    with open("./evaluation/eval_dataset.json", "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data["questions"])

    records = []

    for method_name in CHUNKING_METHODS.keys():

        chunking_method = CHUNKING_METHODS[method_name]
        chunks, chunk_sources = doc_loading_and_chunking("corpus", chunking_method)
        collection = vector_database_setup(chunks, chunk_sources, method_name)

        for i in range(len(df)):
            query = df["question"][i]
            print(f"\n[{method_name}] Question {df['id'][i]}: {query}")

            model, query_embedding = query_processing(query)
            search_results, search_sources = vector_search(collection, query_embedding, top_k=2)

            context = "\n\n".join(search_results)
            augmented_prompt = context_augmentation(query, search_results)
            answer = generate_response(augmented_prompt)

            faithfulness_score = compute_faithfulness(answer, context)
            print(f"   > Faithfulness: {faithfulness_score}")

            records.append({
                "question_id": df["id"][i],
                "chunking_method": method_name,
                "answer": answer,
                "faithfulness": faithfulness_score,
            })

    faithfulness_df = pd.DataFrame(records)
    faithfulness_df.to_parquet("./evaluation/faithfulness_results.parquet")

    print("\nFaithfulness by chunking method:")
    print(faithfulness_df.groupby("chunking_method")["faithfulness"].mean())
