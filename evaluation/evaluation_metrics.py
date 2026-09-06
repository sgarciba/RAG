from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
import pandas as pd
import json
from pipeline import complete_rag_pipeline

# ==============
# CHUNKING METHODS
# =============

chunking_methods = {
    "rec_char_256_0": RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=0),
    "rec_char_512_50": RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50),
    "rec_char_1024_100": RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100),
    "markdown_header": MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "Header 2")],
        strip_headers=False
    ),
}



with open("./evaluation/eval_dataset.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data["questions"])


N = len(df)

for i in range(0, N):

    query = df['question'][i]
    response = complete_rag_pipeline(df['question'][i])
    break




# ===============
# 1. Recall@5
# ===============



