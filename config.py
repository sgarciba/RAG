from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

CHUNKING_METHODS = {
    "markdown_header": MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "Header 2")],
        strip_headers=False
    ),
    "rec_char_256_0": RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=0),
    "rec_char_512_50": RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50),
    "rec_char_1024_100": RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100),
}
