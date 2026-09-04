from credentials import OPENAI_API_KEY
from openai import OpenAI


client = OpenAI(api_key=OPENAI_API_KEY)

response = client.responses.create(
    model="gpt-5.6-luna",
    input="Say hello! Explain in one sentence what an LLM is."
)

print(response.output_text)


