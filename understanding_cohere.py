import cohere
import os
from dotenv import load_dotenv
load_dotenv()
cohere_key = os.getenv("api_key")
co = cohere.Client(api_key=cohere_key)

file_path = "question.txt"

with open(file_path, 'r') as file:
    prompt = file.read()

response = co.generate(
    model="command-r-plus-08-2024",
    prompt=prompt,
    max_tokens=100,
    temperature=0.0
)
print(response.generations[0].text.strip() + "\n")

