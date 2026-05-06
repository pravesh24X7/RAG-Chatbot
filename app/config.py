import os

from dotenv import load_dotenv


load_dotenv()


DATA_PATH = "data/raw/"
VECTORSTORE_PATH = "vectorstore/"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PROMPT_PATH = "prompts/"
CHATMODEL = "meta-llama/llama-4-scout-17b-16e-instruct"