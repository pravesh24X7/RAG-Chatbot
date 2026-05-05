from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

from app.config import EMBEDDING_MODEL


load_dotenv()


def get_embedding_model():

    embedding_model = HuggingFaceEmbeddings(model=EMBEDDING_MODEL)
    return embedding_model