from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import EMBEDDING_MODEL


def get_embedding_model():

    embedding_model = HuggingFaceEmbeddings(model=EMBEDDING_MODEL)
    return embedding_model