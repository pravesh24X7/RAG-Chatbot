from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

from app.config import VECTORSTORE_PATH, EMBEDDING_MODEL
from .embeddings import get_embedding_model
from .text_splitter import get_all_documents


load_dotenv()


def create_vector_store(filepath: str):

    embedding_model = get_embedding_model()

    vector_store = FAISS.from_documents(
        embedding=embedding_model,
        documents=get_all_documents(filepath=filepath)
    )

    vector_store.save_local(VECTORSTORE_PATH)


def load_vector_store():

    embedding_model = HuggingFaceEmbeddings(model=EMBEDDING_MODEL)
    vector_store = FAISS.load_local(VECTORSTORE_PATH, embeddings=embedding_model,
                                    allow_dangerous_deserialization=True)
    
    return vector_store


def get_retriever():
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_type="mmr",
                                          search_kwargs={
                                              "k": 5,
                                              "lambda_mult": 0.5,
                                          })
    
    return retriever