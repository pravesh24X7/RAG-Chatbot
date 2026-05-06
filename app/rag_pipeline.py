import os

from rag.retriever import (create_vector_store,
                             get_retriever)
from rag.prompt import generate_prompt
from rag.execution_chain import execute
from app.config import VECTORSTORE_PATH, PROMPT_PATH



class RAGPipeline:
    def __init__(self, filepath: str, ):
        # first create the vector store, if VECTORSTORE_PATH directory is empty.
        if not os.listdir(VECTORSTORE_PATH):
            create_vector_store(filepath=filepath)

        # get the retriever object.
        self.retriever = get_retriever()


        if not os.listdir(PROMPT_PATH):
            print("[+] Generating base prompt template.")
            generate_prompt(name="base.json")

        self.execution_chain = execute(self.retriever,
                                       name="base.json")


    def run(self, query: str) -> str:
        try:
            response = self.execution_chain.invoke({
                "query": query
            })
            return response
        except Exception as e:
            raise RuntimeError(f"Execution failed: {e}")