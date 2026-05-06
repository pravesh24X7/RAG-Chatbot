from rag.retriever import (create_vector_store,
                             get_retriever)
from rag.prompt import generate_prompt
from rag.execution_chain import execute



class RAGPipeline:
    def __init__(self, filepath: str, ):
        # first create the vector store
        create_vector_store(filepath=filepath)

        # get the retriever object.
        self.retriever = get_retriever()

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
            print("[*] Execution failed\n\n", e)