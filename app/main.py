from app.rag_pipeline import RAGPipeline
from app.config import DATA_PATH


def main():
    pipeline = RAGPipeline(
        filepath=f"{DATA_PATH}demo.pdf"
    )

    print("[*] Environment setup completed ...")
    while True:
        user_query = input("[ HUMAN ] : ").strip()

        if user_query.lower() == "exit":
            break

        response = pipeline.run(query=user_query)
        print(f"[ AI ] : {response}")
    
if __name__ == "__main__":
    main()