from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import load_prompt
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from app.config import PROMPT_PATH, CHATMODEL
from utils.format_docs import format_docs


def execute(retriever, name: str):
    llm_model = ChatGroq(model=CHATMODEL,
                         temperature=0.5,
                         model_kwargs={},
                         streaming=True)
    
    prompt = load_prompt(f"{PROMPT_PATH}{name}")
    parser = StrOutputParser()

    chain = ({
        "context": retriever | RunnableLambda(format_docs),
        "query": RunnableLambda(lambda x: x["query"])
    }) | prompt | llm_model | parser

    return chain