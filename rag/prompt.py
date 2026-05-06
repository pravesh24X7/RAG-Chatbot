from langchain_core.prompts import PromptTemplate

from app.config import PROMPT_PATH


def generate_prompt(name: str):
    template = """
        You are a helpful assistant, Provide answers to user query. strictly on the basis of given context. If context is insufficient says `In-Sufficient Context provided`.
        \n\n
        Context: {context}
        \n\n
        Query: {query}
    """

    prompt = PromptTemplate(template=template,
                            validate_template=True,
                            input_variables=['context', 'query'])
    prompt.to_json(f"{PROMPT_PATH}{name}")