from langchain_core.prompts import PromptTemplate

from app.config import PROMPT_PATH


def generate_prompt(name: str):
    template = """
        Instructions:
            1. Use only the given context to generate your answer.
            2. Do not use prior knowledge or make assumptions beyond the context.
            3. If the answer is not explicitly or implicitly present in the context, say:
                "I don’t have enough information to answer that."
            4. Keep the answer clear, concise, and relevant to the question.
            5. If applicable, cite or reference the exact part of the context that supports your answer.
        \n\n
        Context: {context}
        \n\n
        Question: {query}
    """

    prompt = PromptTemplate(template=template,
                            validate_template=True,
                            input_variables=['context', 'query'])
    prompt.save(f"{PROMPT_PATH}{name}")