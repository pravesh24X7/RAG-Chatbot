from langchain_community.document_loaders import PyPDFLoader


def load_file(filepath: str=""):
    if not filepath:
        raise ValueError("File not given")
    
    loader = PyPDFLoader(file_path=filepath)
    return loader.lazy_load()