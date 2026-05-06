from langchain_text_splitters import RecursiveCharacterTextSplitter

from .document_loader import load_file

def get_all_documents(filepath: str):
    pdf_pages = []

    # load all pages in form of document object
    for page in load_file(filepath=filepath):
        pdf_pages.append(page)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                              chunk_overlap=100,
                                              separators=["\n\n", "\n", ".", " "]
                                              )
    
    documents = splitter.split_documents(pdf_pages)
    return documents