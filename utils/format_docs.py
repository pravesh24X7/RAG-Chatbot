def format_docs(docs):

    print("[*] Retrieved docs ...", docs)
    return "\n\n".join(
        doc.page_content for doc in docs
    )