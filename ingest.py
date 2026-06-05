import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def run_ingestion():
    if not os.path.exists("data"):
        os.makedirs("data")
        return

    loader = DirectoryLoader("data", glob="*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("faiss_index")

if __name__ == "__main__":
    run_ingestion()