from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

def create_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embedding=embeddings)
    return vectorstore