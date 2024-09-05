import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


os.environ["OPENAI_API_KEY"]
os.environ["PINECONE_API_KEY"] 

def ingest_docs() -> None:
    loader = PyPDFLoader("biografia.pdf")
    documents = loader.load()
    print(documents)
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)


    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    PineconeVectorStore.from_documents(docs, embeddings, index_name="chatbot")
    print("-> Added to Pinecone vectorstore vectors")


if __name__ == "__main__":
    ingest_docs()