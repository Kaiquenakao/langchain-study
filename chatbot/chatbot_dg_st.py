import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter

# Solicitar e definir a variável de ambiente da chave da OpenAI
api_key = st.text_input("OpenAI API Key", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

# Campo de entrada para os caminhos dos arquivos, separados por vírgula
file_paths_input = st.text_input("Insira os caminhos dos arquivos PDF, separados por vírgula")
file_paths = [path.strip() for path in file_paths_input.split(',') if path.strip()]

# Função para processar documentos e responder perguntas
def process_input(question, file_paths):
    # Configuração do modelo da OpenAI com limite de tokens para a resposta
    model_local = ChatOpenAI(model="gpt-4-turbo-preview", max_tokens=1500)  # A chave da API é obtida da variável de ambiente
    
    all_docs = []
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        docs_list = loader.load_and_split()
        all_docs.extend(docs_list)

    # Dividir o texto em chunks menores
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1500, chunk_overlap=100)  # Ajustar o chunk_size
    doc_splits = text_splitter.split_documents(all_docs)

    # Criar embeddings e armazenar no banco de dados vetorial
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OpenAIEmbeddings(model="text-embedding-ada-002"),
    )
    retriever = vectorstore.as_retriever()

    # Configurar e executar o RAG com chunks menores
    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )

    # Invocar a cadeia RAG com a pergunta
    try:
        response = after_rag_chain.invoke(question)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        response = "An error occurred. Please try again later."

    return response

# Configuração da interface do Streamlit
st.title("Buscar documentos através da IA")

# Campo de entrada para a pergunta
question = st.text_input("Digite sua pergunta:")

# Botão para processar a entrada
if st.button('Query Document'):
    if not api_key:
        st.error("Por favor, insira sua OpenAI API Key.")
    elif not file_paths:
        st.error("Por favor, insira os caminhos dos arquivos PDF.")
    else:
        with st.spinner('Processing...'):
            answer = process_input(question, file_paths)
            st.text_area("Resposta", value=answer, height=300)
