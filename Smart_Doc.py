from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnablePassthrough
import streamlit as st
import tempfile
from langsmith import traceable
from dotenv import load_dotenv
load_dotenv()
import os
os.environ["LANGCHAIN_PROJECT"] = "Smart_Doc-ragQuery"

st.title('Smart_Doc System')
st.subheader('A smart workflow that takes file uploads and answers questions based on them')
uploaded_file = st.file_uploader(
    "Upload file", accept_multiple_files=False, type="pdf"
)
if uploaded_file:
    # 1. Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    @traceable(name='load_file')
    def load_file(path:str):
        document = PyPDFLoader(tmp_path)
        docs = document.load()
        return docs
    @traceable(name='split_file')
    def split_file(docs, chunk_size:int, chunk_overlap:int):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_docs = text_splitter.split_documents(docs)
        return split_docs
    @traceable(name='build_store')
    def build_store(split_docs):
        embedding_func = OllamaEmbeddings(model='embeddinggemma')
        embed_docs = FAISS.from_documents(split_docs, embedding_func)
        return embed_docs
    @traceable(name='setup_pipeline')
    def setup_pipeline(path:str, chunk_size:int, chunk_overlap:int):
        docs = load_file(path=path)
        split_docs = split_file(docs=docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        embed_docs = build_store(split_docs=split_docs)
        return embed_docs
    query = st.text_input("Enter your query")
    @traceable(name='setup_and_query')
    def setup_pipeline_and_query(path:str, query:str):
        prompt = ChatPromptTemplate.from_template(
            template = """You are a helpful assistant. Answer the questions based on the context provided only. Every correct answer will be rewarded
            with 1000 points and every wrong answer will be penalised. Don't try to come up with random answers, only answer the queries 
            from the provided context. Strictly follow this rule: If you don't find the answer to a question in the context, don't generate your own answers, just explain in a polite and friendly tone with your own generated text that 'you don't have the context for it so you don't know'.
            <context>
            {context}
            </context>
            question: {input}"""
        )
        model = ChatOllama(model='qwen2.5:3b')
        # retriever = embed_docs.as_retriever()
        vector_store = setup_pipeline(path, 1000, 200)
        retriever = vector_store.as_retriever()
        parser = StrOutputParser()
        def format_docs(docs):
            return "\n".join(doc.page_content for doc in docs)
        chain = (
            {'context': retriever|format_docs, 'input': RunnablePassthrough()} | prompt | model | parser
        )
        final_response = chain.invoke(query)
        return final_response
    if query:
        final_response = setup_pipeline_and_query(path=tmp_path, query=query)
        st.write(final_response)