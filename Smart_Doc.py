from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnablePassthrough
import streamlit as st
import tempfile
from dotenv import load_dotenv
load_dotenv()

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
    # document = PyPDFLoader('ml_paper1.pdf')
    document = PyPDFLoader(tmp_path)
    content = document.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(content)
    embedding_func = OllamaEmbeddings(model='embeddinggemma')
    embed_docs = FAISS.from_documents(split_docs, embedding_func)
    # query = input('Enter your query: ')
    query = st.text_input("Enter your query")
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
    retriever = embed_docs.as_retriever()
    parser = StrOutputParser()
    def format_docs(docs):
        return "\n".join(doc.page_content for doc in docs)
    chain = (
        {'context': retriever|format_docs, 'input': RunnablePassthrough()} | prompt | model | parser
    )
    # final_response = chain.invoke(query)
    # print(final_response)
    if query:
        st.write(chain.invoke(query))