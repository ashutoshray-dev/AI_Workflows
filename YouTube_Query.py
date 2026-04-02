from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import YoutubeLoader
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# video_id = 'VJgdOMXhEj0'
# try:
#     transcript_list = YouTubeTranscriptApi().fetch(video_id=video_id, languages=['en', 'hi'])
#     transcript = " ".join(chunk.text for chunk in transcript_list)
#     print(transcript_list)
# except TranscriptsDisabled:
#     print('No transcript available for this video')

st.title('YouTube Query')
st.subheader('Get answers from a YouTube video')
yt_url = st.text_input("Enter the YouTube video URL")
# yt_url = input("Enter url of the YouTube video: ")
# transcript = None
if yt_url:
    try:
        loader = YoutubeLoader.from_youtube_url(
        yt_url, add_video_info = False, language = ['en', 'hi'],
        )
        transcript = loader.load()
        # print(transcript)
    except Exception as e:
        print(f"Error encountered: {e}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    # result = splitter.create_documents([transcript])
    split_docs = splitter.split_documents(transcript)
    embedding_funtn = OllamaEmbeddings(model='embeddinggemma')
    db = FAISS.from_documents(split_docs, embedding=embedding_funtn)
    # query = "What is f1?"
    # query = input("Enter your query: ")
    query = st.text_input("Enter Your Query")
    # search_res = db.similarity_search(query, k=5)
    prompt = ChatPromptTemplate.from_template(
        template = """You are a helpful assistant. Answer the questions based on the context provided only. Every correct answer will be rewarded
        with 1000 points and every wrong answer will be penalised. Don't try to come up with random answers, only answer the queries 
        from the provided context. If you don't find the answer to a question, just say you don't know.
        <context>
        {context}
        </context>
        question: {input}""",
        # input_variables=['res', 'query']
    )
    # resp = prompt.invoke({'context':res, 'query':query})
    model = ChatOllama(model='qwen2.5:3b')
    retriever = db.as_retriever()
    parser = StrOutputParser()
    def format_docs(docs):
        return "\n".join(doc.page_content for doc in docs)

    chain = (
        {'context': retriever|format_docs, 'input': RunnablePassthrough()} | prompt | model | parser
    )
    if query:
        # final_response = chain.invoke(query)
        # print(final_response)
        st.write(chain.invoke(query))

