from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
load_dotenv()
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langgraph.graph.message import add_messages
import os
os.environ.get("LANGCHAIN_PROJECT") = "Chatbot-graph"
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

model = ChatOllama(model='qwen2.5:3b')
def chat_node(state:ChatState):
    messages = state['messages']
    # print(messages)
    response = model.invoke(messages)
    return {'messages': [response]}

conn = sqlite3.connect(database='langraph_chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)
graph = StateGraph(ChatState)
graph.add_node('chat', chat_node)
graph.add_edge(START, 'chat')
graph.add_edge('chat', END)
chatbot = graph.compile(checkpointer=checkpointer)
# thread_id = 1
# while True:
#     user_message = input('enter here: ')
#     print('User: ', user_message)
#     if user_message.strip().lower() in ['exit', 'quit', 'bye']:
#         break
#     config = {'configurable':{'thread_id':thread_id}}
#     response = chatbot.invoke({'messages': [HumanMessage(content=user_message)]}, config=config)
#     print('AI: ', response['messages'][-1].content)

def retrieve_threads_list():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)