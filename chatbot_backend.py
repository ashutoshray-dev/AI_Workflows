from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
load_dotenv()
import operator
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

model = ChatOllama(model='qwen2.5:3b')
def chat_node(state:ChatState):
    messages = state['messages']
    # print(messages)
    response = model.invoke(messages)
    return {'messages': [response]}
checkpointer = MemorySaver()
graph = StateGraph(ChatState)
graph.add_node('chat', chat_node)
graph.add_edge(START, 'chat')
graph.add_edge('chat', END)
chatbot = graph.compile(checkpointer=checkpointer)
thread_id = 1
while True:
    user_message = input('enter here: ')
    print('User: ', user_message)
    if user_message.strip().lower() in ['exit', 'quit', 'bye']:
        break
    config = {'configurable':{'thread_id':thread_id}}
    response = chatbot.invoke({'messages': [HumanMessage(content=user_message)]}, config=config)
    print('AI: ', response['messages'][-1].content)