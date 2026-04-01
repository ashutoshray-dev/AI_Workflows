from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
load_dotenv()
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition, ToolNode, tool_node
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_tavily import TavilySearch
import os
os.environ["LANGCHAIN_PROJECT"] = "Chatbot-graph"
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

search = DuckDuckGoSearchRun()
# search = TavilySearch()
def calculator(first_num: float, second_num: float, operation:str)-> dict:
    """Perform a basic arithmetic operation on two numbers. Supported operations:add, sub, mul, div."""
    try:
        if operation == "add" or operation=="+":
            result = first_num+second_num
        elif operation == "sub" or operation=="-":
            result = first_num-second_num
        elif operation == "mul" or operation=="*":
            result = first_num*second_num
        elif operation == "div" or operation=="/":
            result = first_num/second_num
        else:
            return {"error":f"Unsupported operation `{operation}`"}
        
        return {"first_num":first_num, "second_name":second_num, "operation":operation, "result":result}
    except Exception as e:
        return {"error": str(e)}
tools = [search, calculator]
model = ChatOllama(model='qwen2.5:3b')
model_with_tools = model.bind_tools(tools)
def chat_node(state:ChatState):
    messages = state['messages']
    # print(messages)
    response = model_with_tools.invoke(messages)
    return {'messages': [response]}

conn = sqlite3.connect(database='langraph_chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)
tool_node = ToolNode(tools)
graph = StateGraph(ChatState)
graph.add_node('chat', chat_node)
graph.add_node('tools', tool_node)
graph.add_edge(START, 'chat')
graph.add_conditional_edges('chat', tools_condition)
graph.add_edge('tools', 'chat')
# graph.add_edge('chat', END)
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