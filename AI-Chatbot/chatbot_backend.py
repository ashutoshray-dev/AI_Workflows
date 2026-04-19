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
from langchain_core.runnables import RunnableConfig
from typing import TypedDict, Annotated
import requests
import os
os.environ["LANGCHAIN_PROJECT"] = "Chatbot-graph"
stock_api=os.getenv('ALPHAVANTAGE_API_KEY')
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    chat_title: str

class chattitle(TypedDict):
    chat_title: Annotated[str, "A brief 4-5word title that captures the essence of the input."]
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
        
        return {"first_num":first_num, "second_num":second_num, "operation":operation, "result":result}
    except Exception as e:
        return {"error": str(e)}

def get_stock_price(symbol:str)->dict:
    """fetch latest stock price for a given symbol (e.g. AAPL, TSLA).
    Use Alpha Vantage api with key in the url."""
    URL = (
        f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={stock_api}"
        )
    response = requests.get(url=URL)
    return response.json()
tools = [search, calculator, get_stock_price]
model = ChatOllama(model='qwen2.5:3b')
model_with_tools = model.bind_tools(tools)
def chat_node(state:ChatState, config: RunnableConfig):
    messages = state['messages']
    # print(messages)
    response = model_with_tools.invoke(messages)
    if not state.get('chat_title'):
        title = generate_title(messages[0].content)
        return {'messages': [response], 'chat_title':title}
    return {'messages':[response]}

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
    seen_threads = set()
    all_threads = list()
    for checkpoint in checkpointer.list(None):
        thread = checkpoint.config['configurable']['thread_id']
        if thread not in seen_threads:
            seen_threads.add(thread)
            all_threads.insert(0, thread)
    return all_threads
def generate_title(user_input):
    structured_model = model.with_structured_output(chattitle)
    title = structured_model.invoke(f'summarise this message input and generate a suitable 4-5words title for this input that feels appropriate for the topic. If no input is present genetrate a random string.\ninput:{user_input}')
    return title['chat_title']
