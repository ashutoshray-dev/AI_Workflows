from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
load_dotenv()
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
# from langgraph.checkpoint.memory import InMemorySaver
import sqlite3
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition, ToolNode, tool_node
from langchain_core.tools import tool, BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_tavily import TavilySearch
from langchain_core.runnables import RunnableConfig
from typing import TypedDict, Annotated
import requests
import os
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
import threading
import aiosqlite
os.environ["LANGCHAIN_PROJECT"] = "Chatbot-graph"
stock_api=os.getenv('ALPHAVANTAGE_API_KEY')

_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()

def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)
def run_async(coro):
    return _submit_async(coro).result()
def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)

client = MultiServerMCPClient(
    {
        # "ddg-search": {
        #     "transport": "stdio",
        #     "command": "duckduckgo-mcp-server",
        #     "args": []
        # },
        "expense": {
            "transport": "streamable_http",
            "url": "https://splendid-gold-dingo.fastmcp.app/mcp"
        }
    }
)
def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception:
        return []

mcp_tools = load_mcp_tools()
# search = DuckDuckGoSearchRun()
# search = TavilySearch()
@tool
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

@tool
def get_stock_price(symbol:str)->dict:
    """fetch latest stock price for a given symbol (e.g. AAPL, TSLA).
    Use Alpha Vantage api with key in the url."""
    URL = (
        f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={stock_api}"
        )
    response = requests.get(url=URL)
    return response.json()
tools = [calculator, get_stock_price, *mcp_tools]
model = ChatOllama(model='qwen2.5:3b')
model_with_tools = model.bind_tools(tools) if tools else model
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    chat_title: str

class chattitle(TypedDict):
    chat_title: Annotated[str, "A brief 4-5word title that captures the essence of the input."]


async def chat_node(state:ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state['messages']
    # print(messages)
    response = await model_with_tools.ainvoke(messages)
    if not state.get('chat_title'):
        title = await generate_title(messages[0].content)
        return {'messages': [response], 'chat_title':title}
    return {'messages':[response]}
    
tool_node = ToolNode(tools) if tools else None

async def _init_checkpointer():
    conn = await aiosqlite.connect(database='langgraph_chatbot.db')
    return AsyncSqliteSaver(conn)
checkpointer = run_async(_init_checkpointer())
graph = StateGraph(ChatState)
graph.add_node('chat', chat_node)
# graph.add_node('tools', tool_node)
graph.add_edge(START, 'chat')
if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat", tools_condition)
    graph.add_edge("tools", "chat")
else:
    graph.add_edge("chat", END)
chatbot = graph.compile(checkpointer=checkpointer)

async def _alist_threads():
    seen_threads = set()
    list_threads = list()
    async for checkpoint in checkpointer.alist(None):
        thread = checkpoint.config["configurable"]["thread_id"]
        if thread not in seen_threads:
            seen_threads.add(thread)
            list_threads.insert(0, thread)
    return list_threads

def retrieve_threads_list():
    return run_async(_alist_threads())

async def generate_title(user_input):
    structured_model = model.with_structured_output(chattitle)
    title = await structured_model.ainvoke(f'summarise this message input and generate a suitable 4-5words title for this input that feels appropriate for the topic. If no input is present genetrate a random string.\ninput:{user_input}')
    return title['chat_title']
