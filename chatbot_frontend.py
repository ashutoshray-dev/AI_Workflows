import streamlit as st
from chatbot_backend import chatbot, retrieve_threads_list
from langchain_core.messages import HumanMessage, AIMessage
import uuid

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id
def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
def load_history(thread_id):
    return chatbot.get_state(config={'configurable':{'thread_id':thread_id}}).values['messages']

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_threads_list()

add_thread(st.session_state['thread_id'])

st.sidebar.title('Langgraph Chatbot')
if st.sidebar.button('New Chat'):
    reset_chat()
st.sidebar.header('My conversations')
for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_history(thread_id)
        temp_message = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role='user'
            else:
                role='assistant'
            temp_message.append({'role':role, 'content':msg.content})
        st.session_state['message_history'] = temp_message

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

CONFIG = {'configurable':{'thread_id':st.session_state['thread_id']}}
user_input = st.chat_input('Enter here')
if user_input:
    st.session_state['message_history'].append({'role':'user', 'content':user_input})
    with st.chat_message('user'):
        st.text(user_input)

    with st.chat_message('assistant'):
    # older version of streaming          -------> returns a tupple with message_chunk and metadata at 0 and 1 indices respectively
        # ai_message = st.write_stream(
        #     message_chunk.content for message_chunk, metadata in chatbot.stream(
        #         {'messages':[HumanMessage(content=user_input)]},
        #         config=CONFIG,
        #         stream_mode='messages'
        #     )
        # )
    # latest version of streaming      -------> returns a dict with type, ns and data as keys with value of data being the tupple returned in v1
        def ai_message_stream():
            for message_chunk in chatbot.stream(
                {'messages':[HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode='messages',
                version='v2'
            ):
                if isinstance(message_chunk['data'][0], AIMessage):
                    msg = message_chunk['data'][0].content
                    yield msg
            st.session_state['message_history'].append({'role':'assistant', 'content':msg})
        ai_message = st.write_stream(ai_message_stream())

        # st.session_state['message_history'].append({'role':'assistant', 'content':ai_message})
        # with st.chat_message('ai'):
        #     st.text(type(ai_message))