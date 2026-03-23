import streamlit as st
from chatbot_backend import chatbot
from langchain_core.messages import HumanMessage

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

CONFIG = {'configurable':{'thread_id':'thread-1'}}
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
        ai_message = st.write_stream(
            message_chunk['data'][0].content for message_chunk in chatbot.stream(
                {'messages':[HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode='messages',
                version='v2'
            )
        )
        
    st.session_state['message_history'].append({'role':'assistant', 'content':ai_message})