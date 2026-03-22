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
    
    response = chatbot.invoke({'messages':[HumanMessage(content=user_input)]}, config=CONFIG)
    st.session_state['message_history'].append({'role':'assistant', 'content':response['messages'][-1].content})
    with st.chat_message('assistant'):
        # st.text(st.session_state['message_history'][-1]['content'])
        st.text(response['messages'][-1].content)