import streamlit as st

from langserve import RemoteRunnable

# 设置页面标题
st.title("Chatbot App")

# 初始化会话状态以存储对话历史
if 'messages' not in st.session_state:
    st.session_state.messages = []

# 显示对话历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 创建用户输入框
if prompt := st.chat_input("你:"):
    # 将用户消息添加到对话历史
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)
    client = RemoteRunnable('http://127.0.0.1:8091/chain/')
    #print(client.invoke({'language': 'italian', 'text': '你好！'}))
    response = client.invoke({'input': prompt})
    # 生成机器人的响应（这里使用简单的示例逻辑）
    #response = f"机器人: 嗨，你刚才说 {prompt}！"
    result = response['answer']
    st.session_state.messages.append({"role": "assistant", "content": result})
    st.chat_message("assistant").markdown(result)
