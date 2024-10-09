import openai
import streamlit as st

st.title("친근한 챗봇")

openai.api_key = st.secrets["OPENAI_API_KEY"]

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

system_message = '''
너의 이름은 친구봇이야.
너는 항상 반말을 하는 챗봇이야. 다나까나 요 같은 높임말로 절대로 끝내지 마
항상 반말로 친근하게 대답해줘.
영어로 질문을 받아도 무조건 한글로 답변해줘.
한글이 아닌 답변일 때는 다시 생각해서 꼭 한글로 만들어줘
모든 답변 끝에 답변에 맞는 이모티콘도 추가해줘
'''

if "messages" not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) == 0:
    st.session_state.messages = [{"role": "system", "content": system_message}]

for idx, message in enumerate(st.session_state.messages):
    if idx > 0:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # ChatCompletion.create 대신 Completion.create를 사용할 수 있음
        response = openai.Completion.create(
            model=st.session_state["openai_model"],
            prompt=prompt,
            max_tokens=100,
            stream=True,
        )

        full_response = ""
        for chunk in response:
            chunk_message = chunk["choices"][0].get("text", "")
            full_response += chunk_message
            st.write(chunk_message)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
