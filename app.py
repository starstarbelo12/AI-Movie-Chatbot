import streamlit as st
from chatbot import chatbot_response


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Movie Chatbot",
    page_icon="🎬",
    layout="centered",
)


# ============================================================
# TITLE
# ============================================================

st.title("🎬 AI Movie Chatbot")
st.write(
    "Ask about a movie's genre, rating, runtime, budget, "
    "revenue, language, release date, collection, or summary."
)


# ============================================================
# MODEL SELECTION
# ============================================================

selected_model = st.selectbox(
    "Select AI Model",
    ["MLP", "Naive Bayes"],
)


algorithm = {
    "MLP": "mlp",
    "Naive Bayes": "nb",
}[selected_model]


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# USER INPUT
# ============================================================

user_message = st.chat_input(
    "Ask about a movie..."
)


if user_message:

    # Display user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_message)

    # Generate chatbot response
    try:
        response = chatbot_response(
            user_message,
            algorithm=algorithm,
        )

    except Exception as exc:
        response = (
            "⚠️ The chatbot encountered an error while processing "
            f"your request: {exc}"
        )

    # Display chatbot response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
        }
    )

    with st.chat_message("assistant"):
        st.markdown(response)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.subheader("Current Model")
    st.write(selected_model)

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
