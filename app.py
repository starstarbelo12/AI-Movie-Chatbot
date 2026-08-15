import time
import streamlit as st
from src.chatbot import chatbot_response
from src.chatbot.paths import STYLE_CSS_PATH

# Page configuration
st.set_page_config(
    page_title="AI Movie Chatbot",
    page_icon="🎬",
    layout="centered",
)

# Load external CSS file
def load_css(css_path=STYLE_CSS_PATH):
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file not found at {css_path}")

load_css()

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "MLP"

if "show_debug" not in st.session_state:
    st.session_state.show_debug = False

# Sidebar controls
with st.sidebar:
    st.title("⚙️ Settings")
    st.session_state.show_debug = st.checkbox(
        "🐞 Show Debug Info",
        value=st.session_state.show_debug,
        help="Displays inference latency and active model under responses."
    )

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Header
st.title("🎬 AI Movie Chatbot")
st.write(
    "Ask about a movie's genre, rating, runtime, budget, "
    "revenue, language, release date, collection, or plot summary!"
)

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and st.session_state.show_debug and message.get("debug_info"):
            st.caption(message["debug_info"])

# Map UI label to algorithm key
algorithm = {
    "MLP": "mlp",
    "NB": "nb",
}[st.session_state.selected_model]

# Sticky bottom input container
with st.bottom:
    col_input, col_model = st.columns([0.91, 0.09], vertical_alignment="center")

    with col_input:
        user_message = st.chat_input("Ask about a movie...")

    with col_model:
        with st.popover(st.session_state.selected_model, use_container_width=False):
            st.markdown("### Select AI Model")
            if st.button("Multilayer Perceptron (MLP)", use_container_width=True):
                st.session_state.selected_model = "MLP"
                st.rerun()
            if st.button("Naive Bayes (NB)", use_container_width=True):
                st.session_state.selected_model = "NB"
                st.rerun()

# Process user input
if user_message:
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    start_time = time.perf_counter()
    debug_str = ""

    try:
        response = chatbot_response(user_message, algorithm=algorithm)
        latency_ms = (time.perf_counter() - start_time) * 1000
        debug_str = f"⚙️ **Model:** {st.session_state.selected_model} | ⏱️ **Latency:** {latency_ms:.2f} ms"
    except Exception as exc:
        response = f"⚠️ The chatbot encountered an error: {exc}"

    with st.chat_message("assistant"):
        st.markdown(response)
        if st.session_state.show_debug and debug_str:
            st.caption(debug_str)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "debug_info": debug_str,
    })