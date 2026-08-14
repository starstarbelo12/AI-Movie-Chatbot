import time
import streamlit as st
from chatbot import chatbot_response


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="AI Movie Chatbot",
    page_icon="🎬",
    layout="centered",
)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "MLP"

if "show_debug" not in st.session_state:
    st.session_state.show_debug = False


# ============================================================
# SIDEBAR (MODEL SELECTION & CONTROLS)
# ============================================================
with st.sidebar:
    st.title("⚙️ Engine & Settings")

    # Model Switcher
    st.session_state.selected_model = st.radio(
        "Select AI Model:",
        ["MLP", "Naive Bayes"],
        index=0 if st.session_state.selected_model == "MLP" else 1,
        help="Select the ML engine used for query intent classification."
    )

    st.divider()

    # Debug Mode Toggle
    st.session_state.show_debug = st.checkbox(
        "🐞 Show Debug Info",
        value=st.session_state.show_debug,
        help="Displays active model and inference latency (in ms) under responses."
    )

    st.divider()

    # Clear Conversation Button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# MAIN INTERFACE & TITLE
# ============================================================
st.title("🎬 AI Movie Chatbot")
st.write(
    "Ask about a movie's genre, rating, runtime, budget, "
    "revenue, language, release date, collection, or plot summary!"
)


# ============================================================
# CHAT HISTORY
# ============================================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and st.session_state.show_debug and message.get("debug_info"):
            st.caption(message["debug_info"])


# ============================================================
# CHAT INPUT (ROOT SCOPE -> NATIVE STICKY BOTTOM ANCHOR)
# ============================================================
# Map UI selection to the string expected by chatbot_response()
algorithm = {
    "MLP": "mlp",
    "Naive Bayes": "nb",
}[st.session_state.selected_model]


if user_message := st.chat_input("Ask about a movie..."):

    # --- 1. Render & Append User Message ---
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    # --- 2. Generate AI Response & Track Latency ---
    start_time = time.perf_counter()
    debug_str = ""

    try:
        response = chatbot_response(user_message, algorithm=algorithm)
        latency_ms = (time.perf_counter() - start_time) * 1000
        debug_str = f"⚙️ **Model:** {st.session_state.selected_model} | ⏱️ **Latency:** {latency_ms:.2f} ms"
    except Exception as exc:
        response = f"⚠️ The chatbot encountered an error while processing your request: {exc}"

    # --- 3. Render & Append Assistant Message ---
    with st.chat_message("assistant"):
        st.markdown(response)
        if st.session_state.show_debug and debug_str:
            st.caption(debug_str)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "debug_info": debug_str,
    })

    # NO st.rerun() needed — Streamlit renders the new message smoothly inline!