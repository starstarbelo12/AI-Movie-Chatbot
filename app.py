import time
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
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "MLP"

if "show_debug" not in st.session_state:
    st.session_state.show_debug = False


# ============================================================
# TITLE
# ============================================================

st.title("🎬 AI Movie Chatbot")

st.write(
    "Ask about a movie's genre, rating, runtime, budget, "
    "revenue, language, release date, collection, or summary."
)


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display stored debug info if Debug Mode is enabled
        if "debug_info" in message and st.session_state.show_debug and message["debug_info"]:
            st.caption(message["debug_info"])


# ============================================================
# MODEL SELECTION + CHAT INPUT
# ============================================================

model_col, chat_col = st.columns(
    [0.22, 0.78],
    vertical_alignment="bottom",
)


with model_col:

    with st.popover(
        f"🤖 {st.session_state.selected_model}",
        use_container_width=True,
    ):

        st.markdown("### Select AI Model")

        mlp_selected = st.button(
            "🧠 MLP",
            use_container_width=True,
        )

        nb_selected = st.button(
            "📊 Naive Bayes",
            use_container_width=True,
        )

        if mlp_selected:
            st.session_state.selected_model = "MLP"
            st.rerun()

        if nb_selected:
            st.session_state.selected_model = "Naive Bayes"
            st.rerun()


with chat_col:

    user_message = st.chat_input(
        "Ask about a movie..."
    )


# Convert displayed model name to the value expected by chatbot_response().
algorithm = {
    "MLP": "mlp",
    "Naive Bayes": "nb",
}[st.session_state.selected_model]


# ============================================================
# HANDLE USER MESSAGE
# ============================================================

if user_message:

    # -------------------------
    # Add user message
    # -------------------------
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_message)

    # -------------------------
    # Generate response & record latency
    # -------------------------
    start_time = time.perf_counter()
    debug_str = ""

    try:

        response = chatbot_response(
            user_message,
            algorithm=algorithm,
        )

        # Calculate inference time in milliseconds
        latency_ms = (time.perf_counter() - start_time) * 1000
        debug_str = f"⚙️ **Model:** {st.session_state.selected_model} | ⏱️ **Latency:** {latency_ms:.2f} ms"

    except Exception as exc:

        response = (
            "⚠️ The chatbot encountered an error while "
            f"processing your request: {exc}"
        )

    # -------------------------
    # Add assistant response
    # -------------------------
    assistant_msg = {
        "role": "assistant",
        "content": response,
        "debug_info": debug_str,
    }
    
    st.session_state.messages.append(assistant_msg)

    with st.chat_message("assistant"):
        st.markdown(response)
        if st.session_state.show_debug and debug_str:
            st.caption(debug_str)

    st.rerun()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.subheader("⚙️ Settings & Model Info")

    st.write(
        f"Currently selected: "
        f"**{st.session_state.selected_model}**"
    )

    st.divider()

    # Debug Mode Toggle Switch
    st.session_state.show_debug = st.checkbox(
        "🐞 Show Debug Info",
        value=st.session_state.show_debug,
        help="Displays inference latency and active model under each response.",
    )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()