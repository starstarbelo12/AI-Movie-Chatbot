import streamlit as st
from src.chatbot import chatbot_response


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
    st.session_state.selected_model = "Hybrid"


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


# ============================================================
# MODEL SELECTION + CHAT INPUT
# ============================================================
#
# The model selector button is layered directly into the
# bottom-right corner of the chat input box itself (like the
# model picker in Claude/ChatGPT), instead of sitting beside
# it in its own column.
#
# This is done by putting both widgets inside one keyed
# container, then using CSS to absolutely-position the
# popover on top of the chat_input's bottom-right corner.
#
# The selected model is then passed to chatbot_response().
# ============================================================

st.markdown(
    """
    <style>
    /* Container that holds the popover + chat_input together */
    .st-key-chat_input_row {
        position: relative;
    }

    /* Pull the model popover out of normal flow and pin it
       inside the chat input's bottom-right corner */
    .st-key-chat_input_row [data-testid="stPopover"] {
        position: absolute;
        right: 52px;
        bottom: 10px;
        width: auto !important;
        z-index: 999;
    }

    /* Make the popover trigger look like a small pill button
       rather than a full-width Streamlit button */
    .st-key-chat_input_row [data-testid="stPopover"] > div > button {
        border-radius: 999px;
        padding: 4px 12px;
        font-size: 0.8rem;
        min-height: 2rem;
        border: 1px solid rgba(49, 51, 63, 0.2);
    }

    /* Leave room in the textarea so typed text never runs
       underneath the overlaid button */
    .st-key-chat_input_row [data-testid="stChatInput"] textarea {
        padding-right: 140px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container(key="chat_input_row"):

    with st.popover(f"🤖 {st.session_state.selected_model}"):

        st.markdown("### Select AI Model")

        hybrid_selected = st.button(
            "🧠 Hybrid (MLP+KNN)",
            use_container_width=True,
        )

        nb_selected = st.button(
            "📊 Naive Bayes",
            use_container_width=True,
        )

        if hybrid_selected:
            st.session_state.selected_model = "Hybrid"
            st.rerun()

        if nb_selected:
            st.session_state.selected_model = "Naive Bayes"
            st.rerun()

    user_message = st.chat_input(
        "Ask about a movie..."
    )


# Convert displayed model name to the value expected
# by chatbot_response().
algorithm = {
    "Hybrid": "hybrid",
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
    # Generate response
    # -------------------------
    try:

        response = chatbot_response(
            user_message,
            algorithm=algorithm,
        )

    except Exception as exc:

        response = (
            "⚠️ The chatbot encountered an error while "
            f"processing your request: {exc}"
        )

    # -------------------------
    # Add assistant response
    # -------------------------
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

    st.subheader("AI Model")

    st.write(
        f"Currently selected: "
        f"**{st.session_state.selected_model}**"
    )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()
