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
# The model selector is placed immediately beside the chat
# input, similar to the model selector in Claude.
#
# Click the model button to choose:
#   - MLP
#   - Naive Bayes
#
# The selected model is then passed to chatbot_response().
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


# Convert displayed model name to the value expected
# by chatbot_response().
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
