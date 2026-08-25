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
    st.session_state.selected_model = "Hybrid"

if "show_debug" not in st.session_state:
    st.session_state.show_debug = False

# Sidebar controls
with st.sidebar:
    st.title("⚙️ Settings")
    st.session_state.show_debug = st.checkbox(
        "🐞 Show Debug Info",
        value=st.session_state.show_debug,
        help="Displays latency, active model, predicted intent, and confidence scores under responses."
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
        
        # Render stored debug UI elements
        if message["role"] == "assistant" and st.session_state.show_debug and message.get("debug_data"):
            debug = message["debug_data"]
            
            # Basic info caption line
            st.caption(f"⚙️ **Model:** {debug.get('model', 'N/A')} | ⏱️ **Latency:** {debug.get('latency', 0):.2f} ms | 🎯 **Predicted Intent:** `{debug.get('intent', 'N/A')}`")
            
            # Interactive visual breakdown of prediction values/scores
            scores = debug.get("scores", {})
            if scores and isinstance(scores, dict):
                with st.expander("📊 Prediction Scores Breakdown", expanded=False):
                    for intent_name, score_val in scores.items():
                        col_label, col_bar = st.columns([0.3, 0.7])
                        col_label.text(f"{intent_name}: {score_val:.2%}" if isinstance(score_val, float) else f"{intent_name}: {score_val}")
                        if isinstance(score_val, (int, float)):
                            col_bar.progress(min(max(float(score_val), 0.0), 1.0))

            if debug.get("movie_match"):
                st.caption(
                    f"🎬 **Movie Match:** {debug['movie_match']['title']} | "
                    f"Method: {debug['movie_match']['method']} | "
                    f"Match Score: {debug['movie_match']['match_score']:.2%}"
                )
                candidates = debug["movie_match"].get("candidates", [])
                if len(candidates) > 1:
                    with st.expander("🎬 Movie Match Candidates", expanded=False):
                        st.markdown("\n".join(
                            f"{position}. **{candidate['title']}** — "
                            f"{candidate['confidence']:.2%}"
                            for position, candidate in enumerate(candidates, start=1)
                        ))

# Map UI label to algorithm key
algorithm = {
    "Hybrid": "hybrid",
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
            if st.button("🧠 Hybrid Model (MLP + KNN)", use_container_width=True):
                st.session_state.selected_model = "Hybrid"
                st.rerun()
            if st.button("📊 Naive Bayes (NB)", use_container_width=True):
                st.session_state.selected_model = "NB"
                st.rerun()

# Process user input
if user_message:
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    start_time = time.perf_counter()
    
    intent = "N/A"
    scores_dict = {}
    movie_match = None

    try:
        res = chatbot_response(user_message, algorithm=algorithm)
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Parse return values depending on backend implementation
        if isinstance(res, dict):
            response = res.get("response", "")
            intent = res.get("intent", res.get("prediction", "N/A"))
            scores_dict = res.get("scores", res.get("probabilities", {}))
            movie_match = res.get("movie_match")
        elif isinstance(res, tuple):
            response = res[0]
            intent = res[1] if len(res) > 1 else "N/A"
            scores_dict = res[2] if len(res) > 2 and isinstance(res[2], dict) else {}
        else:
            response = str(res)

    except Exception as exc:
        response = f"⚠️ The chatbot encountered an error: {exc}"
        latency_ms = (time.perf_counter() - start_time) * 1000

    # Package debug metadata object
    debug_data = {
        "model": st.session_state.selected_model,
        "latency": latency_ms,
        "intent": intent,
        "scores": scores_dict,
        "movie_match": movie_match,
    }

    # Render assistant response & Debug UI elements
    with st.chat_message("assistant"):
        st.markdown(response)
        
        if st.session_state.show_debug:
            st.caption(f"⚙️ **Model:** {debug_data['model']} | ⏱️ **Latency:** {debug_data['latency']:.2f} ms | 🎯 **Predicted Intent:** `{debug_data['intent']}`")
            
            if debug_data["scores"] and isinstance(debug_data["scores"], dict):
                with st.expander("📊 Prediction Scores Breakdown", expanded=True):
                    for intent_name, score_val in debug_data["scores"].items():
                        col_label, col_bar = st.columns([0.3, 0.7])
                        col_label.text(f"{intent_name}: {score_val:.2%}" if isinstance(score_val, float) else f"{intent_name}: {score_val}")
                        if isinstance(score_val, (int, float)):
                            col_bar.progress(min(max(float(score_val), 0.0), 1.0))

            if debug_data.get("movie_match"):
                st.caption(
                    f"🎬 **Movie Match:** "
                    f"{debug_data['movie_match']['title']} | "
                    f"Method: {debug_data['movie_match']['method']} | "
                    f"Match Score: "
                    f"{debug_data['movie_match']['match_score']:.2%}"
                )
                candidates = debug_data["movie_match"].get("candidates", [])
                if len(candidates) > 1:
                    with st.expander("🎬 Movie Match Candidates", expanded=True):
                        st.markdown("\n".join(
                            f"{position}. **{candidate['title']}** — "
                            f"{candidate['confidence']:.2%}"
                            for position, candidate in enumerate(candidates, start=1)
                        ))

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "debug_data": debug_data,
    })
    st.rerun()