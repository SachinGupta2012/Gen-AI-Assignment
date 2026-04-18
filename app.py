import streamlit as st
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.crawler import WebsiteCrawler
from src.embeddings import EmbeddingsManager
from src.chatbot import WebsiteChatbot

# Page config
st.set_page_config(
    page_title="Website/Information Extraction RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🔐 Load API key securely (NO UI INPUT)
groq_key = st.secrets.get("GROQ_API_KEY", None)

if not groq_key:
    st.error("❌ API key not configured. Please add GROQ_API_KEY in Streamlit secrets.")
    st.stop()

# Initialize session state
defaults = {
    'chatbot': None,
    'vector_store': None,
    'chat_history': [],
    'indexing_done': False,
    'current_url': None,
    'current_title': None,
    'index_path': "./saved_index",
    'crawled_data': None
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

st.title("🤖 Website-Information RAG Chatbot")
st.markdown("Crawl any website, index it with embeddings, and chat with it using **Groq LLM**")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Model Selection
    model = st.selectbox(
        "LLM Model",
        options=[
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma-7b-it"
        ],
        index=0
    )

    st.divider()

    # Crawling Options
    st.subheader("🕷️ Crawling Options")

    use_js = st.toggle(
        "Use JavaScript Rendering",
        value=False,
        help="Enable for dynamic sites (slower)."
    )

    # Text Processing
    st.divider()
    st.subheader("📝 Text Processing")

    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.number_input("Chunk Size", 500, 2000, 1000, step=100)
    with col2:
        chunk_overlap = st.number_input("Overlap", 0, 500, 200, step=50)

    # Session Management
    st.divider()
    st.subheader("💾 Session Management")

    if st.session_state.indexing_done:
        st.success(f"✅ Indexed: {str(st.session_state.current_title)[:30]}...")

        if st.button("🔄 Index New Website", use_container_width=True):
            st.session_state.chatbot = None
            st.session_state.vector_store = None
            st.session_state.chat_history = []
            st.session_state.indexing_done = False
            st.session_state.current_url = None
            st.session_state.current_title = None
            st.rerun()

        if st.button("🗑️ Clear Chat Memory", use_container_width=True):
            st.session_state.chat_history = []
            if st.session_state.chatbot:
                st.session_state.chatbot.clear_memory()
            st.rerun()

    # Save vector DB
    st.divider()
    if st.session_state.indexing_done and st.session_state.vector_store:
        if st.button("💾 Save Index", use_container_width=True):
            try:
                os.makedirs(st.session_state.index_path, exist_ok=True)
                st.session_state.vector_store.save_local(st.session_state.index_path)
                st.success("Index saved! ✅")
            except Exception as e:
                st.error(f"Save failed: {e}")

# Main Area
tab1, tab2 = st.tabs(["🌐 Index Website", "💬 Chat Interface"])

# ------------------ INDEX TAB ------------------
with tab1:
    st.subheader("Step 1: Enter course website / FAQ page")

    url_col, btn_col = st.columns([4, 1])

    with url_col:
        url = st.text_input(
            "Website URL",
            placeholder="https://en.wikipedia.org/wiki/Artificial_intelligence",
            disabled=st.session_state.indexing_done,
            label_visibility="collapsed"
        )

    with btn_col:
        crawl_btn = st.button(
            "🚀 Crawl & Index",
            type="primary",
            disabled=not url or st.session_state.indexing_done,
            use_container_width=True
        )

    # Load existing index
    if not st.session_state.indexing_done:
        st.divider()
        st.subheader("Or Load Existing Index")

        if st.button("📂 Load Saved Index", use_container_width=True):
            try:
                embed_manager = EmbeddingsManager(chunk_size, chunk_overlap)
                vector_store = embed_manager.load_vector_store(st.session_state.index_path)

                st.session_state.chatbot = WebsiteChatbot(
                    vector_store=vector_store,
                    groq_api_key=groq_key,
                    model_name=model
                )

                st.session_state.vector_store = vector_store
                st.session_state.indexing_done = True
                st.session_state.current_url = "Loaded from disk"
                st.session_state.current_title = "Saved Index"

                st.success("Index loaded successfully! ✅")
                st.rerun()

            except Exception as e:
                st.error(f"No saved index found: {e}")

    # Crawl process
    if crawl_btn:
        progress = st.progress(0, text="Starting...")
        status = st.empty()

        try:
            status.info("🕷️ Initializing crawler...")
            progress.progress(10)

            crawler = WebsiteCrawler(use_selenium=use_js)

            status.info("📡 Fetching content...")
            progress.progress(30)

            data = crawler.fetch_content(url)

            if not data['success']:
                st.error(f"❌ Crawling failed: {data['error']}")
                st.stop()

            st.session_state.crawled_data = data

            status.info("✂️ Processing text...")
            progress.progress(50)

            embed_manager = EmbeddingsManager(chunk_size, chunk_overlap)
            vector_store = embed_manager.create_vector_store(data)

            st.session_state.vector_store = vector_store

            status.info("🤖 Initializing chatbot...")
            progress.progress(80)

            st.session_state.chatbot = WebsiteChatbot(
                vector_store=vector_store,
                groq_api_key=groq_key,
                model_name=model
            )

            st.session_state.indexing_done = True
            st.session_state.current_url = url
            st.session_state.current_title = data['title']

            progress.progress(100)
            status.empty()

            st.success(f"✅ Indexed: {data['title']}")

        except Exception as e:
            progress.empty()
            status.empty()
            st.error(f"❌ Error: {str(e)}")

# ------------------ CHAT TAB ------------------
with tab2:
    if not st.session_state.indexing_done:
        st.info("👈 Please index a website first")

    else:
        st.subheader("💬 Chat with Website")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask something..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.chatbot.ask(prompt)
                        st.markdown(response)

                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": response}
                        )

                    except Exception as e:
                        st.error(str(e))

# Footer
st.divider()
st.caption("Built with Streamlit + LangChain + Groq + FAISS")