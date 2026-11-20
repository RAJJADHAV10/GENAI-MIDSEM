"""
app.py - Streamlit Web Interface for Project 9
Modern History & Archival Chronicle Assistant (GenAI Midsem Project)
Grounding Knowledge Source: data/indian_railways_and_telegraph_history_50p.pdf
"""

import json
import streamlit as st
from rag_engine import answer

# Page Configuration
st.set_page_config(
    page_title="Project 9: Modern History & Archival Chronicle Assistant",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .source-badge {
        display: inline-block;
        background-color: #EEF2F6;
        color: #1E293B;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-family: monospace;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #CBD5E1;
        margin-right: 0.4rem;
        margin-top: 0.3rem;
    }
    .not-found-badge {
        display: inline-block;
        background-color: #FEF2F2;
        color: #991B1B;
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        font-family: monospace;
        font-weight: 700;
        border: 1px solid #FCA5A5;
    }
</style>
""", unsafe_allow_html=True)

# Assessment Questions Definition
ASSESSMENT_SAMPLES = [
    {
        "label": "Q1: 1849 Guaranteed Interest Rate",
        "question": "What exact annual guaranteed interest rate, expressed as a percentage, did the colonial administration promise private British railway companies under the 1849 contract terms?",
        "tag": "In-Corpus (Page 13)"
    },
    {
        "label": "Q2: Bhor Ghat Obstacle & Gradient",
        "question": "Which engineering obstacle on the Bhor Ghat incline made the reversing station described in Chapter 3 necessary, and what was the gradient ratio?",
        "tag": "In-Corpus (Page 26)"
    },
    {
        "label": "Q3: Vande Bharat Ticket Price (Negative)",
        "question": "What is the ticket price for an Executive Chair Car seat on the Vande Bharat Express between New Delhi and Varanasi?",
        "tag": "Out-of-Corpus (Negative Test)"
    },
    {
        "label": "Q4: 1851 Diamond Harbour Telegraph",
        "question": "According to the archival records in the text, who supervised the laying of the experimental Diamond Harbour underwater telegraph cable in 1851?",
        "tag": "In-Corpus (Page 43)"
    },
    {
        "label": "Q5: 2022 FIFA World Cup (Negative)",
        "question": "Which country won the 2022 FIFA World Cup?",
        "tag": "Out-of-Domain (Negative Test)"
    }
]

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_query" not in st.session_state:
    st.session_state.input_query = ""

# Sidebar: Metadata & Sample Question Triggers
with st.sidebar:
    st.image("https://img.icons8.com/color/96/train.png", width=64)
    st.title("Archival Navigator")
    st.caption("GenAI Midsem Project 9")
    
    st.markdown("---")
    st.markdown("### 📋 System Information")
    st.markdown("""
    - **Knowledge Corpus**: `indian_railways_and_telegraph_history_50p.pdf` (50 Pages)
    - **Vector Store**: `./chroma_db` (Persistent ChromaDB)
    - **Embeddings**: Local `all-MiniLM-L6-v2` (ONNX Runtime, 384-dim)
    - **Chunking**: RecursiveCharacter (Size: 1000, Overlap: 200)
    - **Grounding Mode**: Strict anti-hallucination (`[not found]` fallback)
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Assessment Questions (Q1–Q5)")
    st.caption("Click any button below to instantly populate and run:")
    
    for item in ASSESSMENT_SAMPLES:
        if st.button(item["label"], use_container_width=True, help=item["question"]):
            st.session_state.input_query = item["question"]
            st.rerun()

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.input_query = ""
        st.rerun()

# Main Header
st.markdown('<div class="main-header">📜 Project 9: Modern History & Archival Chronicle Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Strictly grounded in primary 19th-century railway and telegraph construction records (<code>indian_railways_and_telegraph_history_50p.pdf</code>). Zero external hallucinations permitted.</div>', unsafe_allow_html=True)

# Query Input Form
with st.container():
    col1, col2 = st.columns([5, 1])
    with col1:
        user_query = st.text_input(
            "Enter your archival question:",
            value=st.session_state.input_query,
            placeholder="e.g. What was the guaranteed interest rate offered under the 1849 contract?",
            label_visibility="collapsed"
        )
    with col2:
        submit_btn = st.button("Ask Assistant", type="primary", use_container_width=True)

# Handle Query Execution
active_query = None
if submit_btn and user_query.strip():
    active_query = user_query.strip()
elif st.session_state.input_query and (not st.session_state.messages or st.session_state.messages[-1].get("question") != st.session_state.input_query):
    active_query = st.session_state.input_query

if active_query:
    with st.spinner("Retrieving archival passages and verifying ground truth..."):
        result = answer(active_query)
        st.session_state.messages.append({
            "question": active_query,
            "result": result
        })
        st.session_state.input_query = ""
        st.rerun()

# Display Conversation History
st.markdown("---")
if not st.session_state.messages:
    st.info("💡 **Getting Started**: Type a question above or click any of the **Sample / Assessment Questions (Q1–Q5)** in the left sidebar to inspect the assistant's retrieval and exact dictionary output.")
else:
    for idx, exchange in enumerate(reversed(st.session_state.messages), 1):
        q = exchange["question"]
        res = exchange["result"]
        ans = res.get("answer", "")
        sources = res.get("sources", [])
        is_not_found = (ans == "[not found]")

        with st.container():
            st.markdown(f"#### ❓ Question: {q}")

            if is_not_found:
                st.error("**Answer:** `[not found]`")
                st.warning("⚠️ **Strict Grounding Notice**: The requested information is not explicitly documented within `indian_railways_and_telegraph_history_50p.pdf`. As per archival fidelity rules, external or general knowledge has been withheld.")
            else:
                st.success(f"**Answer:** {ans}")
                
            # Render Sources
            st.markdown("**Cited Archival Sources:**")
            if sources:
                badges_html = " ".join([f'<span class="source-badge">📄 {src}</span>' for src in sources])
                st.markdown(badges_html, unsafe_allow_html=True)
            else:
                st.markdown("*(None - empty sources list `[]`)*")

            # Expandable Raw JSON Dictionary Output
            with st.expander("🔍 Inspect Exact `answer()` JSON Output", expanded=False):
                st.json(res)

            st.markdown("---")
