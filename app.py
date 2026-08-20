"""
NexChat Studio - Pure Dark & Starfield AI Conversation Engine
=============================================================
Author: Kunal Rawat
Tech Stack: Streamlit, Groq API, Custom Minimalist CSS Starfield
"""

import os
import time
import streamlit as st
from groq import Groq, APIConnectionError, RateLimitError, APIStatusError

# Page Configuration
st.set_page_config(
    page_title="NexChat Studio",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimalist Deep-Black Starfield & Sleek UI Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global Typography & Deep Dark Background */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #f1f5f9 !important;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Pure Black Canvas with Crisp Starfield Grid */
    .stApp {
        background-color: #000000 !important;
        background-image: 
            radial-gradient(1px 1px at 20px 30px, #ffffff 100%, transparent),
            radial-gradient(1px 1px at 75px 120px, rgba(255,255,255,0.7) 100%, transparent),
            radial-gradient(1.5px 1.5px at 160px 45px, #ffffff 100%, transparent),
            radial-gradient(1px 1px at 240px 190px, rgba(255,255,255,0.4) 100%, transparent),
            radial-gradient(1.5px 1.5px at 320px 260px, rgba(255,255,255,0.8) 100%, transparent),
            radial-gradient(1px 1px at 410px 80px, #ffffff 100%, transparent),
            radial-gradient(1px 1px at 480px 220px, rgba(255,255,255,0.6) 100%, transparent);
        background-size: 550px 550px !important;
    }

    /* Minimalist Sidebar */
    section[data-testid="stSidebar"] {
        background: #030303 !important;
        border-right: 1px solid #171717 !important;
    }

    /* Clean Sleek Header */
    .brand-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.8px;
        color: #ffffff;
        margin-bottom: 2px;
    }

    .brand-sub {
        color: #71717a;
        font-size: 0.95rem;
        margin-bottom: 24px;
    }

    /* Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: #09090b;
        color: #a1a1aa;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
        border: 1px solid #27272a;
        margin-bottom: 18px;
    }

    .status-dot {
        width: 6px;
        height: 6px;
        background-color: #22c55e;
        border-radius: 50%;
    }

    /* Minimalist Monochromatic Metric Panels */
    .metric-card {
        background: #09090b;
        border: 1px solid #18181b;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 10px;
    }
    
    .metric-card:hover {
        border-color: #27272a;
    }

    /* Clean Solid Buttons */
    .stButton > button {
        background: #18181b !important;
        color: #fafafa !important;
        border: 1px solid #27272a !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        background: #27272a !important;
        border-color: #3f3f46 !important;
        color: #ffffff !important;
    }

    /* Inline Telemetry Chip */
    .telemetry-chip {
        display: inline-block;
        background: #09090b;
        color: #a1a1aa;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        margin-top: 6px;
        border: 1px solid #18181b;
    }
</style>
""", unsafe_allow_html=True)

# System Personas
PERSONA_PRESETS = {
    "Technical Architect": (
        "You are a Principal Software & AI Architect. Provide modular, production-grade "
        "code solutions, system architecture patterns, and concise technical breakdowns."
    ),
    "Machine Learning Engineer": (
        "You are an AI/ML Specialist. Break down model architectures, optimization mathematics, "
        "and training strategies with clear empirical reasoning."
    ),
    "Senior Code Reviewer": (
        "You are a Senior Code Auditor. Review code snippets for edge cases, memory leaks, "
        "time complexity (Big-O), and deliver refactored solutions."
    ),
    "Custom Persona": ""
}

# Cost Constants
INPUT_COST_PER_M = 0.15
OUTPUT_COST_PER_M = 0.60

# State Management
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_tokens" not in st.session_state:
    st.session_state.session_tokens = 0
if "session_cost" not in st.session_state:
    st.session_state.session_cost = 0.0
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

# Sidebar Control Panel
with st.sidebar:
    st.markdown('<div class="status-badge"><span class="status-dot"></span> GROQ ENGINE ACTIVE</div>', unsafe_allow_html=True)
    st.markdown("### Settings")
    
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Obtain a free key from console.groq.com"
    )
    
    model_name = st.selectbox(
        "Model Engine",
        [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.6-27b"
        ],
        index=0
    )
    
    persona_choice = st.selectbox("System Persona", list(PERSONA_PRESETS.keys()), index=0)
    
    if persona_choice == "Custom Persona":
        system_prompt = st.text_area("Custom System Directives", height=100, placeholder="Define behavioral rules...")
    else:
        system_prompt = st.text_area("System Directives", value=PERSONA_PRESETS[persona_choice], height=100)

    with st.expander("Hyperparameters", expanded=False):
        temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
        max_tokens = st.slider("Max Tokens", 128, 4096, 1024, 128)
        top_p = st.slider("Top-P", 0.1, 1.0, 0.9, 0.05)

    st.markdown("---")
    st.markdown("### Telemetry")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <span style="font-size:0.72rem; color:#71717a;">REQUESTS</span><br>
            <span style="font-size:1.15rem; font-weight:600; color:#fafafa;">{st.session_state.query_count}</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card">
            <span style="font-size:0.72rem; color:#71717a;">EST. COST</span><br>
            <span style="font-size:1.15rem; font-weight:600; color:#fafafa;">${st.session_state.session_cost:.5f}</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <span style="font-size:0.72rem; color:#71717a;">TOKENS</span><br>
            <span style="font-size:1.15rem; font-weight:600; color:#fafafa;">{st.session_state.session_tokens}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.session_state.messages:
        chat_markdown = "# NexChat Session Export\n\n"
        for m in st.session_state.messages:
            chat_markdown += f"### {m['role'].upper()}\n{m['content']}\n\n---\n\n"
            
        st.download_button(
            label="Export Session (.md)",
            data=chat_markdown,
            file_name="session_log.md",
            mime="text/markdown",
            use_container_width=True
        )

    if st.button("Reset Session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_tokens = 0
        st.session_state.session_cost = 0.0
        st.session_state.query_count = 0
        st.rerun()

# Main Header
st.markdown('<div class="brand-title">✦ NexChat Studio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="brand-sub">High-throughput conversational AI workspace with streaming inference and token telemetry.</div>',
    unsafe_allow_html=True
)

# Starter Prompt Chips
if not st.session_state.messages:
    st.markdown("<span style='font-size:0.85rem; color:#a1a1aa; font-weight:500;'>Quick Start</span>", unsafe_allow_html=True)
    p_col1, p_col2, p_col3 = st.columns(3)
    
    with p_col1:
        if st.button("Explain Transformer Attention", use_container_width=True):
            st.session_state.preset_prompt = "Explain Multi-Head Self-Attention mathematically and architecturally."
    with p_col2:
        if st.button("Optimize PyTorch Training", use_container_width=True):
            st.session_state.preset_prompt = "How can I resolve GPU data transfer bottlenecks in a PyTorch pipeline?"
    with p_col3:
        if st.button("Vector Cosine vs Dot Product", use_container_width=True):
            st.session_state.preset_prompt = "Compare Cosine Similarity and Dot-Product metric spaces for embeddings."

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Query Submission Pipeline
starter_val = st.session_state.pop("preset_prompt", None)
user_prompt = st.chat_input("Ask a question or enter a command...") or starter_val

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    effective_key = api_key.strip() or os.getenv("GROQ_API_KEY", "")

    if not effective_key:
        err_msg = "Please enter a valid Groq API key in the sidebar."
        with st.chat_message("assistant"):
            st.error(err_msg)
        st.session_state.messages.append({"role": "assistant", "content": err_msg})
    else:
        client = Groq(api_key=effective_key)
        
        payload_messages = [{"role": "system", "content": system_prompt.strip() or PERSONA_PRESETS["Technical Architect"]}]
        for m in st.session_state.messages:
            payload_messages.append({"role": m["role"], "content": m["content"]})

        with st.chat_message("assistant"):
            start_time = time.time()
            try:
                stream = client.chat.completions.create(
                    model=model_name,
                    messages=payload_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stream=True
                )
                
                response_content = st.write_stream(stream)
                latency = round(time.time() - start_time, 2)

                prompt_tokens = len(str(payload_messages)) // 4
                completion_tokens = len(response_content) // 4
                total_tokens = prompt_tokens + completion_tokens
                cost = (
                    (prompt_tokens / 1_000_000 * INPUT_COST_PER_M) +
                    (completion_tokens / 1_000_000 * OUTPUT_COST_PER_M)
                )

                st.session_state.session_tokens += total_tokens
                st.session_state.session_cost += cost
                st.session_state.query_count += 1

                st.markdown(
                    f'<div class="telemetry-chip">Latency: <b>{latency}s</b> | '
                    f'Tokens: <b>{total_tokens}</b> | '
                    f'Cost: <b>${cost:.6f}</b></div>',
                    unsafe_allow_html=True
                )
                
                st.session_state.messages.append({"role": "assistant", "content": response_content})

            except RateLimitError:
                err = "Rate limit reached. Please retry in a few seconds."
                st.warning(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            except APIConnectionError:
                err = "Connection failed. Check your network link."
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            except APIStatusError as e:
                err = f"API Error ({e.status_code}): {e.message}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            except Exception as e:
                err = f"Error: {str(e)}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
