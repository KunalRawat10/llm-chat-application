"""
NexChat Engine - Cyber-Modern Multi-Turn LLM Conversation System
================================================================
Author: Kunal Rawat
Tech Stack: Streamlit, Groq API, Custom CSS Animations
"""

import os
import time
import streamlit as st
from groq import Groq, APIConnectionError, RateLimitError, APIStatusError

# Page Configuration
st.set_page_config(
    page_title="NexChat Studio | Next-Gen AI Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injected Custom CSS Animations & Glassmorphism Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&family=JetBrains+Mono:wght@400;600&display=swap');

    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Animated Gradient Title */
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .animated-title {
        background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899, #3b82f6);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 6s ease infinite;
        font-weight: 800;
        font-size: 2.6rem;
        letter-spacing: -0.5px;
        margin-bottom: 0px;
    }

    /* Pulsing Status Dot */
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(34, 197, 94, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(34, 197, 94, 0.12);
        color: #4ade80;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid rgba(34, 197, 94, 0.3);
        margin-bottom: 15px;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #22c55e;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }

    /* Glassmorphic Metric Cards */
    .glass-card {
        background: rgba(30, 34, 45, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px;
        transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.3s ease;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25);
        margin-bottom: 12px;
    }

    .glass-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.4);
    }

    /* Sidebar Refinement */
    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* Button Animations */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.25s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.25);
        border-color: #6366f1;
    }

    /* Telemetry Bar */
    .telemetry-chip {
        display: inline-block;
        background: rgba(99, 102, 241, 0.1);
        color: #a5b4fc;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.8rem;
        margin-top: 6px;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# System Persona Presets
PERSONA_PRESETS = {
    "⚡ Technical Architect": (
        "You are a Principal Software & AI Architect. Provide modular, production-ready "
        "code solutions, detailed system design patterns, and zero conversational filler."
    ),
    "🔬 Deep Learning Researcher": (
        "You are an AI Research Scientist. Break down neural architectures, loss curves, "
        "and mathematical formulations with precise analytical rigor."
    ),
    "🐞 Senior Code Reviewer": (
        "You are an Elite Code Auditor. Review code snippets for edge cases, memory leaks, "
        "algorithmic time complexity (Big-O), and deliver refactored solutions."
    ),
    "🎯 Product Strategist": (
        "You are a Tech Product Lead. Deliver concise, high-impact bulleted summaries, "
        "product roadmaps, and actionable insights."
    ),
    "Custom Persona": ""
}

# Cost Constants
INPUT_COST_PER_M = 0.15
OUTPUT_COST_PER_M = 0.60

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_tokens" not in st.session_state:
    st.session_state.session_tokens = 0
if "session_cost" not in st.session_state:
    st.session_state.session_cost = 0.0
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

# Sidebar Control Hub
with st.sidebar:
    st.markdown('<div class="status-badge"><span class="status-dot"></span> GROQ ACCELERATED ENGINE</div>', unsafe_allow_html=True)
    st.markdown("## ⚙️ Control Center")
    st.caption("Engineered by **Kunal Rawat**")
    
    st.markdown("---")
    st.subheader("🔑 Authentication")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get a free ultra-fast inference API key at console.groq.com"
    )
    
    st.markdown("---")
    st.subheader("🧠 Model Architecture")
    model_name = st.selectbox(
        "Inference Engine",
        [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.6-27b"
        ],
        index=0
    )
    
    persona_choice = st.selectbox("System Persona", list(PERSONA_PRESETS.keys()), index=0)
    
    if persona_choice == "Custom Persona":
        system_prompt = st.text_area("Custom System Directives", height=110, placeholder="Inject custom behavioral rules...")
    else:
        system_prompt = st.text_area("Active System Directives", value=PERSONA_PRESETS[persona_choice], height=110)

    with st.expander("🎛️ Hyperparameter Calibration", expanded=False):
        temperature = st.slider("Temperature (Stochasticity)", 0.0, 1.5, 0.7, 0.05)
        max_tokens = st.slider("Max Completion Tokens", 128, 4096, 1024, 128)
        top_p = st.slider("Top-P (Nucleus Sampling)", 0.1, 1.0, 0.9, 0.05)

    st.markdown("---")
    st.subheader("📊 Session Analytics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="glass-card">
            <span style="font-size:0.75rem; color:#94a3b8;">REQUESTS</span><br>
            <span style="font-size:1.3rem; font-weight:700; color:#f8fafc;">{st.session_state.query_count}</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="glass-card">
            <span style="font-size:0.75rem; color:#94a3b8;">EST. COST</span><br>
            <span style="font-size:1.3rem; font-weight:700; color:#38bdf8;">${st.session_state.session_cost:.5f}</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="glass-card">
            <span style="font-size:0.75rem; color:#94a3b8;">TOKENS</span><br>
            <span style="font-size:1.3rem; font-weight:700; color:#a855f7;">{st.session_state.session_tokens}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.session_state.messages:
        chat_markdown = "# NexChat Multi-Turn Session Log\n\n"
        for m in st.session_state.messages:
            chat_markdown += f"### {m['role'].upper()}\n{m['content']}\n\n---\n\n"
            
        st.download_button(
            label="📥 Export Chat Log (.md)",
            data=chat_markdown,
            file_name="nexchat_session_log.md",
            mime="text/markdown",
            use_container_width=True
        )

    if st.button("🗑️ Flush Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_tokens = 0
        st.session_state.session_cost = 0.0
        st.session_state.query_count = 0
        st.rerun()

# Main App Header
st.markdown('<h1 class="animated-title">⚡ NexChat Studio</h1>', unsafe_allow_html=True)
st.markdown(
    "<p style='color: #94a3b8; font-size: 1.05rem; margin-top: -8px; margin-bottom: 24px;'>"
    "Next-generation conversational interface with real-time streaming, hyperparameter calibration, and token cost telemetry."
    "</p>",
    unsafe_allow_html=True
)

# Starter Quick Prompt Cards
if not st.session_state.messages:
    st.markdown("##### 🚀 Quick Start Workflows")
    p_col1, p_col2, p_col3 = st.columns(3)
    
    with p_col1:
        if st.button("🔍 Explain Transformer Attention", use_container_width=True):
            st.session_state.preset_prompt = "Explain Multi-Head Self-Attention mathematically and architecturally."
    with p_col2:
        if st.button("⚡ Optimize PyTorch Training", use_container_width=True):
            st.session_state.preset_prompt = "How can I resolve GPU data transfer bottlenecks in a PyTorch pipeline?"
    with p_col3:
        if st.button("📊 Vector Cosine vs Dot Product", use_container_width=True):
            st.session_state.preset_prompt = "Compare Cosine Similarity and Dot-Product metric spaces for embeddings."

# Render Message History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input Execution Pipeline
starter_val = st.session_state.pop("preset_prompt", None)
user_prompt = st.chat_input("Enter your prompt or instruction...") or starter_val

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    effective_key = api_key.strip() or os.getenv("GROQ_API_KEY", "")

    if not effective_key:
        err_msg = "❌ **Missing API Key:** Please provide your Groq API key in the left sidebar to initialize inference."
        with st.chat_message("assistant"):
            st.error(err_msg)
        st.session_state.messages.append({"role": "assistant", "content": err_msg})
    else:
        client = Groq(api_key=effective_key)
        
        payload_messages = [{"role": "system", "content": system_prompt.strip() or PERSONA_PRESETS["⚡ Technical Architect"]}]
        for m in st.session_state.messages:
            payload_messages.append({"role": m["role"], "content": m["content"]})

        with st.chat_message("assistant"):
            start_time = time.time()
            try:
                # Live streaming generation
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

                # Telemetry Calculations
                prompt_tokens = len(str(payload_messages)) // 4
                completion_tokens = len(response_content) // 4
                total_tokens = prompt_tokens + completion_tokens
                cost = (
                    (prompt_tokens / 1_000_000 * INPUT_COST_PER_M) +
                    (completion_tokens / 1_000_000 * OUTPUT_COST_PER_M)
                )

                # Accumulate telemetry
                st.session_state.session_tokens += total_tokens
                st.session_state.session_cost += cost
                st.session_state.query_count += 1

                st.markdown(
                    f'<div class="telemetry-chip">⚡ Latency: <b>{latency}s</b> | '
                    f'🔢 Tokens: <b>{total_tokens}</b> | '
                    f'💰 Est. Cost: <b>${cost:.6f}</b></div>',
                    unsafe_allow_html=True
                )
                
                st.session_state.messages.append({"role": "assistant", "content": response_content})

            except RateLimitError:
                err = "⚠️ Rate limit exceeded. Please wait a moment before re-submitting."
                st.warning(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            except APIConnectionError:
                err = "❌ Connection failed. Check your network link."
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            except APIStatusError as e:
                err = f"❌ API Error ({e.status_code}): {e.message}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            except Exception as e:
                err = f"❌ Error: {str(e)}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
