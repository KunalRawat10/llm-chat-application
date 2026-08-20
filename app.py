"""
NexChat Studio - Cyber-Futuristic 3D & Aurora Animated Interface
================================================================
Author: Kunal Rawat
Tech Stack: Streamlit, Groq API, Advanced CSS3 Keyframes & 3D Lighting
"""

import os
import time
import streamlit as st
from groq import Groq, APIConnectionError, RateLimitError, APIStatusError

# Page Configuration
st.set_page_config(
    page_title="NexChat Studio | Next-Gen AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced 3D Aesthetics, Aurora Mesh Animation, and Cyberpunk Glassmorphism
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700;900&family=JetBrains+Mono:wght@400;600&display=swap');

    /* Global Typography */
    html, body, [class*="css"], .stApp {
        font-family: 'Space Grotesk', sans-serif !important;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Ambient Moving Aurora Mesh Background */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(124, 58, 237, 0.18) 0%, transparent 40%),
                    radial-gradient(circle at 90% 80%, rgba(6, 182, 212, 0.18) 0%, transparent 40%),
                    radial-gradient(circle at 50% 50%, rgba(236, 72, 153, 0.12) 0%, transparent 60%),
                    linear-gradient(180deg, #070913 0%, #0c1022 50%, #05070e 100%) !important;
        background-attachment: fixed !important;
        background-size: 200% 200% !important;
        animation: auroraFlow 14s ease infinite alternate !important;
    }

    @keyframes auroraFlow {
        0% { background-position: 0% 0%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 100%; }
    }

    /* 3D Floating Isometric Header */
    @keyframes float3D {
        0% { transform: translateY(0px) rotateX(0deg); }
        50% { transform: translateY(-7px) rotateX(4deg); }
        100% { transform: translateY(0px) rotateX(0deg); }
    }

    .hero-3d-title {
        font-size: 3.2rem;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #ffffff;
        text-shadow: 
            0 1px 0 #9333ea,
            0 2px 0 #7e22ce,
            0 3px 0 #6b21a8,
            0 4px 0 #581c87,
            0 6px 12px rgba(147, 51, 234, 0.4),
            0 12px 30px rgba(6, 182, 212, 0.3);
        animation: float3D 5s ease-in-out infinite;
        margin-bottom: 0px;
        perspective: 1000px;
    }

    /* Neon Hologram Live Status Badge */
    @keyframes pulseHolo {
        0% { box-shadow: 0 0 5px rgba(6, 182, 212, 0.6), inset 0 0 5px rgba(6, 182, 212, 0.3); }
        50% { box-shadow: 0 0 20px rgba(6, 182, 212, 0.9), inset 0 0 10px rgba(6, 182, 212, 0.6); }
        100% { box-shadow: 0 0 5px rgba(6, 182, 212, 0.6), inset 0 0 5px rgba(6, 182, 212, 0.3); }
    }

    .holo-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(6, 182, 212, 0.1);
        color: #38bdf8;
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        border: 1px solid rgba(56, 189, 248, 0.4);
        animation: pulseHolo 3s infinite;
        margin-bottom: 15px;
        backdrop-filter: blur(10px);
    }

    /* 3D Glassmorphic Cards */
    .glass-card-3d {
        background: rgba(18, 22, 38, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5),
                    inset 0 1px 1px rgba(255, 255, 255, 0.15);
        transition: all 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        margin-bottom: 12px;
    }

    .glass-card-3d:hover {
        transform: translateY(-5px) scale(1.02);
        border-color: rgba(147, 51, 234, 0.5);
        box-shadow: 0 15px 35px -5px rgba(124, 58, 237, 0.3),
                    inset 0 1px 1px rgba(255, 255, 255, 0.25);
    }

    /* Sidebar Custom Glass Effect */
    section[data-testid="stSidebar"] {
        background: rgba(8, 11, 22, 0.85) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* 3D Button Depth */
    .stButton > button {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(129, 140, 248, 0.3) !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 10px 20px !important;
        box-shadow: 0 4px 0 #1e1b4b, 0 8px 20px rgba(79, 70, 229, 0.25) !important;
        transition: all 0.15s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 0 #1e1b4b, 0 12px 25px rgba(124, 58, 237, 0.4) !important;
        border-color: #a855f7 !important;
    }

    .stButton > button:active {
        transform: translateY(3px) !important;
        box-shadow: 0 1px 0 #1e1b4b !important;
    }

    /* Glowing Telemetry Chip */
    .telemetry-chip {
        display: inline-block;
        background: linear-gradient(90deg, rgba(124, 58, 237, 0.15), rgba(6, 182, 212, 0.15));
        color: #e0e7ff;
        padding: 6px 14px;
        border-radius: 10px;
        font-size: 0.82rem;
        margin-top: 8px;
        border: 1px solid rgba(124, 58, 237, 0.3);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# System Personas
PERSONA_PRESETS = {
    "⚡ Technical Architect": (
        "You are a Principal Software & AI Architect. Provide modular, production-grade "
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

# Session State Initialization
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
    st.markdown('<div class="holo-badge">⚡ GROQ ACCELERATED ENGINE</div>', unsafe_allow_html=True)
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
    st.subheader("📊 Live Telemetry")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="glass-card-3d">
            <span style="font-size:0.75rem; color:#94a3b8; font-weight:700;">REQUESTS</span><br>
            <span style="font-size:1.4rem; font-weight:800; color:#f8fafc;">{st.session_state.query_count}</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="glass-card-3d">
            <span style="font-size:0.75rem; color:#94a3b8; font-weight:700;">EST. COST</span><br>
            <span style="font-size:1.4rem; font-weight:800; color:#38bdf8;">${st.session_state.session_cost:.5f}</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="glass-card-3d">
            <span style="font-size:0.75rem; color:#94a3b8; font-weight:700;">TOKENS</span><br>
            <span style="font-size:1.4rem; font-weight:800; color:#c084fc;">{st.session_state.session_tokens}</span>
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

# Main Header & 3D Title
st.markdown('<h1 class="hero-3d-title">⚡ NexChat Studio</h1>', unsafe_allow_html=True)
st.markdown(
    "<p style='color: #94a3b8; font-size: 1.1rem; margin-top: -6px; margin-bottom: 26px;'>"
    "Next-generation conversational interface with real-time streaming, hyperparameter calibration, and token cost telemetry."
    "</p>",
    unsafe_allow_html=True
)

# Starter Quick Prompt Cards
if not st.session_state.messages:
    st.markdown("##### 🚀 Fast-Track Prompt Presets")
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
