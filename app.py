"""
NexChat Studio - Monochromatic 3D Obsidian & Starfield Interface
================================================================
Author: Kunal Rawat
Tech Stack: Streamlit, Groq API, 3D Obsidian Glassmorphic CSS

Features:
- Real-time token streaming with live throughput velocity (tok/s)
- Multi-turn conversation state & context window utilization buffer
- 3D obsidian beveled cards, tactile buttons, and animated starfield
- Multi-format session export (.md and ML fine-tuning .jsonl)
- Defensive API exception handling & granular per-query telemetry
"""

import os
import time
import json
import streamlit as st
from groq import Groq, APIConnectionError, RateLimitError, APIStatusError

# Page Configuration
st.set_page_config(
    page_title="NexChat Studio",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Deep Space Starfield with 3D Obsidian & Metallic Rim Lighting
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Space Grotesk', -apple-system, sans-serif !important;
        color: #f4f4f5 !important;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    @keyframes starPulse {
        0%, 100% { opacity: 0.85; }
        50% { opacity: 1.0; }
    }

    .stApp {
        background-color: #000000 !important;
        background-image: 
            radial-gradient(1px 1px at 25px 35px, #ffffff 100%, transparent),
            radial-gradient(1px 1px at 85px 130px, rgba(255,255,255,0.7) 100%, transparent),
            radial-gradient(1.5px 1.5px at 170px 50px, #ffffff 100%, transparent),
            radial-gradient(1px 1px at 260px 210px, rgba(255,255,255,0.5) 100%, transparent),
            radial-gradient(2px 2px at 340px 280px, #ffffff 100%, transparent),
            radial-gradient(1px 1px at 430px 90px, rgba(255,255,255,0.8) 100%, transparent),
            radial-gradient(1.5px 1.5px at 510px 240px, #ffffff 100%, transparent);
        background-size: 550px 550px !important;
        animation: starPulse 6s ease-in-out infinite !important;
    }

    .title-3d {
        font-size: 2.3rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #ffffff;
        text-shadow: 
            0 1px 0 #52525b,
            0 2px 0 #3f3f46,
            0 3px 0 #27272a,
            0 4px 0 #18181b,
            0 8px 24px rgba(255, 255, 255, 0.12);
        margin-bottom: 2px;
    }

    .subtitle-text {
        color: #a1a1aa;
        font-size: 0.95rem;
        margin-bottom: 24px;
    }

    .card-3d {
        background: linear-gradient(180deg, #111113 0%, #080809 100%);
        border-top: 1px solid rgba(255, 255, 255, 0.15);
        border-left: 1px solid rgba(255, 255, 255, 0.08);
        border-right: 1px solid rgba(255, 255, 255, 0.04);
        border-bottom: 1px solid #000000;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 
            0 8px 20px -4px rgba(0, 0, 0, 0.8),
            inset 0 1px 0 rgba(255, 255, 255, 0.12);
        transition: all 0.25s cubic-bezier(0.2, 0.8, 0.2, 1);
    }

    .card-3d:hover {
        transform: translateY(-3px);
        border-top: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 
            0 14px 28px -4px rgba(0, 0, 0, 0.9),
            0 0 15px rgba(255, 255, 255, 0.05),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
    }

    .stButton > button {
        background: linear-gradient(180deg, #1f1f23 0%, #121215 100%) !important;
        color: #f4f4f5 !important;
        border-top: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-left: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-right: 1px solid rgba(0, 0, 0, 0.5) !important;
        border-bottom: 1px solid rgba(0, 0, 0, 0.8) !important;
        border-radius: 9px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        box-shadow: 0 4px 0 #09090b, 0 6px 14px rgba(0, 0, 0, 0.6) !important;
        transition: all 0.12s ease !important;
    }

    .stButton > button:hover {
        background: linear-gradient(180deg, #27272a 0%, #18181b 100%) !important;
        border-top-color: rgba(255, 255, 255, 0.35) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 5px 0 #09090b, 0 8px 18px rgba(255, 255, 255, 0.06) !important;
    }

    .stButton > button:active {
        transform: translateY(3px) !important;
        box-shadow: 0 1px 0 #09090b !important;
    }

    section[data-testid="stSidebar"] {
        background: #050507 !important;
        border-right: 1px solid #18181b !important;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: #09090b;
        color: #d4d4d8;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.76rem;
        font-weight: 600;
        border: 1px solid #27272a;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
        margin-bottom: 16px;
    }

    .status-dot {
        width: 6px;
        height: 6px;
        background-color: #22c55e;
        border-radius: 50%;
        box-shadow: 0 0 8px #22c55e;
    }

    .telemetry-chip {
        display: inline-block;
        background: #09090b;
        color: #a1a1aa;
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 0.8rem;
        margin-top: 8px;
        border-top: 1px solid rgba(255, 255, 255, 0.12);
        border-left: 1px solid rgba(255, 255, 255, 0.06);
        border-right: 1px solid rgba(0, 0, 0, 0.5);
        border-bottom: 1px solid rgba(0, 0, 0, 0.8);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4);
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

# Cost Constants & Context Limits
INPUT_COST_PER_M = 0.15
OUTPUT_COST_PER_M = 0.60
CONTEXT_LIMIT = 8192

# State Management
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_tokens" not in st.session_state:
    st.session_state.session_tokens = 0
if "session_cost" not in st.session_state:
    st.session_state.session_cost = 0.0
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "current_context_tokens" not in st.session_state:
    st.session_state.current_context_tokens = 0

# Sidebar Control Hub
with st.sidebar:
    st.markdown('<div class="status-badge"><span class="status-dot"></span> INFERENCE READY</div>', unsafe_allow_html=True)
    st.markdown("### Controls")
    
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
        system_prompt = st.text_area("Custom Directives", height=100, placeholder="Define behavioral rules...")
    else:
        system_prompt = st.text_area("Active Directives", value=PERSONA_PRESETS[persona_choice], height=100)

    with st.expander("Hyperparameters", expanded=False):
        temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
        max_tokens = st.slider("Max Tokens", 128, 4096, 1024, 128)
        top_p = st.slider("Top-P", 0.1, 1.0, 0.9, 0.05)

    st.markdown("---")
    st.markdown("### Context & Memory")
    
    # Live Context Window Utilization Meter
    context_ratio = min(1.0, st.session_state.current_context_tokens / CONTEXT_LIMIT)
    st.progress(
        context_ratio,
        text=f"Buffer: {st.session_state.current_context_tokens} / {CONTEXT_LIMIT} tok ({int(context_ratio * 100)}%)"
    )

    st.markdown("### Session Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="card-3d">
            <span style="font-size:0.72rem; color:#71717a; font-weight:600;">REQUESTS</span><br>
            <span style="font-size:1.25rem; font-weight:700; color:#fafafa;">{st.session_state.query_count}</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card-3d">
            <span style="font-size:0.72rem; color:#71717a; font-weight:600;">EST. COST</span><br>
            <span style="font-size:1.25rem; font-weight:700; color:#fafafa;">${st.session_state.session_cost:.5f}</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="card-3d">
            <span style="font-size:0.72rem; color:#71717a; font-weight:600;">SESSION TOKENS</span><br>
            <span style="font-size:1.25rem; font-weight:700; color:#fafafa;">{st.session_state.session_tokens}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Multi-Format Session Export
    if st.session_state.messages:
        # Markdown Export
        chat_markdown = "# NexChat Session Export\n\n"
        for m in st.session_state.messages:
            chat_markdown += f"### {m['role'].upper()}\n{m['content']}\n\n---\n\n"
            
        st.download_button(
            label="Export Log (.md)",
            data=chat_markdown,
            file_name="session_log.md",
            mime="text/markdown",
            use_container_width=True
        )

        # ML Fine-Tuning JSONL Export
        jsonl_data = json.dumps({"messages": st.session_state.messages}, indent=None) + "\n"
        st.download_button(
            label="Export Dataset (.jsonl)",
            data=jsonl_data,
            file_name="train_dataset.jsonl",
            mime="application/json",
            use_container_width=True
        )

    if st.button("Reset Session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_tokens = 0
        st.session_state.session_cost = 0.0
        st.session_state.query_count = 0
        st.session_state.current_context_tokens = 0
        st.rerun()

# Main Canvas Header
st.markdown('<div class="title-3d">✦ NEXCHAT STUDIO</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle-text">High-throughput conversational AI workspace with streaming inference and token telemetry.</div>',
    unsafe_allow_html=True
)

# Starter Quick Prompt Cards
if not st.session_state.messages:
    st.markdown("<span style='font-size:0.85rem; color:#a1a1aa; font-weight:600;'>QUICK WORKFLOWS</span>", unsafe_allow_html=True)
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

# Text Stream Generator to cleanly extract content strings
def stream_text_chunks(raw_stream):
    for chunk in raw_stream:
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if hasattr(delta, 'content') and delta.content:
                yield delta.content

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
                raw_stream = client.chat.completions.create(
                    model=model_name,
                    messages=payload_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stream=True
                )
                
                response_content = st.write_stream(stream_text_chunks(raw_stream))
                latency = round(time.time() - start_time, 2)

                # Telemetry & Throughput Calculations
                prompt_tokens = len(str(payload_messages)) // 4
                completion_tokens = len(response_content) // 4
                total_tokens = prompt_tokens + completion_tokens
                tokens_per_sec = round(completion_tokens / max(latency, 0.01), 1)
                
                cost = (
                    (prompt_tokens / 1_000_000 * INPUT_COST_PER_M) +
                    (completion_tokens / 1_000_000 * OUTPUT_COST_PER_M)
                )

                # State Updates
                st.session_state.session_tokens += total_tokens
                st.session_state.session_cost += cost
                st.session_state.query_count += 1
                st.session_state.current_context_tokens = total_tokens

                st.markdown(
                    f'<div class="telemetry-chip">⚡ Latency: <b>{latency}s</b> | '
                    f'🚀 Speed: <b>{tokens_per_sec} tok/s</b> | '
                    f'🔢 Tokens: <b>{total_tokens}</b> (In: {prompt_tokens}, Out: {completion_tokens}) | '
                    f'💰 Cost: <b>${cost:.6f}</b></div>',
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
