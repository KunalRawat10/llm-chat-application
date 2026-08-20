"""
NexChat Engine - Production Multi-Turn LLM Conversation System
==============================================================
Author: Kunal Rawat
Tech Stack: Streamlit, Groq API, Python
"""

import os
import time
import json
import streamlit as st
from groq import Groq, APIConnectionError, RateLimitError, APIStatusError

# Page Configuration
st.set_page_config(
    page_title="NexChat | AI Conversation Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-box {
        background-color: #1e222d;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #2e3440;
        text-align: center;
    }
    .stChatFloatingInputContainer {
        padding-bottom: 20px;
    }
    .prompt-chip {
        display: inline-block;
        background-color: #2b303c;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        margin: 2px;
        border: 1px solid #3b4252;
    }
</style>
""", unsafe_allow_html=True)

# Preset System Personas
PERSONA_PRESETS = {
    "⚡ Technical Architect": (
        "You are a Principal Software and AI Architect. Provide highly optimized, production-grade "
        "code solutions, detailed architectural breakdowns, and precise technical explanations."
    ),
    "🔬 Machine Learning Researcher": (
        "You are an AI Research Scientist. Explain deep learning architectures, loss functions, "
        "optimization mathematics, and model telemetry rigorously with mathematical grounding."
    ),
    "🐞 Code Debugger & Optimizer": (
        "You are a Senior Code Reviewer. Inspect code for edge cases, memory leaks, algorithmic complexity "
        "(Big-O), and provide refactored, robust implementations."
    ),
    "🎯 Concise Product Strategist": (
        "You are a high-level Tech Product Lead. Deliver concise, high-impact bulleted summaries, "
        "strategic roadmaps, and actionable insights with zero filler."
    ),
    "Custom Persona": ""
}

# Cost calculation constants (Estimated for LLaMA 3.3 70B)
INPUT_COST_PER_M = 0.15
OUTPUT_COST_PER_M = 0.60

# Initialize Session States
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_tokens" not in st.session_state:
    st.session_state.session_tokens = 0
if "session_cost" not in st.session_state:
    st.session_state.session_cost = 0.0
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

# Sidebar Controls
with st.sidebar:
    st.title("⚡ NexChat Studio")
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
    st.subheader("⚙️ Model Configuration")
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
        system_prompt = st.text_area("Custom System Prompt", height=100, placeholder="Enter custom behavioral instructions...")
    else:
        system_prompt = st.text_area("System Instructions", value=PERSONA_PRESETS[persona_choice], height=100)

    with st.expander("🎛️ Hyperparameters", expanded=False):
        temperature = st.slider("Temperature (Sampling)", 0.0, 1.5, 0.7, 0.05)
        max_tokens = st.slider("Max Response Tokens", 128, 4096, 1024, 128)
        top_p = st.slider("Top-P (Nucleus Sampling)", 0.1, 1.0, 0.9, 0.05)

    st.markdown("---")
    st.subheader("📊 Session Telemetry")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Calls", st.session_state.query_count)
        st.metric("Est. Cost", f"${st.session_state.session_cost:.5f}")
    with col2:
        st.metric("Tokens Used", st.session_state.session_tokens)
    
    st.markdown("---")
    
    # Export Chat Features
    if st.session_state.messages:
        chat_markdown = "# NexChat Session Export\n\n"
        for m in st.session_state.messages:
            chat_markdown += f"### {m['role'].capitalize()}\n{m['content']}\n\n"
            
        st.download_button(
            label="📥 Export Chat (.MD)",
            data=chat_markdown,
            file_name="nexchat_session.md",
            mime="text/markdown",
            use_container_width=True
        )

    if st.button("🗑️ Reset Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_tokens = 0
        st.session_state.session_cost = 0.0
        st.session_state.query_count = 0
        st.rerun()

# Main Application Header
st.title("🤖 NexChat | Multi-Turn AI Conversation Engine")
st.markdown(
    "A production-grade conversational interface featuring **real-time streaming**, "
    "**persona orchestration**, **parameter tuning**, and **token cost profiling**."
)

# Starter Quick-Prompt Chips (Only show if chat is empty)
if not st.session_state.messages:
    st.markdown("**💡 Try a starter prompt:**")
    prompt_cols = st.columns(3)
    
    with prompt_cols[0]:
        if st.button("🔍 Explain Transformer Self-Attention", use_container_width=True):
            st.session_state.preset_prompt = "Explain Multi-Head Self-Attention mechanisms mathematically and conceptually."
    with prompt_cols[1]:
        if st.button("🚀 Optimize PyTorch DataLoader", use_container_width=True):
            st.session_state.preset_prompt = "How can I optimize a PyTorch DataLoader for multi-GPU training bottlenecks?"
    with prompt_cols[2]:
        if st.button("📊 Compare Cosine vs Dot-Product", use_container_width=True):
            st.session_state.preset_prompt = "Compare Cosine Similarity and Dot-Product for high-dimensional vector embeddings."

# Render Past Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Capture User Input
starter_val = st.session_state.pop("preset_prompt", None)
user_prompt = st.chat_input("Type your message or prompt here...") or starter_val

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    effective_key = api_key.strip() or os.getenv("GROQ_API_KEY", "")

    if not effective_key:
        err_msg = "❌ **Authentication Required:** Please enter your Groq API key in the left sidebar to begin."
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
                # Streaming Response Engine
                stream = client.chat.completions.create(
                    model=model_name,
                    messages=payload_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stream=True
                )
                
                # Stream words live to the UI
                response_content = st.write_stream(stream)
                latency = round(time.time() - start_time, 2)

                # Heuristic Telemetry Calculation
                prompt_tokens = len(str(payload_messages)) // 4
                completion_tokens = len(response_content) // 4
                total_tokens = prompt_tokens + completion_tokens
                cost = (
                    (prompt_tokens / 1_000_000 * INPUT_COST_PER_M) +
                    (completion_tokens / 1_000_000 * OUTPUT_COST_PER_M)
                )

                # Update Global Telemetry State
                st.session_state.session_tokens += total_tokens
                st.session_state.session_cost += cost
                st.session_state.query_count += 1

                st.caption(
                    f"⚡ **Latency:** `{latency}s` | 🔢 **Tokens:** `{total_tokens}` | 💰 **Cost:** `${cost:.6f}`"
                )
                
                st.session_state.messages.append({"role": "assistant", "content": response_content})

            except RateLimitError:
                err = "⚠️ Rate limit reached. Please wait a few seconds before trying again."
                st.warning(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            except APIConnectionError:
                err = "❌ Network connection error. Please verify your internet link."
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            except APIStatusError as e:
                err = f"❌ API Error ({e.status_code}): {e.message}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            except Exception as e:
                err = f"❌ Unexpected Error: {str(e)}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
