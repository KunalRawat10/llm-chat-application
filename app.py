import os
import time
import streamlit as st
from groq import Groq, APIConnectionError, RateLimitError, APIStatusError

st.set_page_config(
    page_title="AI Chat & Assistant Engine",
    page_icon="🤖",
    layout="wide"
)

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert AI engineering assistant. Provide technically accurate, "
    "concise, and well-structured responses with clear code examples where applicable."
)

INPUT_TOKEN_COST_PER_MILLION = 0.15
OUTPUT_TOKEN_COST_PER_MILLION = 0.60

# Sidebar Controls
with st.sidebar:
    st.title("⚙️ Configuration")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_... (get one free at console.groq.com)"
    )
    model_name = st.selectbox(
        "LLM Model Engine",
        [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.6-27b"
        ],
        index=0
    )
    system_prompt = st.text_area(
        "System Instructions (Persona / Rules)",
        value=DEFAULT_SYSTEM_PROMPT,
        height=120
    )
    
    st.markdown("### Hyperparameters")
    temperature = st.slider("Temperature (Creativity)", 0.0, 1.5, 0.7, 0.1)
    max_tokens = st.slider("Max Output Tokens", 64, 4096, 1024, 64)
    top_p = st.slider("Top-P (Nucleus Sampling)", 0.1, 1.0, 0.9, 0.05)
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("🤖 AI Chat Application & Conversation Engine")
st.caption("A multi-turn LLM chat app with configurable system personas, hyperparameter controls, and token cost telemetry.")

# Initialize session message state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Message Handling
if prompt := st.chat_input("Type your query and press Enter..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    effective_key = api_key.strip() or os.getenv("GROQ_API_KEY", "")
    
    if not effective_key:
        err_msg = "❌ API key is missing. Please enter your Groq API key in the sidebar."
        with st.chat_message("assistant"):
            st.error(err_msg)
        st.session_state.messages.append({"role": "assistant", "content": err_msg})
    else:
        client = Groq(api_key=effective_key)
        
        # Build payload with system prompt & multi-turn history
        payload_messages = [{"role": "system", "content": system_prompt.strip() or DEFAULT_SYSTEM_PROMPT}]
        for m in st.session_state.messages:
            payload_messages.append({"role": m["role"], "content": m["content"]})

        with st.chat_message("assistant"):
            start_time = time.time()
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=payload_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p
                )
                response = completion.choices[0].message.content
                latency = round(time.time() - start_time, 2)

                prompt_tokens = completion.usage.prompt_tokens if completion.usage else len(str(payload_messages)) // 4
                completion_tokens = completion.usage.completion_tokens if completion.usage else len(response) // 4
                total_tokens = prompt_tokens + completion_tokens
                cost = (
                    (prompt_tokens / 1_000_000 * INPUT_TOKEN_COST_PER_MILLION) +
                    (completion_tokens / 1_000_000 * OUTPUT_TOKEN_COST_PER_MILLION)
                )

                st.markdown(response)
                st.caption(f"⚡ Latency: {latency}s | 🔢 Tokens: {total_tokens} (Prompt: {prompt_tokens}, Completion: {completion_tokens}) | 💰 Est. Cost: ${cost:.6f}")
                
                st.session_state.messages.append({"role": "assistant", "content": response})

            except RateLimitError:
                err = "⚠️ Rate limit reached. Please wait a few seconds before trying again."
                st.warning(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            except APIConnectionError:
                err = "❌ Connection failed. Please check network connectivity."
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
