import os
import time
from typing import List, Dict, Tuple
import gradio as gr
from groq import Groq, APIConnectionError, RateLimitError, APIStatusError

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert AI engineering assistant. Provide technically accurate, "
    "concise, and well-structured responses with clear code examples where applicable."
)

INPUT_TOKEN_COST_PER_MILLION = 0.59
OUTPUT_TOKEN_COST_PER_MILLION = 0.79

def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)

def chat_response(
    message: str,
    history: List[Dict[str, str]],
    api_key: str,
    system_prompt: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    top_p: float
) -> Tuple[List[Dict[str, str]], str]:
    if not message.strip():
        return history, "Please enter a message."

    effective_api_key = api_key.strip() or os.getenv("GROQ_API_KEY", "")
    if not effective_api_key:
        error_msg = "Error: API key is missing. Please enter your Groq API key in the sidebar."
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return history, "Missing API Key"

    client = Groq(api_key=effective_api_key)
    messages = [{"role": "system", "content": system_prompt.strip() or DEFAULT_SYSTEM_PROMPT}]
    
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
        
    messages.append({"role": "user", "content": message})

    start_time = time.time()
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=False
        )

        response_content = completion.choices[0].message.content
        latency = round(time.time() - start_time, 2)

        prompt_tokens = completion.usage.prompt_tokens if completion.usage else estimate_tokens(str(messages))
        completion_tokens = completion.usage.completion_tokens if completion.usage else estimate_tokens(response_content)
        total_tokens = prompt_tokens + completion_tokens

        cost = (
            (prompt_tokens / 1_000_000 * INPUT_TOKEN_COST_PER_MILLION) +
            (completion_tokens / 1_000_000 * OUTPUT_TOKEN_COST_PER_MILLION)
        )

        metrics_info = (
            f"Latency: {latency}s | Tokens: {total_tokens} (Prompt: {prompt_tokens}, Completion: {completion_tokens}) | "
            f"Est. Cost: ${cost:.6f}"
        )

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response_content})
        return history, metrics_info

    except RateLimitError:
        err = "Rate limit reached. Please wait a few moments before trying again."
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": err})
        return history, "Rate Limit Exceeded"
    except APIConnectionError:
        err = "Connection failed. Please check network connectivity."
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": err})
        return history, "Network Error"
    except APIStatusError as e:
        err = f"API Error ({e.status_code}): {e.message}"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": err})
        return history, f"HTTP Error {e.status_code}"
    except Exception as e:
        err = f"Unexpected Error: {str(e)}"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": err})
        return history, "Internal Error"

def clear_conversation():
    return [], "", "Session reset. Conversation memory cleared."

with gr.Blocks(title="AI Chat & Assistant Engine") as demo:
    gr.Markdown("# 🤖 AI Chat Application & Conversation Engine")
    gr.Markdown("A multi-turn LLM chat app with configurable system personas, hyperparameter controls, and token cost telemetry.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Configuration")
            api_key_input = gr.Textbox(
                label="Groq API Key",
                type="password",
                placeholder="gsk_... (Get one free at console.groq.com)"
            )
            model_selector = gr.Dropdown(
                choices=[
                    "llama-3.3-70b-versatile",
                    "llama-3.1-8b-instant",
                    "mixtral-8x7b-32768",
                    "gemma2-9b-it"
                ],
                value="llama-3.3-70b-versatile",
                label="LLM Model Engine"
            )
            system_prompt_input = gr.Textbox(
                label="System Instructions (Persona / Rules)",
                value=DEFAULT_SYSTEM_PROMPT,
                lines=4
            )
            
            with gr.Accordion("Hyperparameter Controls", open=False):
                temp_slider = gr.Slider(0.0, 1.5, value=0.7, step=0.1, label="Temperature")
                tokens_slider = gr.Slider(64, 2048, value=512, step=64, label="Max Output Tokens")
                topp_slider = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-P")

            clear_btn = gr.Button("🗑️ Clear Chat History", variant="secondary")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation", type="messages", height=520)
            user_msg = gr.Textbox(label="Message", placeholder="Type your query and press Enter...", lines=2)
            metrics_display = gr.Markdown("📊 Session Status: Ready")
            send_btn = gr.Button("Send Message", variant="primary")

    send_btn.click(
        fn=chat_response,
        inputs=[user_msg, chatbot, api_key_input, system_prompt_input, model_selector, temp_slider, tokens_slider, topp_slider],
        outputs=[chatbot, metrics_display]
    ).then(lambda: "", outputs=[user_msg])

    user_msg.submit(
        fn=chat_response,
        inputs=[user_msg, chatbot, api_key_input, system_prompt_input, model_selector, temp_slider, tokens_slider, topp_slider],
        outputs=[chatbot, metrics_display]
    ).then(lambda: "", outputs=[user_msg])

    clear_btn.click(fn=clear_conversation, outputs=[chatbot, user_msg, metrics_display])

if __name__ == "__main__":
    demo.launch()
