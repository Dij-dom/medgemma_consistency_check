import gradio as gr
from llama_cpp import Llama
import os, time

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--medgemma-27b-text-it-GGUF/snapshots/334fbf6811c963d223f6ac107a459347353f068d/medgemma-27b-text-it-Q4_K_M.gguf"
)


llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=8,   # adjust based on your CPU cores
    verbose=False
)

# Chat function
def chat_fn(message, history):
    # Convert Gradio history to chat format
    messages = []
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    
    # Add current message
    messages.append({"role": "user", "content": message})

    start_time = time.time()
    # Generate response
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=512,
        temperature=0.0,
    )
    end_time = time.time()

    output = response["choices"][0]["message"]["content"]
    elapsed = end_time - start_time
    # Append timing info
    output += f"\n\n⏱️ Response generated in {elapsed:.2f} seconds."
    return output

# Launch Gradio ChatInterface
demo = gr.ChatInterface(
    fn=chat_fn,
    title="MedGemma 27B - Consistency Analysis",
    description="Chat with the MedGemma model (quantized Q4_K_M) running locally",
    theme="soft"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
