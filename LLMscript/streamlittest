import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Phi-2 Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Phi-2 Chatbot")
st.write("Powered by Microsoft's Phi-2 model")

MODEL_NAME = "microsoft/phi-2"

# ----------------------------
# Load Model Once
# ----------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model.eval()
    return tokenizer, model

with st.spinner("Loading Phi-2 model..."):
    tokenizer, model = load_model()

# ----------------------------
# Session State
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# Display Chat History
# ----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if "input_tokens" in message:
            st.caption(
                f"Input Tokens: {message['input_tokens']} | "
                f"Output Tokens: {message['output_tokens']} | "
                f"Total Tokens: {message['total_tokens']}"
            )

# ----------------------------
# Chat Input
# ----------------------------
prompt = st.chat_input("Type your message")

if prompt:
    # Display User Message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Count Input Tokens
    input_ids = tokenizer.encode(
        prompt,
        return_tensors="pt"
    )

    input_token_count = input_ids.shape[1]

    # Generate Response
    with st.spinner("Thinking..."):
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )

    generated_ids = output_ids[0][input_token_count:]

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    ).strip()

    output_token_count = len(generated_ids)
    total_token_count = input_token_count + output_token_count

    # Display Assistant Response
    with st.chat_message("assistant"):
        st.markdown(response)

        st.caption(
            f"Input Tokens: {input_token_count} | "
            f"Output Tokens: {output_token_count} | "
            f"Total Tokens: {total_token_count}"
        )

    # Save to History
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "input_tokens": input_token_count,
            "output_tokens": output_token_count,
            "total_tokens": total_token_count,
        }
    )

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("Options")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.write(f"Model: {MODEL_NAME}")
