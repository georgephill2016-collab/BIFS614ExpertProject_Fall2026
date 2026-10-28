"""BIFS 614 V7 Q&A Tutor — Streamlit adaptation of CourseAgentV7Final.
Run from the folder containing knowledge.index and chunks.pkl:
    streamlit run CourseAgentV7_Streamlit_Corrected.py
"""
import pickle
from pathlib import Path

import faiss
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(page_title="BIFS 614 Tutor", page_icon="🎓", layout="wide")
st.title("🎓 BIFS 614 Question and Answer Tutor")
st.caption("Course-grounded question answering with Phi-3")

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CANDIDATE_K = 5
CONTEXT_K = 2
MAX_DISTANCE = 1.50
NOT_FOUND = "I could not find the answer in the provided context."
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource(show_spinner=False)
def load_models():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
    model.eval()
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    end_token_id = tokenizer.convert_tokens_to_ids("<|end|>")
    return tokenizer, model, embedding_model, end_token_id

@st.cache_resource(show_spinner=False)
def load_knowledge():
    index_path = BASE_DIR / "knowledge.index"
    chunks_path = BASE_DIR / "chunks.pkl"
    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError("Place knowledge.index and chunks.pkl beside this Streamlit script.")
    index = faiss.read_index(str(index_path))
    with chunks_path.open("rb") as file:
        chunks = pickle.load(file)  # Load only your team's trusted chunks.pkl file.
    if index.ntotal != len(chunks):
        raise ValueError("FAISS index and chunks.pkl contain different numbers of entries. Rebuild both together.")
    return index, chunks

with st.spinner("Loading course tutor..."):
    try:
        tokenizer, model, embedding_model, END_TOKEN_ID = load_models()
        index, chunks = load_knowledge()
    except Exception as exc:
        st.error(f"Unable to load the tutor: {exc}")
        st.stop()

def retrieve_context(question):
    query_embedding = embedding_model.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, CANDIDATE_K)
    if indices[0][0] == -1 or float(distances[0][0]) > MAX_DISTANCE:
        return None
    selected_chunks = []
    for distance, idx in zip(distances[0], indices[0]):
        if idx == -1 or distance > MAX_DISTANCE:
            continue
        selected_chunks.append(chunks[idx])
        if len(selected_chunks) >= CONTEXT_K:
            break
    if not selected_chunks:
        return None
    return "\n\n".join(selected_chunks)

def generate_answer(question, context):
    # Exact wording of the final V7 system prompt, including grounding checks.
    final_prompt = f"""
    <|system|>
    You are a BIFS 614 course tutor.

    Your knowledge for this question is limited to the COURSE CONTEXT
    provided below.

    Answer the QUESTION using only claims explicitly supported by the
    COURSE CONTEXT.

    You may summarize or paraphrase the COURSE CONTEXT, but do not
    introduce additional technical terminology, labels, named concepts,
    or examples unless those terms or concepts appear in COURSE CONTEXT.

    Before answering, silently check each sentence against COURSE CONTEXT.
    Remove any statement that cannot be supported by the supplied text.
    Do not mention your checks or the words COURSE CONTEXT in the answer.

    Do not add facts from your pretrained knowledge, even if those
    facts are correct.

    IMPORTANT:
    Every factual claim in your answer must be traceable to information
    stated in the COURSE CONTEXT.

    For definition questions such as "What is X?", use the definition
    or description given in the COURSE CONTEXT. Do not expand the
    definition using outside knowledge.

    For questions asking "why", "how", "what are some examples",
    advantages, disadvantages, or uses, include the relevant supporting
    information found in the COURSE CONTEXT.

    If the COURSE CONTEXT does not contain enough information to answer
    the QUESTION, respond exactly:

    I could not find the answer in the provided context.

    Prefer the terminology used in the lectures. For example, if the
    context describes a tree, call it a tree; do not introduce another
    technical name for it unless that name appears in the context.

    Give a direct answer in 1-3 sentences. Stop when the QUESTION has
    been answered.
    <|end|>

    <|user|>
    COURSE CONTEXT:
    {context}

    QUESTION:
    {question}
    <|end|>

    <|assistant|>
    """
    input_ids = tokenizer.encode(final_prompt, return_tensors="pt").to(model.device)
    input_token_count = input_ids.shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=120,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=END_TOKEN_ID,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0][input_token_count:]
    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    for stop_tag in ["<|user|>", "<|assistant|>", "<|system|>", "<|end|>"]:
        if stop_tag in response:
            response = response.split(stop_tag)[0].strip()
    return response if response else NOT_FOUND

if "v7_messages" not in st.session_state:
    st.session_state.v7_messages = []

if st.button("Clear conversation"):
    st.session_state.v7_messages = []
    st.rerun()

for message in st.session_state.v7_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Student Question: Ask a BIFS 614 question...")
if question and question.strip():
    question = question.strip()
    st.session_state.v7_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(f"**Student Question:** {question}")
    with st.chat_message("assistant"):
        with st.spinner("Preparing tutor response..."):
            context = retrieve_context(question)
            response = NOT_FOUND if context is None else generate_answer(question, context)
        st.markdown(f"**Tutor Response:** {response}")
    # Save display-ready messages for reruns; don't pass history into Phi-3,
    # because the original V7 tutor processes each question independently.
    st.session_state.v7_messages.append({"role": "assistant", "content": f"**Tutor Response:** {response}"})
