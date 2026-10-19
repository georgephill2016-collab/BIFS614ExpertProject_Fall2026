import streamlit as st
import torch
import faiss
import pickle
import time

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# ============================================================
# Page Setup
# ============================================================

st.set_page_config(
    page_title="BIFS 614 Tutor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 BIFS 614 Tutor Mode")
st.caption("Course Grounded Question Answering with Phi-3")

# ============================================================
# Constants
# ============================================================

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

CANDIDATE_K = 5
CONTEXT_K = 2
MAX_DISTANCE = 1.50

# ============================================================
# Load Models Once
# ============================================================

@st.cache_resource
def load_models():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto"
    )

    model.eval()

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    end_token_id = tokenizer.convert_tokens_to_ids("<|end|>")

    return (
        tokenizer,
        model,
        embedding_model,
        end_token_id
    )


@st.cache_resource
def load_knowledge():

    index = faiss.read_index("knowledge.index")

    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


# ============================================================
# Initialize
# ============================================================

with st.spinner("Loading models..."):

    (
        tokenizer,
        model,
        embedding_model,
        END_TOKEN_ID
    ) = load_models()

    index, chunks = load_knowledge()

# ============================================================
# Sidebar Diagnostics
# ============================================================

st.sidebar.header("System")

if torch.cuda.is_available():
    st.sidebar.success(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )
else:
    st.sidebar.warning("Running on CPU")

st.sidebar.write(f"Knowledge Chunks: {len(chunks)}")
st.sidebar.write(f"Max Distance: {MAX_DISTANCE}")

# ============================================================
# Retrieval Function
# ============================================================

def retrieve_context(question):

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(
        query_embedding,
        CANDIDATE_K
    )

    best_distance = float(distances[0][0])

    if best_distance > MAX_DISTANCE:
        return {
            "context": None,
            "indices": indices,
            "distances": distances,
            "best_distance": best_distance
        }

    selected_chunks = []
    selected_indices = []
    selected_distances = []

    for distance, idx in zip(
        distances[0],
        indices[0]
    ):

        if idx == -1:
            continue

        if distance > MAX_DISTANCE:
            continue

        selected_chunks.append(chunks[idx])
        selected_indices.append(int(idx))
        selected_distances.append(float(distance))

        if len(selected_chunks) >= CONTEXT_K:
            break

    if not selected_chunks:

        return {
            "context": None,
            "indices": indices,
            "distances": distances,
            "best_distance": best_distance
        }

    return {
        "context": "\n\n".join(selected_chunks),
        "indices": selected_indices,
        "distances": selected_distances,
        "best_distance": best_distance
    }

# ============================================================
# Generate Response
# ============================================================

def generate_answer(question, context):

    final_prompt = f"""
<|system|>
You are a BIFS 614 course tutor.

Your knowledge for this question is limited to the COURSE CONTEXT provided below.

Answer the QUESTION using only claims explicitly supported by the COURSE CONTEXT.

You may summarize or paraphrase the COURSE CONTEXT.

Do not add facts from your pretrained knowledge.

If the COURSE CONTEXT does not contain enough information to answer the QUESTION, respond exactly:

I could not find the answer in the provided context.

Give a direct and concise answer.
<|end|>

<|user|>
COURSE CONTEXT:

{context}

QUESTION:

{question}
<|end|>

<|assistant|>
"""

    input_ids = tokenizer.encode(
        final_prompt,
        return_tensors="pt"
    ).to(model.device)

    input_token_count = input_ids.shape[1]

    with torch.no_grad():

        output_ids = model.generate(
            input_ids,
            max_new_tokens=120,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=END_TOKEN_ID,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = output_ids[0][input_token_count:]

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    ).strip()

    for tag in [
        "<|user|>",
        "<|assistant|>",
        "<|system|>",
        "<|end|>"
    ]:

        if tag in response:
            response = response.split(tag)[0].strip()

    if not response:
        response = (
            "I could not find the answer in the provided context."
        )

    return response

# ============================================================
# Chat History
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ============================================================
# User Input
# ============================================================

prompt = st.chat_input(
    "Ask a BIFS 614 question..."
)

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Searching course material..."):

            retrieval_start = time.time()

            retrieval = retrieve_context(prompt)

            retrieval_time = (
                time.time() - retrieval_start
            )

        context = retrieval["context"]

        if context is None:

            response = (
                "I could not find the answer in the provided context."
            )

            st.warning(response)

        else:

            generation_start = time.time()

            response = generate_answer(
                prompt,
                context
            )

            generation_time = (
                time.time() - generation_start
            )

            st.markdown(response)

            with st.expander(
                "Retrieval Diagnostics"
            ):

                st.write(
                    f"Best Distance: "
                    f"{retrieval['best_distance']:.4f}"
                )

                st.write(
                    f"Retrieval Time: "
                    f"{retrieval_time:.2f}s"
                )

                st.write(
                    f"Generation Time: "
                    f"{generation_time:.2f}s"
                )

                st.write(
                    "Chunk Indices:",
                    retrieval["indices"]
                )

                st.write(
                    "Distances:",
                    retrieval["distances"]
                )

            with st.expander(
                "Retrieved Context"
            ):
                st.text(context)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )
