import pickle
import re
import time
from pathlib import Path

import faiss
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

FALLBACK = "I could not find the answer in the provided context."
CANDIDATE_K = 20
CONTEXT_K = 4
MAX_DISTANCE = 1.50
MAX_NEW_TOKENS = 180
MAX_INPUT_TOKENS = 3800

# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def load_resources():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto"
    )
    model.eval()

    stop_id = tokenizer.convert_tokens_to_ids("<|end|>")
    if not isinstance(stop_id, int):
        stop_id = tokenizer.eos_token_id

    embedder = SentenceTransformer(EMBEDDING_MODEL)

    index = faiss.read_index("knowledge.index")

    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    return (
        tokenizer,
        model,
        stop_id,
        embedder,
        index,
        chunks
    )


(
    tokenizer,
    model,
    stop_id,
    embedder,
    index,
    chunks
) = load_resources()

# --------------------------------------------------
# RETRIEVAL
# --------------------------------------------------

def retrieve(query):

    vector = embedder.encode(
        [query],
        convert_to_numpy=True
    )

    faiss.normalize_L2(vector)

    distances, indices = index.search(
        vector,
        CANDIDATE_K
    )

    distance_map = {
        int(i): float(d)
        for d, i in zip(distances[0], indices[0])
        if i >= 0
    }

    topic = query.lower().strip()

    topic_pattern = re.compile(
        r"(?<!\w)" + re.escape(topic) + r"(?!\w)",
        re.I
    )

    ranked = []

    for idx, chunk in enumerate(chunks):

        passage = str(chunk)

        has_topic = bool(
            topic_pattern.search(passage)
        )

        distance = distance_map.get(idx)

        if not has_topic and (
            distance is None or distance > MAX_DISTANCE
        ):
            continue

        score = (2 if has_topic else 0)

        if distance is not None:
            score += max(
                0,
                1.5 - distance
            )

        ranked.append(
            (score, passage)
        )

    ranked.sort(
        key=lambda x: -x[0]
    )

    selected = ranked[:CONTEXT_K]

    if not selected:
        return None

    return "\n\n".join(
        p for _, p in selected
    )

# --------------------------------------------------
# LLM GENERATION
# --------------------------------------------------

def generate(context, topic, task):

    system = f"""
You are a friendly BIFS 614 tutor.

Use ONLY information in COURSE EXCERPTS.

If unsupported output exactly:

{FALLBACK}
"""

    user = f"""
TOPIC: {topic}

COURSE EXCERPTS:

{context}

INSTRUCTION:

{task}
"""

    prompt = (
        f"<|system|>\n{system}<|end|>\n"
        f"<|user|>\n{user}<|end|>\n"
        f"<|assistant|>"
    )

    ids = tokenizer.encode(
        prompt,
        return_tensors="pt"
    )

    if ids.shape[1] > MAX_INPUT_TOKENS:
        return FALLBACK

    ids = ids.to(model.device)

    with torch.inference_mode():

        output = model.generate(
            ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=stop_id,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        output[0][ids.shape[1]:],
        skip_special_tokens=True
    ).strip()

    return response if response else FALLBACK

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if "context" not in st.session_state:
    st.session_state.context = None

# --------------------------------------------------
# UI
# --------------------------------------------------

st.set_page_config(
    page_title="BIFS 614 Tutor",
    layout="wide"
)

st.title("🎓 BIFS 614 Guided Tutor")

mode = st.radio(
    "Select Mode",
    [
        "Learn a Topic",
        "Review a Topic"
    ]
)

topic = st.text_input(
    "Course Topic"
)

difficulty = ""

if mode == "Review a Topic":
    difficulty = st.text_input(
        "What part is confusing?"
    )

# --------------------------------------------------
# INITIAL LESSON
# --------------------------------------------------

if st.button("Generate Lesson"):

    with st.spinner(
        "Searching knowledge base..."
    ):

        query = topic

        if difficulty:
            query += " " + difficulty

        context = retrieve(query)

        st.session_state.context = context

        if context is None:

            st.error(FALLBACK)

        else:

            if mode == "Learn a Topic":

                task = (
                    "Explain what this topic is "
                    "using the lecture definition first."
                )

            else:

                task = (
                    f"Clarify this difficulty: {difficulty}"
                )

            answer = generate(
                context,
                topic,
                task
            )

            st.session_state.history.append(
                answer
            )

            st.success(answer)

# --------------------------------------------------
# FOLLOW-UP ACTIONS
# --------------------------------------------------

if st.session_state.context:

    st.markdown("---")
    st.subheader("Learning Options")

    col1, col2, col3 = st.columns(3)

    with col1:

        if st.button("Continue Learning"):

            task = (
                "Explain one new fact from the excerpts."
            )

            answer = generate(
                st.session_state.context,
                topic,
                task
            )

            st.session_state.history.append(
                answer
            )

            st.info(answer)

    with col2:

        if st.button("More Detail"):

            task = (
                "Explain in more detail using only course excerpts."
            )

            answer = generate(
                st.session_state.context,
                topic,
                task
            )

            st.session_state.history.append(
                answer
            )

            st.info(answer)

    with col3:

        if st.button("Practice Question"):

            task = (
                "Ask one short practice question."
            )

            question = generate(
                st.session_state.context,
                topic,
                task
            )

            st.session_state.practice_question = question

# --------------------------------------------------
# PRACTICE
# --------------------------------------------------

if "practice_question" in st.session_state:

    st.markdown("---")
    st.subheader("Practice")

    st.write(st.session_state.practice_question)

    student_answer = st.text_area(
        "Your Answer"
    )

    if st.button("Submit Answer"):

        task = (
            f"Evaluate this answer:\n"
            f"{student_answer}"
        )

        feedback = generate(
            st.session_state.context,
            topic,
            task
        )

        st.success(feedback)

# --------------------------------------------------
# HISTORY
# --------------------------------------------------

if st.session_state.history:

    st.markdown("---")
    st.subheader("Session History")

    for i, item in enumerate(
        st.session_state.history,
        start=1
    ):
        st.write(f"**{i}.** {item}")
