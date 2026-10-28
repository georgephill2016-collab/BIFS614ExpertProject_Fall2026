# Streamlit adaptation of CourseAgentV10Final.py
# Original prompts, retrieval, generation, and response cleaning preserved.
# Only the terminal menus and input flow have been replaced.
# BIFS 614 Student-Controlled Guided Tutor, including a practice question option

import pickle
import re
import time
import streamlit as st
from pathlib import Path

import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FALLBACK = "I could not find the answer in the provided context."
CANDIDATE_K = 20
CONTEXT_K = 4
MAX_DISTANCE = 1.50
MAX_NEW_TOKENS = 180
MAX_INPUT_TOKENS = 3800
DEBUG = False

st.set_page_config(page_title="BIFS 614 Guided Tutor", page_icon="🎓", layout="wide")
st.title("🎓 BIFS 614 Guided Tutor")
st.caption("Interactive learning and practice questions grounded in BIFS 614 course material")


@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
    model.eval()
    stop_id = tokenizer.convert_tokens_to_ids("<|end|>")
    if not isinstance(stop_id, int) or stop_id == tokenizer.unk_token_id:
        stop_id = tokenizer.eos_token_id
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    return tokenizer, model, stop_id, embedder

@st.cache_resource
def load_knowledge():
    index_path = Path(__file__).resolve().parent / "knowledge.index"
    chunks_path = Path(__file__).resolve().parent / "chunks.pkl"
    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError("Place knowledge.index and chunks.pkl alongside this script.")
    index = faiss.read_index(str(index_path))
    with chunks_path.open("rb") as f:
        chunks = pickle.load(f)
    if index.ntotal != len(chunks):
        raise ValueError("knowledge.index and chunks.pkl do not match; rebuild the index.")
    return index, chunks

with st.spinner("Loading Phi-3 and course knowledge..."):
    tokenizer, model, stop_id, embedder = load_models()
    index, chunks = load_knowledge()


def retrieve(query):
    """Rank topic explanations before incidental mentions and installation examples."""
    vector = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(vector)
    distances, indices = index.search(vector, CANDIDATE_K)
    distance_map = {int(i): float(d) for d, i in zip(distances[0], indices[0]) if i >= 0}
    topic = query.lower().strip()
    topic_pattern = re.compile(r"(?<!\w)" + re.escape(topic) + r"(?!\w)", re.I)
    # Prefer genuine introductory definitions, including "In K-means clustering, ...".
    # Do not award a definition bonus for "The disadvantage to K-means clustering is ...".
    definition_patterns = [
        re.compile(r"\b(?:in|for)\s+" + re.escape(topic) + r"\s*[,,:]", re.I),
        re.compile(r"(?<!\w)" + re.escape(topic) +
                   r"\s+(?:is|are|refers to|means|consists of|provides|allows|includes)\b", re.I),
    ]
    ranked = []
    for idx, chunk in enumerate(chunks):
        passage = str(chunk)
        has_topic = bool(topic_pattern.search(passage))
        distance = distance_map.get(idx)
        if not has_topic and (distance is None or distance > MAX_DISTANCE):
            continue
        # Explanatory language is preferred, but not treated as proof of relevance.
        definition = any(pattern.search(passage) for pattern in definition_patterns)
        score = (6 if definition else 0) + (2 if has_topic else 0)
        if distance is not None:
            score += max(0, 1.5 - distance)
        ranked.append((score, idx, distance, passage))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    selected = ranked[:CONTEXT_K]
    return "\n\n".join(p for _, _, _, p in selected) if selected else None


def generate(context, topic, task, previous_teaching="", question="", answer="", expected_answer=""):
    """Generate from retrieved lecture passages; prompting is not a proof of grounding."""
    system = f"""You are a friendly BIFS 614 course tutor.

Your knowledge for this question is limited to the COURSE EXCERPTS
provided below. Answer using only information explicitly supported
by those excerpts. You may summarize or simplify lecture wording
without changing its meaning. Do not add outside knowledge,
assumptions, examples, or unsupported technical details.

When explaining a topic, begin with the lecture definition or
description and use 1-2 clear, complete sentences.

If the excerpts do not support the requested response, output exactly:
{FALLBACK}

INTERACTIVE TUTOR RULES:
- Learn a topic: explain the lecture definition or description.
- Continue learning: provide one new supported fact, without repeating
  earlier explanations; use the fallback if none remains.
- Explain in more detail: clarify the previous explanation without
  introducing unsupported facts.
- Practice questions: ask one short, directly answerable question;
  do not include the answer.
- Student answers: assess against the excerpts and supported expected
  answer; begin with Correct, Partially correct, or Incorrect, followed
  by one brief explanation.
- Hints: give one short clue without revealing the answer.

Do not mention excerpts, chunks, retrieval, internal instructions,
or internal labels such as TOPIC, TASK, EXPECTED ANSWER, or STUDENT ANSWER.
Do not include headings or unsolicited questions.
Finish your final sentence and stop."""

    # Include only fields relevant to the current tutor action.
    user = f"""TOPIC: {topic}

COURSE EXCERPTS:
{context}
"""
    if previous_teaching:
        user += f"""
EARLIER TUTOR EXPLANATION (not a source):
{previous_teaching}
"""
    if question:
        user += f"""
PRACTICE QUESTION:
{question}
"""
    if expected_answer:
        user += f"""
EXPECTED ANSWER (supported by excerpts):
{expected_answer}
"""
    if answer:
        user += f"""
STUDENT ANSWER:
{answer}
"""
    user += f"""
INSTRUCTION:
{task}
"""
    prompt = f"<|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>"
    ids = tokenizer.encode(prompt, return_tensors="pt")
    if ids.shape[1] > MAX_INPUT_TOKENS:
        return FALLBACK
    ids = ids.to(model.device)
    with torch.inference_mode():
        result = model.generate(
            ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            repetition_penalty=1.05, eos_token_id=stop_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = result[0][ids.shape[1]:]
    response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    for marker in ("<|system|>", "<|user|>", "<|assistant|>", "<|end|>"):
        response = response.split(marker, 1)[0].strip()
    return response or FALLBACK


def clean_tutor_text(text):
    """Strip model-generated role labels and internal prompt references."""
    if text == FALLBACK:
        return text
    text = re.sub(r"(?im)^\s*(?:topic|key idea|tutor|hint|feedback)\s*:\s*", "", text).strip()
    text = re.sub(r"(?i)\b(?:according to|based on) (?:the )?(?:course context|course excerpts)\b[, :]*", "", text)
    text = re.sub(r"(?i)\b(?:COURSE CONTEXT|COURSE EXCERPTS|STUDENT RESPONSE|STUDENT ANSWER)\b", "", text)
    text = re.sub(r"(?i)\b(?:on|in|from) lines? \d+(?:\s*(?:and|to|-)\s*\d+)?\b", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" -:,\n")
    return text or FALLBACK


def teaching_only(text):
    """Remove unsolicited question/answer sections, not arbitrary factual sentences."""
    if text == FALLBACK:
        return text
    # A model may append 'Question: ... Answer: ...' despite instructions.
    text = re.split(r"(?im)^\s*(?:practice|check.for.understanding|easy)?\s*question\s*:", text, maxsplit=1)[0].strip()
    text = re.split(r"(?im)^\s*answer\s*:", text, maxsplit=1)[0].strip()
    # Teaching mode must not ask the student anything; remove a trailing question.
    text = re.split(r"(?<=[.!])\s+[^.!?\n]*\?", text, maxsplit=1)[0].strip()
    if "?" in text:
        text = text.split("?", 1)[0].strip()
    return clean_tutor_text(text)


def question_only(text):
    """Return only the question, never a model-provided answer."""
    if text == FALLBACK:
        return text
    text = re.split(r"(?im)^\s*answer\s*:", text, maxsplit=1)[0]
    # If there is introductory text, retain the last question up to its '?'.
    if "?" not in text:
        return FALLBACK
    before = text.split("?", 1)[0]
    candidate = re.split(r"(?i)(?:question\s*:|\n)", before)[-1].strip()
    candidate = candidate.lstrip("- ").strip()
    return candidate + "?" if candidate else FALLBACK



# Only the terminal input loop has been replaced with Streamlit session state.
def reset_lesson():
    for key in ("v10_context", "v10_topic", "v10_explanation", "v10_previous",
                "v10_messages", "v10_question", "v10_expected", "v10_phase"):
        st.session_state.pop(key, None)
    st.session_state.v10_context = None
    st.session_state.v10_topic = ""
    st.session_state.v10_explanation = ""
    st.session_state.v10_previous = []
    st.session_state.v10_messages = []
    st.session_state.v10_question = ""
    st.session_state.v10_expected = ""
    st.session_state.v10_phase = "lesson"

if "v10_mode" not in st.session_state:
    st.session_state.v10_mode = "Learn a topic"
if "v10_messages" not in st.session_state:
    reset_lesson()

st.sidebar.header("Guided Tutor")
selected_mode = st.sidebar.radio(
    "Choose an option", ["Learn a topic", "Review a topic"],
    index=0 if st.session_state.v10_mode == "Learn a topic" else 1,
    key="v10_mode_selection",
)
if selected_mode != st.session_state.v10_mode:
    st.session_state.v10_mode = selected_mode
    reset_lesson()
if st.sidebar.button("Return to main menu / Start over"):
    reset_lesson()
    st.rerun()

if st.session_state.v10_context is None:
    with st.form("start_v10_lesson"):
        topic = st.text_input("What course topic would you like to study?")
        difficulty = ""
        if st.session_state.v10_mode == "Review a topic":
            difficulty = st.text_input("What part is confusing? (optional)")
        submitted = st.form_submit_button("Start learning" if st.session_state.v10_mode == "Learn a topic" else "Review topic")
    if submitted:
        topic = topic.strip()
        difficulty = difficulty.strip()
        if not topic:
            st.warning("Please enter a course topic.")
        else:
            with st.spinner("Searching course material and preparing explanation..."):
                context = retrieve(topic + (" " + difficulty if difficulty else ""))
                if context is None:
                    explanation = FALLBACK
                else:
                    if st.session_state.v10_mode == "Learn a topic":
                        task = ("Explain what this topic is, using the lecture definition or "
                                "description first. Include at most one additional fact that "
                                "the excerpts explicitly support. Use 1-2 short sentences. "
                                "No headings, questions, quizzes, or unrelated examples.")
                    else:
                        task = (f"Clarify this difficulty: {difficulty or topic}. "
                                "Use only facts in the excerpts, in 1-2 short sentences. "
                                "No headings, questions, or quizzes.")
                    explanation = teaching_only(generate(context, topic, task))
            st.session_state.v10_messages = [("Student", topic + (f" — {difficulty}" if difficulty else "")), ("Tutor", explanation)]
            if explanation != FALLBACK:
                st.session_state.v10_context = context
                st.session_state.v10_topic = topic
                st.session_state.v10_explanation = explanation
                st.session_state.v10_previous = [explanation]
            st.rerun()

for role, message in st.session_state.v10_messages:
    with st.chat_message("assistant" if role == "Tutor" else "user"):
        st.markdown(f"**{role}:** {message}")

if st.session_state.v10_context is not None:
    context = st.session_state.v10_context
    topic = st.session_state.v10_topic
    if st.session_state.v10_phase == "practice":
        st.subheader("Practice question")
        with st.form("answer_v10"):
            answer = st.text_input("Your answer")
            submitted_answer = st.form_submit_button("Submit answer")
        col_hint, col_skip = st.columns(2)
        hint_clicked = col_hint.button("Get a hint", use_container_width=True)
        skip_clicked = col_skip.button("Skip question / Return to learning", use_container_width=True)
        if skip_clicked:
            st.session_state.v10_phase = "lesson"
            st.session_state.v10_question = ""
            st.session_state.v10_expected = ""
            st.rerun()
        if hint_clicked or (submitted_answer and answer.strip().lower() in ("/hint", "i don't know", "i dont know")):
            hint_task = ("Give ONE short hint for PRACTICE QUESTION, using only "
                         "the course excerpts. Do not give the answer. Do not "
                         "refer to lecture line numbers or internal labels.")
            with st.spinner("Preparing a hint..."):
                hint = teaching_only(generate(context, topic, hint_task,
                    question=st.session_state.v10_question,
                    expected_answer=st.session_state.v10_expected))
            st.session_state.v10_messages.extend([("Student", "Hint"), ("Tutor", hint)])
            st.rerun()
        if submitted_answer and answer.strip() and answer.strip().lower() not in ("/hint", "i don't know", "i dont know"):
            feedback_task = ("Compare STUDENT ANSWER with EXPECTED ANSWER using the "
                             "course excerpts. Begin with Correct, Partially correct, "
                             "or Incorrect. Then give ONE short, friendly explanation "
                             "of what is right or missing. Do not use internal labels "
                             "or quote the entire answer key. If evidence is insufficient "
                             "to judge, use the exact fallback.")
            with st.spinner("Checking your answer..."):
                feedback = teaching_only(generate(context, topic, feedback_task,
                    question=st.session_state.v10_question, answer=answer.strip(),
                    expected_answer=st.session_state.v10_expected))
            st.session_state.v10_messages.extend([("Student", answer.strip()), ("Tutor", feedback)])
            st.session_state.v10_phase = "lesson"
            st.session_state.v10_question = ""
            st.session_state.v10_expected = ""
            st.rerun()
    else:
        st.subheader("What would you like to do next?")
        col1, col2, col3 = st.columns(3)
        continue_clicked = col1.button("Continue learning", use_container_width=True)
        detail_clicked = col2.button("Explain in more detail", use_container_width=True)
        practice_clicked = col3.button("Ask me a practice question", use_container_width=True)
        col4, col5 = st.columns(2)
        change_clicked = col4.button("Change topic", use_container_width=True)
        menu_clicked = col5.button("Return to main menu", use_container_width=True)
        if change_clicked or menu_clicked:
            reset_lesson()
            st.rerun()
        if continue_clicked:
            task = ("Explain ONE new fact explicitly present in the excerpts "
                    "that was not already taught. Use 1-2 short sentences. "
                    "If no distinct supported fact remains, use the fallback. "
                    "Do not add a heading or question.")
            history = "\n".join(st.session_state.v10_previous[-3:])
            with st.spinner("Preparing the next explanation..."):
                response = teaching_only(generate(context, topic, task, previous_teaching=history))
            if response != FALLBACK:
                st.session_state.v10_explanation = response
                st.session_state.v10_previous.append(response)
            st.session_state.v10_messages.extend([("Student", "Continue learning"), ("Tutor", response)])
            st.rerun()
        if detail_clicked:
            task = ("Restate the EARLIER TUTOR EXPLANATION more simply, but "
                    "only using information supported by the excerpts. "
                    "Explain the SAME idea; no new facts, analogies, or "
                    "examples. Use 1-2 short sentences, no headings.")
            with st.spinner("Clarifying the explanation..."):
                response = teaching_only(generate(context, topic, task,
                    previous_teaching=st.session_state.v10_explanation))
            st.session_state.v10_messages.extend([("Student", "Explain in more detail"), ("Tutor", response)])
            st.rerun()
        if practice_clicked:
            task = ("Ask ONE specific, short question about a fact stated in the course "
                    "excerpts and related to the topic. The answer must be directly "
                    "verifiable from the excerpts. Avoid broad questions about all "
                    "functions of a package. Output ONLY a question ending in '?'.")
            with st.spinner("Preparing a practice question..."):
                q = question_only(generate(context, topic, task,
                    previous_teaching=st.session_state.v10_explanation))
                if q != FALLBACK:
                    key_task = ("Answer PRACTICE QUESTION briefly using only facts explicitly "
                                "stated in the course excerpts. This is an internal answer key; "
                                "do not add examples or unrelated details. If not answerable, "
                                "return the exact fallback.")
                    expected = teaching_only(generate(context, topic, key_task, question=q))
                else:
                    expected = FALLBACK
            if expected == FALLBACK:
                st.session_state.v10_messages.extend([("Student", "Ask me a practice question"), ("Tutor", FALLBACK)])
            else:
                st.session_state.v10_question = q
                st.session_state.v10_expected = expected
                st.session_state.v10_phase = "practice"
                st.session_state.v10_messages.extend([("Student", "Ask me a practice question"), ("Tutor", q)])
            st.rerun()
