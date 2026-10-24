# BIFS 614 Interactive Tutor only
# This script implements an interactive tutor for BIFS 614
# utilizes a language model and embeddings to provide explanations based on course excerpts.

import pickle
import re
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

print("Loading Phi-3...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
model.eval()
stop_id = tokenizer.convert_tokens_to_ids("<|end|>")
if not isinstance(stop_id, int) or stop_id == tokenizer.unk_token_id:
    stop_id = tokenizer.eos_token_id
print("Device:", model.device)

print("Loading embeddings and course index...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
index_path = Path("knowledge.index")
chunks_path = Path("chunks.pkl")
if not index_path.exists() or not chunks_path.exists():
    raise FileNotFoundError("Run from repository root with knowledge.index and chunks.pkl present.")
index = faiss.read_index(str(index_path))
with chunks_path.open("rb") as f:
    chunks = pickle.load(f)
if index.ntotal != len(chunks):
    raise ValueError("knowledge.index and chunks.pkl do not match; rebuild the index.")
print(f"Loaded {len(chunks)} course chunks.")



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


def generate(context, topic, task, previous_teaching=""):
    """Generate a grounded teaching response from retrieved course passages."""
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
    """Remove unsolicited quiz/answer sections, not arbitrary factual sentences."""
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


def lesson_menu():
    print("\nWhat would you like to do next?")
    print("1. Continue learning")
    print("2. Explain that in more detail")
    print("3. Change topic")
    print("4. Return to main menu")
    print("Command: /quit")


def main():
    print("\nWelcome to the BIFS 614 Interactive Tutor!")
    while True:
        print("\nBIFS 614 Interactive Tutor")
        print("1. Learn a topic\n2. Review a topic\n3. Quit")
        mode = input("Choose (1-3): ").strip()
        if mode in ("3", "/quit"):
            print("Goodbye!")
            return
        if mode not in ("1", "2"):
            print("Please choose 1, 2, or 3.")
            continue
        topic = input("What course topic? ").strip()
        if not topic:
            continue
        difficulty = ""
        if mode == "2":
            difficulty = input("What part is confusing? (optional): ").strip()
        context = retrieve(topic + (" " + difficulty if difficulty else ""))
        if context is None:
            print("\nTutor:", FALLBACK)
            continue
        if mode == "1":
            task = ("Explain what this topic is, using the lecture definition or "
                    "description first. Include at most one additional fact that "
                    "the excerpts explicitly support. Use 1-2 short sentences. "
                    "No headings, questions, quizzes, or unrelated examples.")
        else:
            task = (f"Clarify this difficulty: {difficulty or topic}. "
                    "Use only facts in the excerpts, in 1-2 short sentences. "
                    "No headings, questions, or quizzes.")
        explanation = teaching_only(generate(context, topic, task))
        print("\nTutor:", explanation)
        if explanation == FALLBACK:
            continue
        previous_explanations = [explanation]
        while True:
            lesson_menu()
            choice = input("Choose (1-4): ").strip().lower()
            if choice == "/quit":
                return
            if choice == "1":
                task = ("Explain ONE new fact explicitly present in the excerpts "
                        "that was not already taught. Use 1-2 short sentences. "
                        "If no distinct supported fact remains, use the fallback. "
                        "Do not add a heading or question.")
                # Track the most recent teaching steps to discourage repetition.
                history = "\n".join(previous_explanations[-3:])
                response = teaching_only(generate(context, topic, task, previous_teaching=history))
                print("\nTutor:", response)
                if response != FALLBACK:
                    explanation = response
                    previous_explanations.append(response)
            elif choice == "2":
                task = ("Restate the EARLIER TUTOR EXPLANATION more simply, but "
                        "only using information supported by the excerpts. "
                        "Explain the SAME idea; no new facts, analogies, or "
                        "examples. Use 1-2 short sentences, no headings.")
                response = teaching_only(generate(
                    context, topic, task, previous_teaching=explanation))
                print("\nTutor:", response)
                # Keep the original explanation for subsequent clarification.
            elif choice == "3":
                break  # Choose a different topic.
            elif choice == "4":
                break  # Return to main menu.
            else:
                print("Please choose 1 through 4.")


if __name__ == "__main__":
    main()
