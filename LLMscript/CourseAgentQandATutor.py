from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import torch
import time

# ============================================================
# Course Agent Question and Answer Tutor Mode

# This script is for the Question and Answer Version of Tutor Mode
# It is designed to use only course content contained in knowledge.txt from the lecture material
# The student asks a question related to the course material in "Student Question:" Prompt
# The "Tutor Response:" answers students question with relevant course material
# The Input Tokens, Generation Time, Candidate Distances and Indices,
# and Retrieved Context Chunks are all hidden from the student in comments 

# Goal:
# Improve retrieval grounding and prevent Phi-3 from answering
# questions that are unsupported by the BIFS 614 course content.
#
# FAISS searches the five nearest candidate chunks. The two
# closest qualifying chunks are passed to Phi-3 as course
# context.
#
# Questions whose best retrieved chunk exceeds the maximum
# allowed distance are rejected before being sent to Phi-3.
# ============================================================

# print(
#     f"GPU: {torch.cuda.get_device_name(0)}"
#     if torch.cuda.is_available()
#     else "No GPU"
# )

# ============================================================
# Model
# ============================================================

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

print("Loading Phi-3 model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto"
)

model.eval()

print("Model loaded successfully.")
# print(f"Model device: {model.device}")

# Phi-3 uses <|end|> to mark the end of a turn.
END_TOKEN_ID = tokenizer.convert_tokens_to_ids("<|end|>")


# ============================================================
# Embedding Model
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ============================================================
# Load FAISS Vector Database
# ============================================================

index = faiss.read_index("knowledge.index")

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# print(f"Knowledge base loaded: {len(chunks)} chunks.")


# ============================================================
# Retrieval Settings
# ============================================================

# Retrieve a broader set of candidate chunks from FAISS.
CANDIDATE_K = 5

# Number of chunks actually passed to Phi-3.
CONTEXT_K = 2

# If even the best candidate exceeds this distance,
# reject the question as unrelated to the course material.
MAX_DISTANCE = 1.50


# ============================================================
# Retrieval Function
# ============================================================

def retrieve_context(question):

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    # Must match normalization used when building the index.
    faiss.normalize_L2(query_embedding)

    # Retrieve a broader candidate set.
    distances, indices = index.search(
        query_embedding,
        CANDIDATE_K
    )

# Candidate Indices: the IDs of the 5 chunks FAISS found that are most similar to the students question
    # They are basically locations inside chunks.pkl
# Candidate Distances: Show how semantically close each candidate is to the question. Lower is better

#     print("Candidate Indices:", indices)
#     print("Candidate Distances:", distances)

    best_distance = float(distances[0][0])

# Best Distance and Maximum Allowed Distance are the retrieval guardrail
# The max allowed distance is 1.50, so if the best distance is below that, then the retrieval guardrail passes

#     print(f"Best Distance: {best_distance:.4f}")
#     print(f"Maximum Allowed Distance: {MAX_DISTANCE:.4f}")

    # --------------------------------------------------------
    # Retrieval Guardrail
    # --------------------------------------------------------
# This is the important addition from V6, so that when we ask an unrelated question like "Why is the sky blue?"
# The closest chunk is over 1.50 for max distance, never gets passed to Phi-3, and we get the correct prompt returned

    # If the closest chunk is still too far away,
    # consider the question unsupported by the course content.
    if best_distance > MAX_DISTANCE:

#         print("Retrieval Guardrail: REJECTED")
        return None

    # --------------------------------------------------------
    # Context Selection
    # --------------------------------------------------------

    selected_chunks = []
    selected_indices = []
    selected_distances = []

    for distance, idx in zip(distances[0], indices[0]):

        if idx == -1:
            continue

        if distance > MAX_DISTANCE:
            continue

        selected_chunks.append(chunks[idx])
        selected_indices.append(int(idx))
        selected_distances.append(float(distance))

        # Stop once enough evidence has been collected.
        if len(selected_chunks) >= CONTEXT_K:
            break

    if not selected_chunks:

#         print("Retrieval Guardrail: REJECTED")
        return None

#     print(
#         f"Retrieval Guardrail: PASSED "
#         f"({len(selected_chunks)} selected chunk(s))"
#     )

#     print("Selected Indices:", selected_indices)

#     print(
#         "Selected Distances:",
#         [round(d, 4) for d in selected_distances]
#     )

    return "\n\n".join(selected_chunks)


# ============================================================
# Interactive Tutor Loop
# ============================================================

print("\nBIFS 614 Question and Answer Tutor")
# print("Adaptive Retrieval Test")  # Developer-only message
print("Type 'quit', 'exit', or 'q' to stop.\n")

while True:

    user_prompt = input("Student Question: ").strip()

    if user_prompt.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    if not user_prompt:
        continue


    # ========================================================
    # Retrieve Course Content
    # ========================================================

    start = time.time()

    context = retrieve_context(user_prompt)

    retrieval_time = time.time() - start

#     print("Retrieval:", retrieval_time, "seconds")


    # ========================================================
    # Retrieval Guardrail
    # ========================================================

    # If no sufficiently relevant course content was retrieved,
    # do NOT send the question to Phi-3.
    #
    # This prevents Phi-3 from answering an unrelated question
    # using its pretrained outside knowledge.
    if context is None:

        response = (
            "I could not find the answer in the provided context."
        )

        print("Tutor Response:", response)
        continue


#     print(
#         "Retrieved course context chars:",
#         len(context)
#     )


    # ========================================================
    # Diagnostic: Display Retrieved Context
    # ========================================================

    # Keep this visible during testing so we can compare the
    # retrieved material with Phi-3's generated answer.
#     print("\n========== RETRIEVED CONTEXT ==========")
#     print(context)
#     print("========== END RETRIEVED CONTEXT ==========\n")


    # ========================================================
    # Phi-3 Prompt
    # ============================================================

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
    {user_prompt}
    <|end|>

    <|assistant|>
    """


    # ========================================================
    # Tokenize Prompt
    # ========================================================

    input_ids = tokenizer.encode(
        final_prompt,
        return_tensors="pt"
    ).to(model.device)

    input_token_count = input_ids.shape[1]

#     print(f"Input Tokens: {input_token_count}")


    # ========================================================
    # Generate Answer
    # ========================================================

    start = time.time()

    with torch.no_grad():

        output_ids = model.generate(
            input_ids,

            # Allow enough room for a short course answer while
            # discouraging unnecessary long responses.
            max_new_tokens=120,

            # Deterministic generation is preferable for
            # grounded course question answering.
            do_sample=False,

            # Slightly discourage repetitive generation.
            repetition_penalty=1.05,

            # Allow Phi-3's end-of-turn token to stop generation.
            eos_token_id=END_TOKEN_ID,

            pad_token_id=tokenizer.eos_token_id
        )

    generation_time = time.time() - start

#     print("Generation:", generation_time, "seconds")


    # ========================================================
    # Extract Generated Tokens Only
    # ========================================================

    generated_ids = output_ids[0][input_token_count:]

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    response = response.strip()


    # ========================================================
    # Remove Any Unexpected Role Tags
    # ========================================================

    for stop_tag in [
        "<|user|>",
        "<|assistant|>",
        "<|system|>",
        "<|end|>"
    ]:

        if stop_tag in response:
            response = response.split(stop_tag)[0].strip()


    # ========================================================
    # Empty Response Guardrail
    # ========================================================

    if not response:
        response = (
            "I could not find the answer in the provided context."
        )


    # ========================================================
    # Token Statistics
    # ========================================================

    output_token_count = len(generated_ids)

    total_token_count = (
        input_token_count + output_token_count
    )


    # ========================================================
    # Display Answer
    # ========================================================

    print("Tutor Response:", response)

#     print(f"Input Tokens:  {input_token_count}")
#     print(f"Output Tokens: {output_token_count}")
#     print(f"Total Tokens:  {total_token_count}")
