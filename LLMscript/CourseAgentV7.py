from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import torch
import time

# ============================================================
# CourseAgentV7-9
# BIFS 614 Tutor Mode
#
# V7-9 Goal:
# Test adaptive retrieval instead of always sending a fixed
# number of retrieved chunks to Phi-3.
#
# FAISS searches the five nearest candidate chunks, but only
# chunks sufficiently close to the best match are included in
# the final course context.
#
# This version uses the existing knowledge.index and chunks.pkl.
# No rebuild of the FAISS index is required.
# ============================================================

print(
    f"GPU: {torch.cuda.get_device_name(0)}"
    if torch.cuda.is_available()
    else "No GPU"
)

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
print(f"Model device: {model.device}")

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

print(f"Knowledge base loaded: {len(chunks)} chunks.")


# ============================================================
# Retrieval Settings
# ============================================================

# Ask FAISS for the five nearest candidate chunks.
# These are candidates only; all five are NOT automatically
# sent to Phi-3.
CANDIDATE_K = 5

# Absolute maximum FAISS L2 distance allowed.
# If the best result is farther away than this, reject the
# question before sending anything to Phi-3.
MAX_DISTANCE = 1.50

# Additional chunks must be reasonably close to the best
# retrieved chunk.
#
# Example:
# Best distance = 0.70
# Adaptive limit = 0.70 + 0.25 = 0.95
#
# Only candidate chunks with distances <= 0.95 are accepted.
DISTANCE_MARGIN = 0.25


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

    print("Candidate Indices:", indices)
    print("Candidate Distances:", distances)

    best_distance = float(distances[0][0])

    print(f"Best Distance: {best_distance:.4f}")
    print(f"Maximum Allowed Distance: {MAX_DISTANCE:.4f}")

    # --------------------------------------------------------
    # Retrieval Guardrail
    # --------------------------------------------------------

    # If the closest chunk is still too far away,
    # consider the question unsupported by the course content.
    if best_distance > MAX_DISTANCE:

        print("Retrieval Guardrail: REJECTED")
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

        print("Retrieval Guardrail: REJECTED")
        return None

    print(
        f"Retrieval Guardrail: PASSED "
        f"({len(selected_chunks)} selected chunk(s))"
    )

    print("Selected Indices:", selected_indices)

    print(
        "Selected Distances:",
        [round(d, 4) for d in selected_distances]
    )

    return "\n\n".join(selected_chunks)


# ============================================================
# Interactive Tutor Loop
# ============================================================

print("\nBIFS 614 Tutor Mode - V7-9")
print("Adaptive Retrieval Test")
print("Type 'quit', 'exit', or 'q' to stop.\n")

while True:

    user_prompt = input("You: ").strip()

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

    print("Retrieval:", retrieval_time, "seconds")


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

        print("Response:", response)
        continue


    print(
        "Retrieved course context chars:",
        len(context)
    )


    # ========================================================
    # Diagnostic: Display Retrieved Context
    # ========================================================

    # Keep this visible during testing so we can compare the
    # retrieved material with Phi-3's generated answer.
    print("\n========== RETRIEVED CONTEXT ==========")
    print(context)
    print("========== END RETRIEVED CONTEXT ==========\n")


    # ========================================================
    # Phi-3 Prompt
    # ============================================================

    final_prompt = final_prompt = f"""
<|system|>
You are a BIFS 614 course tutor.

Your job is to answer the QUESTION using only the information
explicitly stated in the COURSE CONTEXT.

GROUNDING RULES:

1. Use only facts, definitions, descriptions, examples, and
   explanations that are explicitly stated in the COURSE CONTEXT.

2. Do not add facts from your pretrained knowledge, even if those
   facts are correct or seem logically related to the topic.

3. Do not introduce a factual claim merely because it can be
   inferred from the COURSE CONTEXT. If the COURSE CONTEXT does
   not explicitly support a claim, do not include it.

4. You may summarize or paraphrase the COURSE CONTEXT, but your
   answer must preserve the meaning of the information provided.

5. For definition questions such as "What is X?", base the
   definition directly on how X is described in the COURSE CONTEXT.
   Do not replace the course's description with a definition from
   your pretrained knowledge.

6. For questions asking "why", "how", for examples, advantages,
   disadvantages, uses, or other details, include only the relevant
   information explicitly provided in the COURSE CONTEXT.

7. Answer only what the QUESTION asks. Do not provide additional
   examples, applications, background information, or explanations
   unless they are needed to answer the QUESTION.

8. If the COURSE CONTEXT does not contain enough information to
   answer the QUESTION, respond exactly:
   I could not find the answer in the provided context.

Once the QUESTION has been answered, stop.
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

    print(f"Input Tokens: {input_token_count}")


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

    print("Generation:", generation_time, "seconds")


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

    print("Response:", response)

    print(f"Input Tokens:  {input_token_count}")
    print(f"Output Tokens: {output_token_count}")
    print(f"Total Tokens:  {total_token_count}")
