# CourseAgentV7.py

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import torch
import time


# ==========================
# Configuration
# ==========================

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

# Number of nearest chunks FAISS will initially retrieve
TOP_K = 5

# Initial testing threshold for FAISS L2 distance.
# Smaller distance = greater similarity.
#
# IMPORTANT:
# This is a starting value for testing, not a permanently validated cutoff.
# Test multiple course-related and unrelated questions and adjust if needed.
MAX_DISTANCE = 1.50

# Exact response used when relevant course context cannot be found
FALLBACK_RESPONSE = "I could not find the answer in the provided context."


# ==========================
# Device Information
# ==========================

print(
    f"GPU: {torch.cuda.get_device_name(0)}"
    if torch.cuda.is_available()
    else "No GPU"
)


# ==========================
# Load Phi-3 Model
# ==========================

print("Loading Phi-3 model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto"
)

model.eval()

print("Model loaded successfully.")
print(f"Model device: {model.device}")
print(model.hf_device_map)


# ==========================
# Load Embedding Model
# ==========================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ==========================
# Load Vector Database
# ==========================

index = faiss.read_index("knowledge.index")

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print("Knowledge base loaded.")


# ==========================
# Retrieval Function
# ==========================

def retrieve_context(question, top_k=TOP_K):

    # Encode the user's question using the same normalization
    # used when the knowledge index was created.
    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Search FAISS for the nearest chunks.
    distances, indices = index.search(
        query_embedding,
        top_k
    )

    print("Indices:", indices)
    print("Distances:", distances)

    # The first result is the closest match.
    best_distance = float(distances[0][0])

    print(f"Best Distance: {best_distance:.4f}")
    print(f"Maximum Allowed Distance: {MAX_DISTANCE:.4f}")

    # --------------------------------------------------
    # Guardrail 1:
    # If even the best chunk is too far away from the
    # question, do not send the question to Phi-3.
    # --------------------------------------------------

    if best_distance > MAX_DISTANCE:
        print("Retrieval Guardrail: REJECTED")
        return None

    # --------------------------------------------------
    # Keep only chunks that individually pass the
    # relevance threshold.
    # --------------------------------------------------

    retrieved_chunks = []

    for distance, idx in zip(distances[0], indices[0]):

        if idx != -1 and distance <= MAX_DISTANCE:
            retrieved_chunks.append(chunks[idx])

    print(
        f"Retrieval Guardrail: PASSED "
        f"({len(retrieved_chunks)} relevant chunk(s))"
    )

    if not retrieved_chunks:
        return None

    return "\n\n".join(retrieved_chunks)


# ==========================
# Interactive Chat Loop
# ==========================

print("\nType 'quit', 'exit', or 'q' to stop.\n")

while True:

    # Tutor/proctor mode selection can be added
    # in a later version.
    user_prompt = input("You: ").strip()

    if user_prompt.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    if not user_prompt:
        continue

    # ==========================
    # Retrieve Course Context
    # ==========================

    start = time.time()

    context = retrieve_context(
        user_prompt,
        TOP_K
    )

    retrieval_time = time.time() - start

    print(
        "Retrieval:",
        retrieval_time,
        "seconds"
    )

    # --------------------------------------------------
    # Guardrail 2:
    # If retrieval did not find sufficiently relevant
    # course material, Python returns the fallback
    # response directly.
    #
    # Phi-3 is NOT called.
    # --------------------------------------------------

    if context is None:

        print("Response:", FALLBACK_RESPONSE)
        print()

        continue

    print(
        "Retrieved course context chars:",
        len(context)
    )


    # ==========================
    # Build Grounded Prompt
    # ==========================

    final_prompt = f"""
<|system|>
You are a BIFS 614 course question-answering assistant.

Your ONLY source of information is the COURSE CONTEXT provided below.

Follow these rules exactly:

1. Answer using ONLY facts explicitly supported by the COURSE CONTEXT.

2. Do NOT use your pretrained knowledge, general knowledge, assumptions,
   or information that is not stated in the COURSE CONTEXT.

3. Knowing an answer from your previous training does NOT mean you are
   allowed to use it.

4. Every factual statement in your answer must be supported by the
   COURSE CONTEXT.

5. If the COURSE CONTEXT contains enough information to answer the
   question, answer clearly and directly using that information.

6. You may use complete sentences, paragraphs, bullet points, or
   numbered lists when appropriate.

7. Do not add extra facts, examples, explanations, definitions,
   technologies, names, or details unless they are supported by the
   COURSE CONTEXT.

8. If the COURSE CONTEXT does not contain enough information to answer
   the question, respond with exactly:

I could not find the answer in the provided context.

Do not explain why the information is missing.
Do not apologize.
Do not answer from memory.
<|end|>

<|user|>
COURSE CONTEXT:

{context}

QUESTION:

{user_prompt}
<|end|>

<|assistant|>
"""


    # ==========================
    # Count Input Tokens
    # ==========================

    input_ids = tokenizer.encode(
        final_prompt,
        return_tensors="pt"
    ).to(model.device)

    input_token_count = input_ids.shape[1]

    print(
        f"Input Tokens: {input_token_count}"
    )


    # ==========================
    # Generate Response
    # ==========================

    start = time.time()

    with torch.no_grad():

        output_ids = model.generate(
            input_ids,

            # Maximum generated answer length
            max_new_tokens=350,

            # Deterministic generation is preferable
            # for grounded course QA.
            do_sample=False,

            repetition_penalty=1.1,

            pad_token_id=tokenizer.eos_token_id
        )

    generation_time = time.time() - start

    print(
        "Generation:",
        generation_time,
        "seconds"
    )


    # ==========================
    # Extract Generated Tokens
    # ==========================

    generated_ids = output_ids[0][input_token_count:]

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    # Remove only leading/trailing whitespace.
    #
    # IMPORTANT:
    # We DO NOT restrict the answer to the first line.
    # This allows paragraphs, bullet points, and
    # numbered lists.
    response = response.strip()


    # ==========================
    # Stop at Unexpected Role Tags
    # ==========================

    for stop_tag in [
        "<|user|>",
        "<|assistant|>",
        "<|system|>",
        "<|end|>"
    ]:

        if stop_tag in response:

            response = response.split(
                stop_tag
            )[0].strip()


    # ==========================
    # Token Counts
    # ==========================

    output_token_count = len(
        generated_ids
    )

    total_token_count = (
        input_token_count
        + output_token_count
    )


    # ==========================
    # Display Results
    # ==========================

    print(
        "Response:",
        response
    )

    print(
        f"Input Tokens: {input_token_count}"
    )

    print(
        f"Output Tokens: {output_token_count}"
    )

    print(
        f"Total Tokens: {total_token_count}"
    )

    print()
