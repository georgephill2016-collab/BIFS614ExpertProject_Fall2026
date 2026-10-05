from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import torch
import time

# ============================================================
# CourseAgentV7-3
# BIFS 614 Tutor Mode
#
# V7-3 Goal:
# Improve answer grounding and conciseness while preserving
# the retrieval guardrail introduced in CourseAgentV7-2.
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

# Retrieve the two nearest course-content chunks.
TOP_K = 2

# Maximum FAISS L2 distance allowed for retrieved chunks.
# Smaller distance = more similar.
MAX_DISTANCE = 1.50


# ============================================================
# Retrieval Function
# ============================================================

def retrieve_context(question, top_k=TOP_K):

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    # Must match the normalization used when building the index.
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    print("Indices:", indices)
    print("Distances:", distances)

    best_distance = float(distances[0][0])

    print(f"Best Distance: {best_distance:.4f}")
    print(f"Maximum Allowed Distance: {MAX_DISTANCE:.4f}")

    retrieved_chunks = []

    # Only accept chunks that pass the distance threshold.
    for distance, idx in zip(distances[0], indices[0]):

        if idx != -1 and distance <= MAX_DISTANCE:
            retrieved_chunks.append(chunks[idx])

    if not retrieved_chunks:

        print("Retrieval Guardrail: REJECTED")
        return None

    print(
        f"Retrieval Guardrail: PASSED "
        f"({len(retrieved_chunks)} relevant chunk(s))"
    )

    return "\n\n".join(retrieved_chunks)


# ============================================================
# Interactive Tutor Loop
# ============================================================

print("\nBIFS 614 Tutor Mode - V7-3")
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

    print("\n========== RETRIEVED CONTEXT ==========")
    print(context)
    print("========== END RETRIEVED CONTEXT ==========\n")


    # ========================================================
    # Phi-3 Prompt
    # ========================================================

    final_prompt = final_prompt = final_prompt = f"""
<|system|>
You are a BIFS 614 course tutor.

Answer the QUESTION using only information from the COURSE
CONTEXT.

You may summarize or paraphrase information from the COURSE
CONTEXT, but do not add information from your own knowledge.

When answering a definition question such as "What is X?",
base the definition directly on how X is described in the
COURSE CONTEXT. Do not create a definition from prior knowledge.

Give only the information needed to directly answer the
QUESTION. Once the QUESTION has been answered, stop.

Do not provide additional examples, applications, background
information, or explanations unless the QUESTION asks for them.
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
