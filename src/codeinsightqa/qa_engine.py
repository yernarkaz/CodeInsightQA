import os
import time
import faiss
import openai
import numpy as np
import utils
import uvicorn

from fastapi import FastAPI
from openai import OpenAI
from pathlib import Path
from pydantic import BaseModel
from typing import List, Dict, Tuple

openai.api_key = os.getenv("OPENAI_API_KEY")


ROOT_DIR = Path(__file__).parent.parent.parent
config = utils.read_yaml_file("config/llm_integration_config.yaml")
SIMILARITY_THRESHOLD = config["similarity_search"]["threshold"]
OUT_OF_SCOPE_PROMPT = config["out_of_scope"]["prompt"]


index = faiss.read_index("data/faiss_index.index")
metadata_list = np.load("data/metadata_list.npy", allow_pickle=True).tolist()
print("Index loaded successfully.", type(index))


def get_embedding(client: OpenAI, text: str, model: str) -> List[float]:

    try:
        response = client.embeddings.create(input=text, model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []


def query_index(
    index: faiss.IndexFlatL2,
    metadata_list: List[Dict],
    query_embedding: List[float],
    k: int = 1,
) -> List[Dict]:

    query_vector = np.array(query_embedding, dtype="float32").reshape(1, -1)
    distances, indices = index.search(query_vector, k)
    print("distances:", distances)  # The distance to the nearest neighbor
    print("indices:", indices)  # The index of the nearest neighbor
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx != -1 and idx < len(metadata_list):
            result = metadata_list[idx]
            result["distance"] = distance
            results.append(result)

    return results


def generate_answer(question: str, context_chunks: list) -> Tuple[str, float]:
    """
    Build a prompt using the retrieved context chunks and query the OpenAI LLM for an answer."
    The prompt instructs the LLM:
    - To answer using only the provided repository context.
    - To say 'Out-of-scope' if the context does not match the question.
    - To include GitHub links (with file paths and line numbers) as bonus information.

    Returns:
    - The generated answer.
    - Time taken by the LLM for generation (in seconds).
    """

    # Join context chunks (each with file name, line numbers, GitHub URL, and content)
    context_text = "\n\n".join(
        [
            (
                f"File: {chunk['file_path']} (Lines {chunk['start_line']}-{chunk['end_line']})\n"
                f"URL: {chunk['github_url']}\n"
                f"Content:\n{chunk['chunk_text']}"
            )
            for chunk in context_chunks
        ]
    )

    system_prompt = (
        "You are an assistant that only answers questions using the repository information provided. "
        "If the question is not related to the content below, reply with 'Out-of-scope'. "
        "Include GitHub file links and line number references where applicable."
    )

    user_prompt = f"Repository Context:\n{context_text}\n\nQuestion: {question}"

    start_llm = time.time()
    try:
        client = OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1500,
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        answer = f"Error generating answer: {e}"

    llm_time = time.time() - start_llm
    return answer, llm_time


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    index_time: float = 0.0
    llm_time: float = 0.0


app = FastAPI(title="CodeInsightQA API")
client = OpenAI()


@app.get("/health")
def health_check():
    return {"status": "OK", "message": "CodeInsightQA API is running."}


@app.post("/ask", response_model=QueryResponse)
async def ask_question(query: QueryRequest):
    global index, metadata_list

    question = query.question

    overall_start = time.time()

    question_embedding = get_embedding(client, question, model="text-embedding-ada-002")
    if not question_embedding:
        return QueryResponse(answer="Error creating embedding.")

    retrieved_chunks = query_index(index, metadata_list, question_embedding, k=5)

    # Check the similarity of the top result.
    if not retrieved_chunks or retrieved_chunks[0]["distance"] > SIMILARITY_THRESHOLD:
        total_time = time.time() - overall_start
        return QueryResponse(answer="Out-of-scope", index_time=total_time)

    index_time = time.time() - overall_start
    answer, llm_time = generate_answer(question, retrieved_chunks)

    return QueryResponse(answer=answer, index_time=index_time, llm_time=llm_time)


if __name__ == "__main__":
    # Run the FastAPI app.
    uvicorn.run(app, host="0.0.0.0", port=8000)
