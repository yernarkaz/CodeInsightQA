import os
import time
import faiss
import openai
import numpy as np
import utils
import uvicorn

from index_repo import get_embedding, query_index
from fastapi import FastAPI
from openai import OpenAI, AzureOpenAI
from pathlib import Path
from pydantic import BaseModel
from typing import Tuple


# Load the configuration file.
config = utils.read_yaml_file("config/indexing_config.yaml")


# Load the environment variables or use the default values from the configuration file.
ENDPOINT = os.getenv("endpoint", config["llm_azure"]["endpoint"])
DEPLOYMENT = os.getenv("deployment", config["llm_azure"]["deployment"])
SUBSCRIPTION_KEY = os.getenv(config["llm_api_key"][config["llm_endpoint_type"]])
API_VERSION = os.getenv("api_version", config["llm_azure"]["api_version"])


# Load the configuration parameters.
TOP_K = config["similarity_search"]["top_k"]
SIMILARITY_THRESHOLD = config["similarity_search"]["distance_threshold"]
MAX_TOKENS = config["embedding"]["max_tokens"]
TEMPERATURE = config["embedding"]["temperature"]


# Load the Faiss index and metadata list.
index = faiss.read_index("data/faiss_index.index")
metadata_list = np.load("data/metadata_list.npy", allow_pickle=True).tolist()
print("Index loaded successfully.", type(index))


# Define the Pydantic models for the request and response.
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    index_time: float = 0.0
    llm_time: float = 0.0


# Initialize the OpenAI client based on the endpoint type.
if config["llm_endpoint_type"] == "azure":
    client = AzureOpenAI(
        azure_endpoint=ENDPOINT,
        api_key=SUBSCRIPTION_KEY,
        api_version=API_VERSION,
    )
elif config["llm_endpoint_type"] == "openai":
    client = OpenAI(api_key=SUBSCRIPTION_KEY)


# Initialize the FastAPI app.
app = FastAPI(title="CodeInsightQA API")


# Define the health check endpoint.
@app.get("/health")
def health_check():
    return {"status": "OK", "message": "CodeInsightQA API is running."}


# Define the endpoint to ask a question.
@app.post("/ask", response_model=QueryResponse)
async def ask_question(query: QueryRequest):
    # Use the global variables for the Faiss index and metadata list.

    global index, metadata_list

    question = query.question

    overall_start = time.time()

    question_embedding = get_embedding(client, question)
    if not question_embedding:
        return QueryResponse(answer="Error creating embedding.")

    retrieved_chunks = query_index(index, metadata_list, question_embedding, k=TOP_K)
    print(f"Retrieved {len(retrieved_chunks)} chunks.")
    print(retrieved_chunks[0])

    # Check the similarity of the top result.
    # the list is sorted by similarity, so we only need to check the first element.
    if not retrieved_chunks or retrieved_chunks[0]["distance"] > SIMILARITY_THRESHOLD:
        total_time = time.time() - overall_start
        return QueryResponse(answer="Out-of-the-scope", index_time=total_time)

    index_time = time.time() - overall_start
    answer, llm_time = generate_answer(question, retrieved_chunks)

    return QueryResponse(
        answer=answer, index_time=round(index_time, 2), llm_time=round(llm_time, 2)
    )


def generate_answer(question: str, context_chunks: list) -> Tuple[str, float]:
    # Generate the answer using the LLM model.
    # The context_chunks contain the relevant information from the repository.

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

    # Define the system and user prompts for the LLM model.
    system_prompt = (
        "You are an assistant that only answers questions using the repository information provided. "
        "If the question is not related to the content below, reply with 'Out-of-the-scope'. "
        "Include GitHub file links and line number references where applicable."
    )

    user_prompt = f"Repository Context:\n{context_text}\n\nQuestion: {question}"

    # Generate the answer using the LLM model.
    start_llm = time.time()
    try:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        answer = f"Error generating answer: {e}"

    llm_time = time.time() - start_llm
    return answer, llm_time


if __name__ == "__main__":
    # Run the FastAPI app using Uvicorn.
    uvicorn.run(app, host="0.0.0.0", port=8000)
