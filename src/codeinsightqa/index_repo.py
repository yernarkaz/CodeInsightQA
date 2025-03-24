from openai import OpenAI, AzureOpenAI
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

import numpy as np
import faiss
import os
import sys
import time

import utils


config = utils.read_yaml_file("config/indexing_config.yaml")

REPO_FOLDER_NAME = config["repo"]["folder_name"]
GITHUB_REPO_BASE = config["repo"]["url"]
ALLOWED_EXTENSIONS = config["files"]["extensions"]
IGNORED_DIRS = config["files"]["ignored_dirs"]


ENDPOINT = config["llm_azure"]["endpoint"]
DEPLOYMENT = config["llm_azure"]["deployment"]
SUBSCRIPTION_KEY = os.getenv(config["llm_api_key"][config["llm_endpoint_type"]])
API_VERSION = config["llm_azure"]["api_version"]
EMBEDDING_MODEL = config["embedding"]["model"]


if config["llm_endpoint_type"] == "azure":
    client = AzureOpenAI(
        azure_endpoint=ENDPOINT,
        api_key=SUBSCRIPTION_KEY,
        api_version=API_VERSION,
        azure_deployment=DEPLOYMENT,
    )
elif config["llm_endpoint_type"] == "openai":
    client = OpenAI(api_key=SUBSCRIPTION_KEY)


def is_text_file(file_path: Path) -> bool:
    return file_path.suffix.lower() in ALLOWED_EXTENSIONS


def traverse_repo(repo_dir: Path):

    relevant_files = []
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]

        for file in files:
            file_path = Path(root) / file

            if is_text_file(file_path):
                relevant_files.append(file_path)

    return relevant_files


def process_file(file_path: Path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)
        return None


def process_repo(repo_directory: str, documents: list):

    repo_dir = Path(repo_directory).resolve()
    if not repo_dir.is_dir():
        print(f"Provided repository directory {repo_dir} is not valid.")
        return

    print(f"Traversing repository at {repo_dir}")
    relevant_files = traverse_repo(repo_dir)
    print(f"Found {len(relevant_files)} relevant files.")

    for file_path in relevant_files:
        content = process_file(file_path)
        if content is None:
            continue

        doc = {
            "file_path": str(file_path.relative_to(repo_dir)),
            "content": content,
            "file_size": os.path.getsize(file_path),
            "created_at": os.path.getctime(file_path),
            "file_extension": file_path.suffix,
        }
        documents.append(doc)


def build_github_url(file_path: str) -> str:
    file_path = file_path.lstrip("/")
    return os.path.join(GITHUB_REPO_BASE, file_path)


def split_document(document: Dict, chunk_size: int, overlap: int) -> List[Dict]:

    chunks = []
    lines = document["content"].splitlines()
    total_lines = len(lines)
    file_path = document["file_path"]

    github_url = build_github_url(file_path)

    # Process lines using a sliding window.
    # Use step = chunk_size - overlap to ensure overlap between chunks.
    step = max(chunk_size - overlap, 1)
    for start in range(0, total_lines, step):
        end = min(start + chunk_size, total_lines)
        chunk_lines = lines[start:end]

        chunk_text = "\n".join(chunk_lines)
        chunk_meta = {
            "file_path": file_path,
            "chunk_text": chunk_text,
            "start_line": start + 1,  # Convert 0-index to 1-index.
            "end_line": end,
            "github_url": github_url,
        }
        chunks.append(chunk_meta)

        if end == total_lines:
            break

    return chunks


def get_embedding(client: OpenAI, text: str) -> List[float]:

    try:
        response = client.embeddings.create(
            input=text, model=EMBEDDING_MODEL, dimensions=1024
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []


def embed_chunks(client: OpenAI, chunks: List[Dict]) -> List[Dict]:

    for chunk in tqdm(chunks):
        text_to_embed = chunk.get("chunk_text", "")
        if text_to_embed:
            chunk["embedding"] = get_embedding(client, text_to_embed)
        else:
            chunk["embedding"] = []

    return chunks


def create_faiss_index(
    embedded_chunks: List[Dict],
) -> Tuple[faiss.IndexFlatL2, List[Dict]]:

    if not embedded_chunks:
        raise ValueError("No embedded chunks to index.")

    dim = len(embedded_chunks[0]["embedding"])
    index = faiss.IndexFlatL2(dim)

    embeddings = []
    metadata_list = []

    for chunk in embedded_chunks:
        vector = np.array(chunk["embedding"], dtype="float32")
        embeddings.append(vector)

        meta = {
            "file_path": chunk.get("file_path"),
            "start_line": chunk.get("start_line"),
            "end_line": chunk.get("end_line"),
            "github_url": chunk.get("github_url"),
            "chunk_text": chunk.get("chunk_text"),
        }
        metadata_list.append(meta)

    embeddings_np = np.vstack(embeddings)
    index.add(embeddings_np)

    print(f"Indexed {index.ntotal} document chunks with embedding dimension = {dim}.")
    print(f"Metadata list has {len(metadata_list)} entries.")
    return index, metadata_list


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


def run(repo_directory: str):

    documents = []
    process_repo(repo_directory, documents)

    doc_chunks = [
        split_document(doc, chunk_size=10, overlap=5) for doc in tqdm(documents)
    ]

    embedded_chunks = []
    for chunks in tqdm(doc_chunks):
        embedded_chunks.extend(embed_chunks(client, chunks))

    print(f"Loaded {len(embedded_chunks)} embedded chunks.")
    embedded_chunks = [
        chunks for chunks in embedded_chunks if len(chunks["embedding"]) != 0
    ]
    print(
        f"Loaded {len(embedded_chunks)} embedded chunks excluding chunks with 0 embeddings."
    )

    start_time = time.time()
    index, metadata_list = create_faiss_index(embedded_chunks)
    end_time = time.time()
    print(f"Faiss Index creation took {end_time - start_time:.2f} seconds.")

    faiss.write_index(index, str(ROOT_DIR / "data/faiss_index.index"))
    np.save(ROOT_DIR / "data/metadata_list.npy", metadata_list, allow_pickle=True)


if __name__ == "__main__":
    run(ROOT_DIR / REPO_FOLDER_NAME)
