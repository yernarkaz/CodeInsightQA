from openai import OpenAI
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

import numpy as np
import faiss
import openai
import os
import sys
import time

import utils

ROOT_DIR = Path(__file__).parent.parent.parent
config = utils.read_yaml_file(ROOT_DIR / "config/repo_content_indexing_config.yaml")
REPO_FOLDER_NAME = config["repo"]["folder_name"]
GITHUB_REPO_BASE = config["repo"]["url"]
ALLOWED_EXTENSIONS = config["files"]["extensions"]
IGNORED_DIRS = config["files"]["ignored_dirs"]

openai.api_key = os.getenv("OPENAI_API_KEY")


def is_text_file(file_path: Path) -> bool:
    """
    Determine if the file is likely a text file based on its extension.
    """
    return file_path.suffix.lower() in ALLOWED_EXTENSIONS


def traverse_repo(repo_dir: Path):
    """
    Traverse all files under repo_dir filtering relevant text files.
    """
    relevant_files = []
    for root, dirs, files in os.walk(repo_dir):
        # Remove ignored directories in-place
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]

        for file in files:
            file_path = Path(root) / file

            if is_text_file(file_path):
                relevant_files.append(file_path)

    return relevant_files


def process_file(file_path: Path):
    """
    Read file contents. Optionally, add heuristics to ignore very large files.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)
        return None


def process_repo(repo_directory: str, documents: list):
    """
    Process a repository directory by traversing it for relevant files and extracting their content.
    """

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
    """
    Build a GitHub URL for the file using its relative file path.
    """

    # Ensure there are no leading slashes.
    file_path = file_path.lstrip("/")
    return os.path.join(GITHUB_REPO_BASE, file_path)


def split_document(
    document: Dict, chunk_size: int = 50, overlap: int = 10
) -> List[Dict]:
    """
    "Splits the content of a document into manageable text chunks using sliding window logic."
    Args:
    document: A dictionary with keys:
        'file_path': relative file path in the repository
        'content': full text of the document
    chunk_size: Number of lines to include in each chunk.
    overlap: Number of lines to overlap between consecutive chunks.

    Returns:
    A list of dictionaries. Each dictionary has:
        'file_path': relative file path,
        'chunk_text': text contained in this chunk,
        'start_line': starting line number (1-indexed),
        'end_line': ending line number (1-indexed),
        'github_url': URL linking to the file in GitHub.
    """

    chunks = []
    lines = document["content"].splitlines()
    total_lines = len(lines)
    file_path = document["file_path"]

    # Compute GitHub URL reference once for the entire file.
    github_url = build_github_url(file_path)

    # Process lines using a sliding window.
    # Use step = chunk_size - overlap to ensure overlap between chunks.
    step = max(chunk_size - overlap, 1)
    for start in range(0, total_lines, step):
        end = min(start + chunk_size, total_lines)
        chunk_lines = lines[start:end]
        # Join lines into a text chunk.
        chunk_text = "\n".join(chunk_lines)
        chunk_meta = {
            "file_path": file_path,
            "chunk_text": chunk_text,
            "start_line": start + 1,  # Convert 0-index to 1-index.
            "end_line": end,
            "github_url": github_url,
        }
        chunks.append(chunk_meta)
        # If we reached the end of the document, break.
        if end == total_lines:
            break

    return chunks


def get_embedding(
    client: OpenAI, text: str, model: str = "text-embedding-ada-002"
) -> List[float]:
    """
    Retrieve the embedding vector for a given text using OpenAI's text-embedding-ada-002 model.
        Args:
        text: Input text to embed.
        model: The OpenAI embedding model to use.

    Returns:
        A list of floats representing the embedding.
    """
    try:
        response = client.embeddings.create(input=text, model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []


def embed_chunks(client: OpenAI, chunks: List[Dict]) -> List[Dict]:
    """
    Iterate over a list of text chunk dictionaries and attach an embedding vector to each.
    Each chunk dictionary should have a 'chunk_text' key.

    Args:
        chunks: List of dictionaries containing chunk data.

    Returns:
        The same list with an additional key 'embedding' for each chunk.
    """
    for chunk in chunks:
        text_to_embed = chunk.get("chunk_text", "")
        if text_to_embed:
            chunk["embedding"] = get_embedding(client, text_to_embed)
        else:
            chunk["embedding"] = []

    return chunks


def create_faiss_index(
    embedded_chunks: List[Dict],
) -> Tuple[faiss.IndexFlatL2, List[Dict]]:
    """
    Create and populate a FAISS vector index with document chunk embeddings.
    Also return the metadata list corresponding to each vector in the index."

    Args:
    embedded_chunks: A list of dictionaries where each dictionary includes:
                        - "embedding": list or numpy array representation of the vector.
                        - Additional metadata keys (e.g., file_path, start_line, etc.).

    Returns:
        A tuple (index, metadata_list) where:
            index: The FAISS index populated with embeddings.
            metadata_list: A list of dictionaries of metadata aligned with the index's vectors.

    Note: This code assumes that all embeddings have the same dimension.
    """

    if not embedded_chunks:
        raise ValueError("No embedded chunks to index.")

    # Determine the dimension of the embeddings.
    dim = len(embedded_chunks[0]["embedding"])

    # Create a FAISS index (using the L2 distance).
    index = faiss.IndexFlatL2(dim)

    # Prepare lists for embeddings and metadata.
    embeddings = []
    metadata_list = []

    for chunk in embedded_chunks:
        vector = np.array(chunk["embedding"], dtype="float32")
        embeddings.append(vector)

        # You can store whatever metadata you need, for example:
        meta = {
            "file_path": chunk.get("file_path"),
            "start_line": chunk.get("start_line"),
            "end_line": chunk.get("end_line"),
            "github_url": chunk.get("github_url"),
            "chunk_text": chunk.get("chunk_text"),
        }
        metadata_list.append(meta)

    # Convert list of embeddings into a NumPy array of shape (n_chunks, dim)
    embeddings_np = np.vstack(embeddings)

    # Add these vectors to the FAISS index
    index.add(embeddings_np)

    print(f"Indexed {index.ntotal} document chunks with embedding dimension = {dim}.")
    return index, metadata_list


def query_index(
    embedded_chunks: List[Dict], query_embedding: List[float], k: int = 1
) -> List[Dict]:
    """
    Query the FAISS index with a query embedding; k is the number of nearest neighbors to retrieve.

    Returns:
        A list of metadata dictionaries for the top-k similar chunks.
    """

    # Create the FAISS index with embedded chunks.
    index, metadata_list = create_faiss_index(embedded_chunks)
    query_vector = np.array(query_embedding, dtype="float32").reshape(1, -1)
    distances, indices = index.search(query_vector, k)
    results = []
    for idx in indices[0]:
        if idx != -1 and idx < len(metadata_list):
            result = metadata_list[idx]
            result["distance"] = distances[0][idx]
            results.append(result)

    return results


def measure_indexing_speed(
    embedded_chunks: List[Dict],
) -> Tuple[faiss.IndexFlatL2, List[Dict], float]:
    """
    Args:
        embedded_chunks: List of dictionaries containing embedding vectors and metadata.

    Returns:
        A tuple containing:
            - The FAISS index.
            - The metadata list.
            - The time taken for indexing (in seconds).
    """
    start_time = time.time()

    # Create the FAISS index.
    index, metadata_list = create_faiss_index(embedded_chunks)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(
        f"Indexing completed in {elapsed_time:.2f} seconds for {len(embedded_chunks)} chunks."
    )
    return (
        index,
        metadata_list,
        elapsed_time,
    )


def run(repo_directory: str):
    documents = []
    process_repo(repo_directory, documents)

    # print("\nIndexed Documents Summary:")
    # for doc in documents:
    #     print(f" - {doc['file_path']} (Length: {len(doc['content'])} chars)")

    # Split the sample document.
    doc_chunks = [
        split_document(doc, chunk_size=20, overlap=5) for doc in tqdm(documents)
    ]
    # for idx, chunks in enumerate(doc_chunks[:1]):
    # for chunk in chunks:
    #     print(f"--- Chunk {idx+1} ---")
    #     print(f"File: {chunk['file_path']}")
    #     print(f"Lines: {chunk['start_line']} - {chunk['end_line']}")
    #     print(f"GitHub URL: {chunk['github_url']}")
    #     print(f"Content Preview: {chunk['chunk_text'][:100]}...\n")

    client = OpenAI()

    # Generate embeddings for each chunk.
    embedded_chunks = embed_chunks(client, doc_chunks[0])

    # Print out the embedding length for each chunk for verification.
    for idx, chunk in enumerate(embedded_chunks):
        embedding = chunk.get("embedding", [])
        print(
            f"Chunk {idx+1} from {chunk['file_path']} has embedding of length: {len(embedding)}"
        )

    # Generate a dummy query embedding.
    dummy_query_embedding = np.random.rand(1536).tolist()

    # Retrieve the top match.
    top_results = query_index(embedded_chunks, dummy_query_embedding, k=2)
    for i, res in enumerate(top_results, 1):
        print(f"\nResult {i}:")
        print(
            f"File: {res['file_path']} Lines: {res['start_line']} - {res['end_line']}"
        )
        print(f"GitHub URL: {res['github_url']}")
        print(f"Distance: {res['distance']}")


if __name__ == "__main__":
    run(ROOT_DIR / REPO_FOLDER_NAME)

# Here you can proceed to further steps, such as splitting documents into chunks and embedding them.
