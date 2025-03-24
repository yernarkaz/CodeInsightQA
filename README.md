# CodeInsightQA

This project indexes a repository, builds a vector search (using FAISS) for code/documentation chunks, and then provides a FastAPI endpoint to answer questions based on the repository content using an LLM (Azure/OpenAI).

## Prerequisites

- Python 3.12
- [Docker Desktop](https://www.docker.com/products/docker-desktop) (optional)
- An Azure OpenAI or OpenAI API key  
- Git


## Setup Instructions

### Clone the Repository

If you haven’t already, clone the repository and navigate to its root directory:
```bash
git clone https://github.com/yourusername/CodeInsightQA.git
cd CodeInsightQA
```

## Running the Application Using Docker

After building your Docker image, you can run the container with the proper environment variables and port mapping. This section details each step of the process.

### 1. Build the Docker Image
Make sure the Dockerfile is in the project root. From the project’s root directory, run:

```bash
docker build -t codeinsightqa .
```

### 2. Run the Docker Container:

The application requires several environment variables (such as the API key, endpoint URL, and API version) to be passed in at runtime. Use the --env (or -e) flag provided by Docker to supply these variables.

For example, to run the container with the required Azure OpenAI environment variables, execute:

```bash
docker run \
  --env AZURE_OPENAI_API_KEY=<your_azure_openai_api_key> \
  --env endpoint=<your_endpoint_url> \
  --env api_version=<your_api_version> \
  -p 8000:8000 \
  codeinsightqa
```

### 3. Verify the Application Is Running:
Once the container starts, you should see log output indicating that Uvicorn has launched the FastAPI server on 0.0.0.0:8000.

To verify:

- Open a browser and navigate to: http://localhost:8000/health

- Alternatively, use curl from your terminal:
```bash
curl -X GET http://localhost:8000/health
```

You should receive a JSON response like:
```bash
{
  "status": "OK",
  "message": "CodeInsightQA API is running."
}
```

### 4. Ask a Question about Vanna(https://github.com/vanna-ai/vanna/tree/main) Repository

The application exposes an `/ask` endpoint that accepts a question about the indexed repository. It generates an embedding for your question, queries the FAISS index, and then uses an LLM to provide an answer based on the code/documentation chunks.

#### How It Works

1. **JSON Payload:**  
   The endpoint expects a JSON payload with a field called `question`. For example:

    ```bash
    curl -X POST http://localhost:8000/ask \
    -H "Content-Type: application/json" \
    -d '{"question": "what is the capital of France?"}'
    ```

2. **Processing:**

    The application generates an embedding for your question.
    It queries the pre-built FAISS index for context.
    An LLM (either Azure OpenAI or OpenAI) generates an answer using the context from the repository.
    Response:
        The API responds with a JSON object containing:

        - The original question (q),
        - The generated answer,
        - Timing information for the index search (index_time) in seconds,
        - And timing information for the LLM completion (llm_time) in seconds.

    You may receive a response similar to one of the following:

    - Example response for a Question **Out-of-the-Scope**:
    ```bash
    {
        "q": "what is the capital of France?",
        "response": {
            "answer": "Out-of-the-scope",
            "index_time": 0.98,
            "llm_time": 0
        }
    }
    ```
    
    
    - Example response with **Repository Context**:

    ```bash
    curl -X POST http://localhost:8000/ask \
    -H "Content-Type: application/json" \
    -d '{"question": "How did GPT-4 and Bison compare in SQL generation?"}'
    ```

    ```bash
    {
            "q": "How did GPT-4 and Bison compare in SQL generation?",
            "response": {
                "answer": "GPT-4 and Bison were compared in SQL generation based on their performance across different context strategies. According to the paper, GPT 4 was found to be the best overall LLM for generating SQL, taking the crown when averaged across the three strategies. Bison, on the other hand, started out at the bottom of the heap when using just the Schema and Static context strategies. However, Bison showed a significant improvement and rocketed to the top with a full Contextual strategy, becoming roughly equivalent to GPT 4 when enough context was provided ([Lines 246-255](https://github.com/vanna-ai/vanna/blob/main/papers/ai-sql-accuracy-2023-08-17.md#L246-L255), [Lines 6-15](https://github.com/vanna-ai/vanna/blob/main/papers/ai-sql-accuracy-2023-08-17.md#L6-L15)).",
                "index_time": 0.3,
                "llm_time": 6.53
            }
        }
    ```

### 5. Troubleshooting
* Embedding or LLM Issues:
    - If you see errors related to generating embeddings or responses (e.g., deployment not found, max_tokens too high), verify your configuration in config/indexing_config.yaml and environment variables.

* File Not Found Errors:
    - Ensure the directory structure inside the container matches what the code expects (for example, the config and data folders should be present in /app).

## **FAISS** Vector Database and **text-embedding-3-large** Embedding Model:

The core of the similarity search in CodeInsightQA is built on a vector database implemented using [FAISS](https://github.com/facebookresearch/faiss). 

### FAISS:

- **High Performance:**  
  FAISS, developed by Facebook AI Research, is optimized for efficient similarity searches even in high-dimensional space. It enables near-instantaneous retrieval of the most similar document chunks from the repository.

- **Scalability:**  
  As the number of repository grows, FAISS can handle billions of vectors, ensuring that similarity queries remain fast and accurate. Supports gpu version for faster response.

- **Flexibility:**  
  FAISS supports various index types and distance metrics (L2 distance here), making it a flexible choice for a wide range of vector search applications.


### text-embedding-3-large:

For generating embeddings from code and documentation text, CodeInsightQA utilizes the `text-embedding-3-large` model as specified in the configuration. The reasons for this are:

- **Quality of Representations:**  
  The chosen embedding model is designed to capture semantic meaning from natural language descriptions as well as code snippets. This ensures that the generated embeddings accurately reflect the context and nuances of the repository content.

- **Model Compatibility:**  
  This model is compatible with the Azure/OpenAI API, allowing for straightforward integration and scaling within the solution.

- **Performance Considerations:**  
  With a manageable dimensionality (reduced dimension is 1024), the embeddings are rich enough for effective similarity search, while remaining efficient in terms of storage and computation.