import json
import requests

from tqdm import tqdm

q_list = [
    "what is the capital of France?",
    "how can i create a chromadb rag?",
    "what is the difference between chromadb and qdrant?",
    "how does should_generate_chart function from vanna base works?",
    "What context strategy improved SQL accuracy the most?",
    "How did GPT-4 and Bison compare in SQL generation?",
    "How does the system help business users get SQL answers?",
    "Why is SQL generation by AI useful?",
    "What are the key skills needed to query data warehouses?",
    "What problem do business users face with data access?",
    "Which LLM was considered the best for SQL?",
    "What was the accuracy increase with proper context?",
    "What data is included in the optimal context strategy?",
    "What was the main limitation of using ChatGPT without context?",
]


def test_qa():
    test_results = []
    for q in tqdm(q_list):
        response = requests.post("http://localhost:8000/ask", json={"question": q})

        if response.status_code == 200:
            test_results.append({"q": q, "response": response.json()})

    with open("test_results.json", "w") as f:
        json.dump(test_results, f)


if __name__ == "__main__":
    test_qa()
