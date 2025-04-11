import os
import logging
import time
from ingest.embedder import Embedder
from utils.file_utils import get_md_files
from utils.logging_config import setup_logging
from utils.normalize import normalize_vector
from flask import Flask, request, jsonify
from ingest.embedder import Embedder
import requests


logger = logging.getLogger(__name__)
app = Flask(__name__)
MODEL_DEEPSEEK = 'deepseek-r1:8b'
MODEL_LLAMA = 'llama3.2'
OLLAMA_SERVER = os.environ.get("OLLAMA_SERVER", "http://localhost:11434/api/generate")
# Initialize the embedder with the same ChromaDB path from your main script
embedder = Embedder(chroma_path="./chroma_storage")

def get_chunk_by_index(chunk_index, filename, source):
    results = embedder.collection.get(
        where={
                "$and": [
                    {"chunk_index": chunk_index},
                    {"filename": filename},
                    {"source": source}
                ]
            },
    )
    if results["documents"]:
        return results["documents"][0]
    return None

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "pong"}), 200

@app.route("/search", methods=["GET"])
def search():
    # Expect a 'query' string in the request
    query_text = request.args.get("query", "")
    if not query_text:
        return jsonify({"error": "Missing 'query' parameter"}), 400
    
    # Embed the query text
    raw_query_embedding = embedder.model.encode(query_text)
    query_embedding = normalize_vector(raw_query_embedding)

    logger.info(f"Query embedding: {query_embedding}")
    # Query ChromaDB collection
    # Adjust 'n_results' to define how many results you need
    results = embedder.collection.query(
        query_embeddings=[query_embedding],
        n_results=10,

    )
    
    return jsonify({"query": query_text, "results": results}), 200

@app.route("/rag", methods=["POST"])
def rag_query():
    query_text = request.get_json().get("query")
    if not query_text:
        return jsonify({"error": "Missing 'query' parameter"}), 400

    combined_text = query_text.strip()
    raw_query_embedding = embedder.model.encode(combined_text)
    query_embedding = normalize_vector(raw_query_embedding)

    # 1) Fetch top results from ChromaDB
    results = embedder.collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
    )

    # Get the first 5 documents and their metadata
    retrieved_chunks = [doc for docs in results["documents"][:3] for doc in docs if doc.strip()]
    context = "\n".join(retrieved_chunks)

    logger.info(f"Context: {context}")
    # 3) Tạo prompt

    prompt_deepseek = f"""
    Answer the question below using only the information from the context. Do not make anything up. If the answer is not in the context, respond with "I don't know".
    Context:
    ---------------------
    {context}
    ---------------------
    Question: {query_text}
    Answer:
    """

    try:
        response_deepseek = requests.post(
            OLLAMA_SERVER,
            json={"prompt": prompt_deepseek, "model": MODEL_DEEPSEEK, "stream": False},
        )
        response_deepseek.raise_for_status()
        deepseek_output = response_deepseek.json().get("response", "").strip()
    except requests.RequestException as e:
        return jsonify({"error": f"LLM RAG error: {str(e)}"}), 500

    prompt_llama = f"""
    Bạn là trợ lý AI của công ty HPT Việt Nam. Dưới đây là phần phân tích ban đầu của một mô hình thông minh, bạn hãy tổng hợp lại câu trả lời một cách trau chuốt, dễ hiểu và đầy đủ cho người dùng bằng tiếng Việt.

    Phân tích:
    ---------------------
    {deepseek_output}
    ---------------------
    Câu hỏi: {query_text}

    Câu trả lời:
    """

    try:
        response_llama = requests.post(
            OLLAMA_SERVER,
            json={"prompt": prompt_llama, "model": MODEL_LLAMA, "stream": False},
        )
        response_llama.raise_for_status()
        llama_output = response_llama.json().get("response", "").strip()
    except requests.RequestException as e:
        return jsonify({"error": f"LLM response error: {str(e)}"}), 500

    return jsonify({
        "query": query_text,
        "retrieved_context": context,
        "reasoning_deepseek": deepseek_output,
        "final_answer": llama_output
    }), 200


@app.route("/ingest", methods=["GET"])
def ingest():
    embedder = Embedder(chroma_path="./chroma_storage")
    # Get md files
    md_files = get_md_files("./data")
    if not md_files:
        logger.warning("No Markdown files found in ./data directory")
        return jsonify({"error": "No Markdown files found in ./data directory"}), 400
    
    # Process statistics
    total_files = len(md_files)
    total_chunks = 0
    total_embeddings = 0
    start_time = time.time()
    
    # Process each markdown file
    for i, md_path in enumerate(md_files, 1):
        logger.info(f"Processing file {i}/{total_files}: {os.path.basename(md_path)}")
        
        # Sử dụng hàm embed_markdown_file từ module embedder
        # Hàm này sẽ trích xuất nội dung, chia chunk và nhúng
        embeddings_count = embedder.embed_markdown_file( md_path)
        
        # Cập nhật thống kê
        if embeddings_count > 0:
            # Số chunk tương đương với số embedding trong trường hợp này
            total_chunks += embeddings_count
            total_embeddings += embeddings_count
            logger.info(f"Added {embeddings_count} embeddings from {os.path.basename(md_path)}")
        else:
            logger.warning(f"No embeddings created from {os.path.basename(md_path)}")
    
    # Log summary
    elapsed_time = time.time() - start_time
    logger.info("=" * 50)
    logger.info("Ingestion Pipeline Complete")
    logger.info(f"Processed {total_files} Markdown files")
    logger.info(f"Created {total_chunks} text chunks")
    logger.info(f"Stored {total_embeddings} embeddings in ChromaDB")
    logger.info(f"Completed in {elapsed_time:.2f} seconds")

    return jsonify({
        "message": "Ingestion complete",
        "total_files": total_files,
        "total_chunks": total_chunks,
        "total_embeddings": total_embeddings,
        "elapsed_time": f"{elapsed_time:.2f} seconds"
    }), 200

def main():
    # Setup logging
    setup_logging()

    # Ensure directories exist
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./chroma_storage", exist_ok=True)
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)