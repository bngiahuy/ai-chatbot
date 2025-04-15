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

# Setup logging
setup_logging()

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "pong"}), 200


@app.route("/search", methods=["POST"])
def search():
    # Lấy 'query' từ request
    query_text = request.get_json().get("query").strip()
    if not query_text:
        return jsonify({"error": "Missing 'query' parameter"}), 400

    # try:
    #     query_embedding, llm_output = convert_embedding(query_text)
    # except Exception as e:
    #     return jsonify({"error": f"Error: {str(e)}"}), 500

    query_embedding = embedder.encode_query(query_text)

    results = embedder.collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
    )

    return jsonify({
        "original_query": query_text,
        # "transformed_query": llm_output,
        "results": results
    }), 200

@app.route("/rag", methods=["POST"])
def rag_query():
    query_text = request.get_json().get("query")
    if not query_text:
        return jsonify({"error": "Missing 'query' parameter"}), 400

    # try:
    #     query_embedding, output = convert_embedding(query_text)
    # except Exception as e:
    #     return jsonify({"error": f"Embedding error: {str(e)}"}), 500

    query_embedding = embedder.encode_query(query_text)

    # 1) Fetch top results from ChromaDB
    results = embedder.collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
    )

    retrieved_chunks = [doc for docs in results["documents"] for doc in docs if doc.strip()]
    context = "\n".join(retrieved_chunks)

    logger.info(f"Context: {context}")
    # 3) Tạo prompt

    prompt_deepseek = f"""
    Answer the following question in English, using only the information provided in the context below.
    Do not include any additional information or personal opinions. If the answer is not present in the context, say "I don't know".
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
            json={"prompt": prompt_deepseek, "model": MODEL_DEEPSEEK, "stream": False, "options": {"seed": 13102002}},
        )
        response_deepseek.raise_for_status()
        deepseek_output = response_deepseek.json().get("response", "").strip()
    except requests.RequestException as e:
        return jsonify({"error": f"LLM RAG error: {str(e)}"}), 500

    prompt_llama = f"""
    Bạn là trợ lý AI của công ty HPT Việt Nam. Dưới đây là phần phân tích ban đầu, bạn hãy tổng hợp lại câu trả lời một cách trau chuốt, dễ hiểu và đầy đủ cho người dùng bằng tiếng Việt.

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
            json={"prompt": prompt_llama, "model": MODEL_LLAMA, "stream": False, "options": {"seed": 13102002}},
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
    md_files = get_md_files("./data/vn")
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
    # Ensure directories exist
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./chroma_storage", exist_ok=True)
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)