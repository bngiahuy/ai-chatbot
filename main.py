from dotenv import load_dotenv
import os
import logging
import time
from ingest.embedder import Embedder
from utils.file_utils import get_md_files
from utils.logging_config import setup_logging
from flask import Flask, request, jsonify
from ingest.embedder import Embedder
import requests
from collections import Counter

load_dotenv()

logger = logging.getLogger(__name__)
app = Flask(__name__)
MODEL_DEEPSEEK = 'deepseek-r1:8b'
MODEL_LLAMA = 'llama3.2'
OLLAMA_SERVER = os.environ.get("OLLAMA_SERVER", "http://localhost:11434/api/generate")
# Initialize the embedder with the same ChromaDB path from your main script
embedder = Embedder(chroma_path="./chroma_storage")

# Setup logging
setup_logging()

# Helper function
def run_vector_search(query_text: str, n_results: int =10, metadata_filters=None):
    query_embedding = embedder.encode_query(query_text)
    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
    }
    if metadata_filters:
        query_params["where"] = metadata_filters

    raw_results = embedder.collection.query(**query_params)

    return raw_results


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "pong"}), 200

@app.route("/search", methods=["POST"])
def search():
    start_time = time.time()
    request_data = request.get_json() or {}
    query_text = request_data.get("query", "").strip()

    if not query_text:
        return jsonify({"error": "Missing or empty 'query' parameter"}), 400

    try:
        n_results = int(request_data.get("n_results", 10))
        metadata_filters = request_data.get("metadata_filters", None)

        logger.info(f"Search: '{query_text}' (n_results={n_results})")

        result_data = run_vector_search(
            query_text=query_text,
            n_results=n_results,
            metadata_filters=metadata_filters,
        )

        elapsed = time.time() - start_time
        response = {
            "original_query": query_text,
            "results": result_data,
            "total_results": len(result_data),
            "metrics": {"elapsed_seconds": elapsed}
        }

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({"error": f"Search failed: {str(e)}"}), 500


@app.route("/rag", methods=["POST"])
def rag_query():
    request_data = request.get_json() or {}
    query_text = request_data.get("query", "").strip()
    n_initial_results = int(request_data.get("n_initial_results", 10)) # Số kết quả ban đầu
    n_expanded_results = int(request_data.get("n_expanded_results", 30)) # Số kết quả mở rộng từ file chính

    if not query_text:
        return jsonify({"error": "Missing 'query' parameter"}), 400

    try:
        # Step 1a: Initial search to find relevant files
        initial_results = run_vector_search(query_text, n_results=n_initial_results)

        if not initial_results or not initial_results.get("ids") or not initial_results["ids"][0]:
            logger.warning(f"[RAG] No relevant chunks found initially for query: {query_text}")
            # Trả về phản hồi không tìm thấy thông tin
            return jsonify({
                "query": query_text, "retrieved_context": "",
                "reasoning_deepseek": "I don't know based on the provided context.",
                "final_answer": "Xin lỗi, tôi không tìm thấy thông tin cụ thể để trả lời câu hỏi này."
            }), 200

        # Step 1b: Identify the most relevant file
        metadatas = initial_results.get("metadatas", [[]])[0]
        if not metadatas:
             logger.warning(f"[RAG] No metadata found in initial results for query: {query_text}")
             # Xử lý như không tìm thấy chunks
             return jsonify({
                "query": query_text, "retrieved_context": "",
                "reasoning_deepseek": "I don't know based on the provided context.",
                "final_answer": "Xin lỗi, tôi không tìm thấy thông tin cụ thể để trả lời câu hỏi này."
            }), 200

        filenames = [meta.get("filename") for meta in metadatas if meta and meta.get("filename")]
        if not filenames:
            logger.warning(f"[RAG] No filenames found in metadata for query: {query_text}")
             # Xử lý như không tìm thấy chunks
            return jsonify({
                "query": query_text, "retrieved_context": "",
                "reasoning_deepseek": "I don't know based on the provided context.",
                "final_answer": "Xin lỗi, tôi không tìm thấy thông tin cụ thể để trả lời câu hỏi này."
            }), 200

        most_common_file = Counter(filenames).most_common(1)[0][0]
        logger.info(f"[RAG] Most relevant file identified: {most_common_file}")

        # Step 1c: Retrieve more context from the most relevant file
        expanded_results = run_vector_search(
            query_text,
            n_results=n_expanded_results,
            metadata_filters={"filename": most_common_file}
        )

        # Step 1d: Combine context
        retrieved_docs = expanded_results.get("documents", [[]])[0]
        retrieved_chunks = [doc for doc in retrieved_docs if doc and doc.strip()]

        if not retrieved_chunks:
             logger.warning(f"[RAG] No expanded chunks found for file {most_common_file}, query: {query_text}")
             # Fallback: sử dụng kết quả ban đầu nếu có
             initial_docs = initial_results.get("documents", [[]])[0]
             retrieved_chunks = [doc for doc in initial_docs if doc and doc.strip()]
             if not retrieved_chunks:
                  # Vẫn không có gì, trả về không biết
                  return jsonify({
                      "query": query_text, "retrieved_context": "",
                      "reasoning_deepseek": "I don't know based on the provided context.",
                      "final_answer": "Xin lỗi, tôi không tìm thấy thông tin cụ thể để trả lời câu hỏi này."
                  }), 200

        context = "\n".join(retrieved_chunks)
        logger.info(f"[RAG] Expanded context generated ({len(retrieved_chunks)} chunks from {most_common_file}) for query: {query_text}")


        # Step 2 & 3: (Giữ nguyên phần gọi DeepSeek và Llama)
        # Step 2: Ask DeepSeek for reasoning
        prompt_deepseek = f"""
You are a precise and obedient language model of HPT Vietnam Corporation.

Your task is to answer questions **only** based on the given context.

- Do **not** use any prior knowledge, assumptions, or external information.
- Do **not** make up facts, speculate, or include opinions.
- If the answer cannot be found **explicitly or implicitly** in the context, reply with:
"I don't know based on the provided context."

Respond clearly, concisely, and strictly grounded in the context.

---

Context:
{context}

---

Question:
{query_text}

Answer:
        """

        response_deepseek = requests.post(
            OLLAMA_SERVER,
            json={
                "prompt": prompt_deepseek, 
                "model": MODEL_DEEPSEEK, 
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "seed": 42,
                }
            },
        )
        response_deepseek.raise_for_status()
        deepseek_output = response_deepseek.json().get("response", "").strip()
        if not deepseek_output:
            logger.warning(f"[RAG] No response from DeepSeek for query: {query_text}")
            # Trả về phản hồi không tìm thấy thông tin
            return jsonify({
                "query": query_text, 
                "retrieved_context": context,
                "reasoning_deepseek": "I don't know based on the provided context.",
                "final_answer": "Xin lỗi, tôi không tìm thấy thông tin cụ thể để trả lời câu hỏi này."
            }), 200
        
        deepseek_output_think = deepseek_output.split("</think>\n")[0]
        deepseek_output_answer = deepseek_output.split("</think>\n")[1]

#         # Step 3: Vietnamese final answer using LLaMA
#         prompt_llama = f"""
# Bạn là dịch thuật viên của HPT Vietnam Corporation.
# Bạn có nhiệm vụ dịch câu trả lời từ tiếng Anh sang tiếng Việt.
# Input: {deepseek_output}
# Output:
#         """

#         response_llama = requests.post(
#             OLLAMA_SERVER,
#             json={"prompt": prompt_llama, "model": MODEL_LLAMA, "stream": False},
#         )
#         response_llama.raise_for_status()
#         llama_output = response_llama.json().get("response", "").strip()

        return jsonify({
            "query": query_text,
            "retrieved_context": context,
            "reasoning_deepseek": deepseek_output_think,
            "final_answer": deepseek_output_answer
        }), 200

    except Exception as e:
        logger.error(f"[RAG Error] {e}", exc_info=True) # Log traceback
        return jsonify({"error": f"RAG failed: {str(e)}"}), 500



@app.route("/ingest", methods=["GET"])
def ingest():
    embedder = Embedder(chroma_path="./chroma_storage")
    # Get md files
    md_files = get_md_files("./data/en")
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
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)