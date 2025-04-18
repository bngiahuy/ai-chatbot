import re
import unicodedata
from urllib import response
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
from llama_index.llms.ollama import Ollama
load_dotenv()

logger = logging.getLogger(__name__)
app = Flask(__name__)
MODEL_DEEPSEEK = 'deepseek-r1:7b'
MODEL_LLAMA = 'llama3.2'
OLLAMA_SERVER = os.environ.get("OLLAMA_SERVER", "http://localhost:11434/api/generate")
# Initialize the embedder with the same ChromaDB path from your main script
embedder = Embedder(chroma_path="./chroma_storage")

# Setup logging
setup_logging()

# Helper function
def run_vector_search(query_text: str, n_results: int = 5, metadata_filters=None):
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
    isRewrite = bool(request_data.get("is_rewrite", False))

    if not query_text:
        return jsonify({"error": "Missing or empty 'query' parameter"}), 400
    
    query_text = unicodedata.normalize("NFC", query_text)

    try:
        n_results = int(request_data.get("n_results", 10))
        metadata_filters = request_data.get("metadata_filters", None)

        logger.info(f"Search: '{query_text}' (n_results={n_results})")

        if isRewrite: 
            query_text = rewrite_query(query_text)  # Rewrite the query for better results

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
    n_results = int(request_data.get("n_results", 5)) # Số kết quả ban đầu
    isRewrite = bool(request_data.get("is_rewrite", False))

    if not query_text:
        return jsonify({"error": "Missing 'query' parameter"}), 400

    query_text = unicodedata.normalize("NFC", query_text)
    if isRewrite:
        query_text = rewrite_query(query_text)

    try:
        # Step 1a: Initial search to find relevant files
        initial_results = run_vector_search(query_text, n_results=n_results)

        if not initial_results or not initial_results.get("ids") or not initial_results["ids"][0]:
            logger.warning(f"[RAG] No relevant chunks found initially for query: {query_text}")
            # Trả về phản hồi không tìm thấy thông tin
            return jsonify({
                "query": query_text, "retrieved_context": "",
                "reasoning_deepseek": "I don't know based on the provided context.",
                "final_answer": "Xin lỗi, tôi không tìm thấy thông tin cụ thể để trả lời câu hỏi này."
            }), 200

        retrieved_docs = initial_results.get("documents", [[]])[0]
        retrieved_chunks = [doc for doc in retrieved_docs if doc and doc.strip()]
        context = "\n".join(retrieved_chunks)


        # Step 2 & 3: (Giữ nguyên phần gọi DeepSeek và Llama)
        # Step 2: Ask DeepSeek for reasoning
        prompt_deepseek = f"""
        Context information is below. 
        ---
        {context}
        ---
        Given the context information and not prior knowledge, answer the query.
        Query: {query_text}
        ---
        Constraints:
        - You are a helpful assistant created by Bùi Nguyễn Gia Huy.
        - MUST use English language in any response.
        - Do not include any disclaimers or unnecessary information.
        - Be concise and specific.
        - Use Markdown format for the answer.
        - This is very important to my career. You'd better be careful with the answer.
        """

        response_deepseek = requests.post(
            OLLAMA_SERVER,
            json={
                "prompt": prompt_deepseek, 
                "model": MODEL_DEEPSEEK, 
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "seed": 123,
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
            "reasoning": deepseek_output_think,
            "final_answer": deepseek_output_answer
            # "final_answer": deepseek_output,
        }), 200

    except Exception as e:
        logger.error(f"[RAG Error] {e}", exc_info=True) # Log traceback
        return jsonify({"error": f"RAG failed: {str(e)}"}), 500

def rewrite_query(query_text):
    """
    Rewrite the query to be more specific and detailed.
    
    Args:
        query_text (str): The original query text.
        
    Returns:
        str: The rewritten query text.
    """
    query_text = unicodedata.normalize("NFC", query_text)
    prompt = f"""
    You're an assistant helping reformulate user query for vector search.

    Given a user query, rewrite it into a clear, complete, and context-rich query that is suitable for semantic vector search. Response only the rewritten query without any additional text or explanation.
    - Use English language.
    - Keep it concise but specific.
    - Expand vague terms into corporate-specific language if possible.
    - Do not change the meaning of the query.

    Original query:
    {query_text}

    Rewritten Query:
    """
    
    response = requests.post(
        OLLAMA_SERVER,
        json={
            "prompt": prompt, 
            "model": MODEL_LLAMA, 
            "stream": False,
            "options": {
                "temperature": 0,
                "seed": 123,
            }
        },
    )
    
    response.raise_for_status()
    return response.json().get("response", "").strip()

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

@app.route("/ollama/search", methods=["POST"])
def ollama_search():
    request_data = request.get_json() or {}
    query_text = request_data.get("query", "").strip()
    llm = Ollama(base_url="http://10.6.18.2:11434", model=MODEL_DEEPSEEK, request_timeout=120)
    response = llm.complete(query_text)
    if response:
        return jsonify({"response": response.text}), 200
    else:
        return jsonify({"error": "No response from LLM"}), 500


def main():
    # Ensure directories exist
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./chroma_storage", exist_ok=True)
    
if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)