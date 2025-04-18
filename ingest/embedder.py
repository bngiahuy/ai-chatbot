from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import logging
import os
from ingest.text_extractor import extract_text_from_markdown
from ingest.text_chunker import split_text
from chromadb.utils import embedding_functions
MODEL_EMBEDDING = "sentence-transformers/all-mpnet-base-v2"

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=MODEL_EMBEDDING,
)

logger = logging.getLogger(__name__)

class Embedder:
    def __init__(self, model_name=MODEL_EMBEDDING, chroma_path="./chroma_storage"):
        """
        Initialize the embedder with the specified model and ChromaDB client.
        Args:
            model_name (str): HuggingFace model name or local path.
            chroma_path (str): Path to store ChromaDB data.
        """
        # Create local model directory
        local_models_dir = "./models"
        os.makedirs(local_models_dir, exist_ok=True)

        model_dir_name = model_name.split("/")[-1]
        local_model_path = os.path.join(local_models_dir, model_dir_name)

        # Load model: nếu model đã tồn tại local, dùng path; không thì để HuggingFace auto tải về
        if os.path.exists(local_model_path):
            logger.info(f"Using cached model from: {local_model_path}")
            self.model = SentenceTransformer(local_model_path)
        else:
            logger.info(f"Downloading and caching model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.model.save(local_model_path)  # Lưu lại sau khi đã load model

        # Khởi tạo ChromaDB
        self.client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            "hsv_document_chunks", 
            embedding_function=sentence_transformer_ef,
            metadata={
                "hnsw:space": "cosine", # Cosine distance
                "hnsw:M": 32 , # Number of bi-directional links created for every new element
                "hnsw:construction_ef": 200, # Construction ef is the size of the dynamic list of neighbors
                "hnsw:search_ef": 200, # Search ef is the size of the dynamic list of neighbors during search
            }
        )
    
    def encode_passage(self, text):
        """
        Add the required prefix for passages before encoding with the e5 model.
        
        Args:
            text (str): The text to encode
            
        Returns:
            list: The embedding vector
        """
        # return self.model.encode(f"passage: {text}", normalize_embeddings=True).tolist()
        return self.model.encode(text, normalize_embeddings=True, convert_to_tensor=True).tolist()
    
    def encode_query(self, text):
        """
        Add the required prefix for queries before encoding with the e5 model.
        
        Args:
            text (str): The query text to encode
            
        Returns:
            list: The embedding vector
        """
        # return self.model.encode(f"query: {text}", normalize_embeddings=True).tolist()
        return self.model.encode(text, normalize_embeddings=True, convert_to_tensor=True).tolist()
        
    def embed_chunks(self, chunks, metadata_list):
        """
        Embed chunks and store them in ChromaDB with metadata.
        
        Args:
            chunks (list): List of text chunks.
            metadata_list (list): List of metadata dictionaries.
            
        Returns:
            int: Number of embeddings stored.
        """
        if not chunks:
            logger.warning("No chunks to embed")
            return 0
            
        # Generate IDs
        ids = [f"{meta['filename']}_{meta['page']}_{meta['chunk_index']}" for meta in metadata_list]
        
        # Generate embeddings with passage prefix
        logger.info(f"Embedding {len(chunks)} chunks")
        embeddings = [self.encode_passage(chunk) for chunk in chunks]
        
        # Store in ChromaDB
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadata_list,
            ids=ids
        )
        
        logger.info(f"Stored {len(chunks)} embeddings in ChromaDB")
        return len(chunks)
    
    def embed_markdown_file(self, markdown_path):
        """
        Extract, process, chunk, and embed content from a markdown file.
        
        Args:
            markdown_path (str): Path to the markdown file.
            
        Returns:
            int: Number of embeddings stored.
        """
        logger.info(f"Processing markdown file: {markdown_path}")
        
        # 1. Extract text from markdown file
        markdown_content = extract_text_from_markdown(markdown_path)
        if not markdown_content:
            logger.warning(f"Could not extract content from {markdown_path}")
            return 0
        
        # 2. Get file information
        filename = os.path.basename(markdown_path)
        file_size = os.path.getsize(markdown_path)
        
        # 3. Split content into chunks
        chunks = split_text(markdown_content, verbose=True)
        
        if not chunks:
            logger.warning(f"No chunks created from {markdown_path}")
            return 0
        
        logger.info(f"Created {len(chunks)} chunks from {filename}")
        
        # 4. Create metadata for each chunk
        metadata_list = []
        for idx, chunk in enumerate(chunks):
            # Calculate next and previous chunk indices
            prev_idx = idx - 1 if idx > 0 else -1
            next_idx = idx + 1 if idx < len(chunks) - 1 else -1
            
            # Create metadata
            metadata = {
                "filename": filename,
                "file_path": markdown_path,
                "file_size": file_size,
                "page": 0,  # Markdown files don't have pages, use 0 as default
                "chunk_index": idx,
                "chunk_size": len(chunk),
                "total_chunks": len(chunks),
                "prev_chunk_index": prev_idx,
                "next_chunk_index": next_idx
            }
            metadata_list.append(metadata)
        
        # 5. Embed and store chunks
        return self.embed_chunks(chunks, metadata_list)