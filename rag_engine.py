import uuid
import json
import math
import re
import numpy as np
import db
import file_parser


# ==========================================
# MODULAR VECTOR STORE INTERFACE
# ==========================================

class BaseVectorStore:
    def store_chunks(self, file_id, chunks):
        raise NotImplementedError

    def search(self, file_ids, query_embedding, top_k=4):
        raise NotImplementedError


class MySQLVectorStore(BaseVectorStore):
    def store_chunks(self, file_id, chunks):
        db.save_file_chunks(file_id, chunks)

    def search(self, file_ids, query_embedding, top_k=4):
        rows = db.get_chunks_by_file_ids(file_ids)
        if not rows or not query_embedding:
            return []

        q_vec = np.array(query_embedding, dtype=np.float32)
        norm_q = np.linalg.norm(q_vec)
        if norm_q == 0:
            return []
        q_vec = q_vec / norm_q

        scored_chunks = []
        for r in rows:
            emb = r.get('embedding')
            if not emb or len(emb) != len(query_embedding):
                similarity = 0.0
            else:
                c_vec = np.array(emb, dtype=np.float32)
                norm_c = np.linalg.norm(c_vec)
                if norm_c == 0:
                    similarity = 0.0
                else:
                    similarity = float(np.dot(q_vec, c_vec / norm_c))
            
            scored_chunks.append({
                'chunk': r,
                'similarity': similarity
            })

        # Sort descending by similarity score
        scored_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        return scored_chunks[:top_k]


# Instantiate global vector store
vector_store = MySQLVectorStore()


# ==========================================
# EMBEDDING GENERATOR (Feature Hashing Vectorizer)
# ==========================================

VECTOR_DIM = 256

def compute_embedding(text: str) -> list[float]:
    """
    Computes a 256-dimensional feature hashing vector normalized to unit length.
    Provides fast, dependency-free dense vector representations suitable for cosine similarity.
    """
    words = re.findall(r'\w+', text.lower())
    if not words:
        return [0.0] * VECTOR_DIM

    vec = np.zeros(VECTOR_DIM, dtype=np.float32)
    for word in words:
        # Simple string hash mapped into vector dimension
        h = 0
        for char in word:
            h = (h * 31 + ord(char)) % VECTOR_DIM
        vec[h] += 1.0

    # Sub-linear term frequency scaling (1 + log(tf))
    vec = np.where(vec > 0, 1.0 + np.log(np.maximum(vec, 1.0)), 0.0)

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec.tolist()


# ==========================================
# CHUNKING & RAG PIPELINE
# ==========================================

def chunk_text(text, max_chars=1200, overlap=200):
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start += (max_chars - overlap)
    return chunks


def process_and_store_file(file_id, file_path, mime_type, original_name):
    """
    Parses file, chunks text, computes embeddings, and saves chunks to DB.
    Updates file status to 'ready' or 'error'.
    """
    try:
        segments = file_parser.parse_file(file_path, mime_type, original_name)
        if not segments:
            db.update_file_status(file_id, 'error', 'No text content could be extracted from this file.')
            return False

        chunks_to_save = []
        chunk_idx = 0

        for seg in segments:
            raw_text = seg['content']
            sub_chunks = chunk_text(raw_text)

            for sub_text in sub_chunks:
                chunk_id = str(uuid.uuid4())
                emb = compute_embedding(sub_text)
                
                chunks_to_save.append({
                    'id': chunk_id,
                    'chunk_index': chunk_idx,
                    'content': sub_text,
                    'page_number': seg.get('page_number'),
                    'sheet_name': seg.get('sheet_name'),
                    'start_line': seg.get('start_line'),
                    'end_line': seg.get('end_line'),
                    'embedding': emb
                })
                chunk_idx += 1

        vector_store.store_chunks(file_id, chunks_to_save)
        db.update_file_status(file_id, 'ready')
        return True

    except Exception as e:
        print(f"[RAG] Error processing file {file_id}: {e}")
        db.update_file_status(file_id, 'error', str(e))
        return False


def retrieve_relevant_chunks(file_ids, query, top_k=4, min_similarity=0.15):
    """
    Retrieves Top-K relevant chunks for a query across specified file_ids.
    Returns: (results_list, max_similarity_score)
    """
    if not file_ids or not query.strip():
        return [], 0.0

    q_emb = compute_embedding(query)
    search_results = vector_store.search(file_ids, q_emb, top_k=top_k)

    if not search_results:
        return [], 0.0

    max_sim = search_results[0]['similarity']
    filtered_results = [res for res in search_results if res['similarity'] >= min_similarity]

    return filtered_results, max_sim


def format_context_block(retrieved_results):
    """
    Formats retrieved chunks into a clean prompt context block with citations.
    """
    if not retrieved_results:
        return ""

    context_str = "Below are relevant extracts from the attached file(s) for this conversation:\n\n"
    for idx, item in enumerate(retrieved_results, 1):
        chunk = item['chunk']
        filename = chunk.get('original_name', 'Attached Document')
        location_meta = []
        
        if chunk.get('page_number'):
            location_meta.append(f"Page {chunk['page_number']}")
        if chunk.get('sheet_name'):
            location_meta.append(f"Sheet: {chunk['sheet_name']}")
        if chunk.get('start_line') and chunk.get('end_line'):
            location_meta.append(f"Lines {chunk['start_line']}-{chunk['end_line']}")
            
        loc_str = f" ({', '.join(location_meta)})" if location_meta else ""

        context_str += f"--- Source [{idx}]: {filename}{loc_str} ---\n"
        context_str += f"{chunk['content']}\n\n"

    context_str += "Instructions: Base your answer primarily on these file extracts when answering questions about the file. "
    context_str += "If the question is about the file but the extracts do not contain the answer, explicitly state that the uploaded document does not contain that information."

    return context_str
