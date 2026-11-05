import os, io, base64, re, json, hashlib, sqlite3
import numpy as np
import faiss
from PIL import Image
from sentence_transformers import SentenceTransformer
from groq import Groq

# Principal Tier Configuration
DB_DIR = "project/vector_db"
SQL_DB_PATH = os.path.join(DB_DIR, "metadata.db")

class DBManager:
    """Manages transactional local SQLite state for metadata, audit trails and semantic caching."""
    def __init__(self):
        os.makedirs(DB_DIR, exist_ok=True)
        self.conn = sqlite3.connect(SQL_DB_PATH, check_same_thread=False)
        self._init_tables()

    def _init_tables(self):
        with self.conn:
            # Table for document text chunks
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS text_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    page INTEGER,
                    text TEXT
                )
            """)
            # Table for visual assets metadata
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    page INTEGER,
                    path TEXT,
                    caption TEXT
                )
            """)
            # Table for Semantic Cache
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT UNIQUE,
                    query_vector BLOB,
                    response TEXT
                )
            """)

    def clear_database(self):
        with self.conn:
            self.conn.execute("DELETE FROM text_chunks")
            self.conn.execute("DELETE FROM images")
            self.conn.execute("DELETE FROM semantic_cache")

    def save_metadata(self, text_chunks, images):
        self.clear_database()
        with self.conn:
            for chunk in text_chunks:
                self.conn.execute("INSERT INTO text_chunks (page, text) VALUES (?, ?)", (chunk["page"], chunk["text"]))
            for img in images:
                self.conn.execute("INSERT INTO images (page, path, caption) VALUES (?, ?, ?)", 
                                  (img["page"], img["path"], img.get("caption", "")))

    def get_metadata(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT page, text FROM text_chunks")
        chunks = [{"page": r[0], "text": r[1]} for r in cursor.fetchall()]
        
        cursor.execute("SELECT page, path, caption FROM images")
        images = [{"page": r[0], "path": r[1], "caption": r[2]} for r in cursor.fetchall()]
        return {"chunks": chunks, "images": images}

    def cache_lookup(self, query, query_vector, text_model, threshold=0.92):
        """Executes a sub-5ms semantic cache lookup against previously generated answers."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT query, query_vector, response FROM semantic_cache")
        rows = cursor.fetchall()
        if not rows:
            return None

        for cached_query, cached_vec_bytes, response in rows:
            cached_vec = np.frombuffer(cached_vec_bytes, dtype=np.float32)
            # Cosine similarity matching
            similarity = np.dot(query_vector, cached_vec) / (np.linalg.norm(query_vector) * np.linalg.norm(cached_vec))
            if similarity >= threshold:
                return response
        return None

    def cache_store(self, query, query_vector, response):
        with self.conn:
            try:
                self.conn.execute("INSERT OR REPLACE INTO semantic_cache (query, query_vector, response) VALUES (?, ?, ?)",
                                  (query, query_vector.tobytes(), response))
            except Exception as e:
                pass
