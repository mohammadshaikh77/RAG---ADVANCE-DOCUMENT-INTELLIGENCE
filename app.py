import os, io, base64, re, json, shutil
import fitz                      # PyMuPDF
import faiss
import numpy as np
import streamlit as st
from PIL import Image
from groq import Groq
from sentence_transformers import SentenceTransformer
from core_services import DBManager, DB_DIR

VISION_MODEL = os.environ.get("VISION_MODEL", "qwen/qwen3.8-27b")
IMG_DIR = "app_images"
CHUNK_WORDS, OVERLAP, MIN_SIZE = 200, 40, 100

client = Groq(api_key=os.environ["GROQ_API_KEY"])

@st.cache_resource
def load_resources():
    return SentenceTransformer("all-MiniLM-L6-v2"), DBManager()

def chunk_text(text, page_num):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunks.append({"page": page_num, "text": " ".join(words[start:start + CHUNK_WORDS])})
        start += CHUNK_WORDS - OVERLAP
    return chunks

def generate_image_caption(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((1024, 1024))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64_str = base64.b64encode(buf.getvalue()).decode()

        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this technical diagram, chart, or visual asset in detail. List key entities, relationships, flow steps, and readable text contents. Be concise but highly technical."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}
                    }
                ]
            }],
            max_tokens=300,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return ""

def process_pdf(pdf_bytes, text_model, db_manager):
    shutil.rmtree(IMG_DIR, ignore_errors=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text_chunks, images, seen, rendered = [], [], set(), 0

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            text_chunks.extend(chunk_text(text, page_num))

        found = 0
        for i, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            if xref in seen:
                continue
            seen.add(xref)
            pix = fitz.Pixmap(doc, xref)
            if pix.width < MIN_SIZE or pix.height < MIN_SIZE:
                continue
            if pix.n - pix.alpha >= 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            path = f"{IMG_DIR}/page{page_num}_img{i}.png"
            pix.save(path)
            images.append({"page": page_num, "path": path})
            found += 1

        if found == 0 and len(page.get_drawings()) > 20 and rendered < 10:
            path = f"{IMG_DIR}/page{page_num}_render.png"
            page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(path)
            images.append({"page": page_num, "path": path})
            rendered += 1

    # Generate captions
    image_captions = []
    for img in images:
        caption = generate_image_caption(img["path"])
        img["caption"] = caption
        image_captions.append(caption if caption else "empty technical description")

    # Store Transactional Metadata inside relational DB
    db_manager.save_metadata(text_chunks, images)

    # Generate Vectors and save FAISS Indexes
    t_emb = text_model.encode([c["text"] for c in text_chunks], normalize_embeddings=True)
    text_index = faiss.IndexFlatIP(t_emb.shape[1])
    text_index.add(np.array(t_emb).astype("float32"))
    faiss.write_index(text_index, os.path.join(DB_DIR, "text.index"))

    if image_captions:
        i_emb = text_model.encode(image_captions, normalize_embeddings=True)
        image_index = faiss.IndexFlatIP(i_emb.shape[1])
        image_index.add(np.array(i_emb).astype("float32"))
        faiss.write_index(image_index, os.path.join(DB_DIR, "image.index"))
    else:
        image_index = None

    return {"chunks": text_chunks, "images": images, "text_index": text_index, "image_index": image_index}

def retrieve(question, kb, text_model, min_score_threshold=0.25):
    q_vector = text_model.encode([question], normalize_embeddings=True).astype("float32")

    scores, ids = kb["text_index"].search(q_vector, min(3, kb["text_index"].ntotal))
    text_hits = [{"page": kb["chunks"][i]["page"], "text": kb["chunks"][i]["text"], "score": float(s)}
                 for s, i in zip(scores[0], ids[0])]

    image_hits = []
    if kb["image_index"] is not None:
        scores, ids = kb["image_index"].search(q_vector, min(2, kb["image_index"].ntotal))
        for s, i in zip(scores[0], ids[0]):
            score = float(s)
            if score >= min_score_threshold:
                image_hits.append({
                    "page": kb["images"][i]["page"],
                    "path": kb["images"][i]["path"],
                    "score": score,
                    "caption": kb["images"][i].get("caption", "")
                })
    return text_hits, image_hits

def answer(question, text_hits, image_hits):
    context = "\n\n".join(f"[Page {h['page']}] {h['text']}" for h in text_hits)
    prompt = f"""You are an expert system answering user questions about a technical document.
Use ONLY the text excerpts and image descriptions provided below to construct your answer.
If the answer is not present, respond that you could not locate the details.
Always cite the source page numbers in your answer.

TEXT EXCERPTS:
{context}

QUESTION: {question}"""

    content = [{"type": "text", "text": prompt}]
    for h in image_hits:
        img = Image.open(h["path"]).convert("RGB")
        img.thumbnail((1024, 1024))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    resp = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=800,
        temperature=0.2,
    )
    text = resp.choices[0].message.content or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# ---------- Streamlit Custom Beautiful Styling ----------
st.set_page_config(page_title="Archival-RAG | Document Intelligence Studio", page_icon="🧠", layout="wide")

# Inject Custom CSS for stunning styling, card elevations, color accents and high density interfaces
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main typography and backgrounds */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* App Header Title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #58a6ff 0%, #bc8cff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    
    .sub-caption {
        font-size: 1rem;
        color: #8b949e;
        margin-bottom: 2rem;
    }
    
    /* Styled Cards for citations */
    .citation-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    .citation-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #21262d;
        padding-bottom: 0.5rem;
        margin-bottom: 0.75rem;
    }
    
    .badge-page {
        background-color: #1f6feb;
        color: #ffffff;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
    }
    
    .badge-score {
        color: #58a6ff;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    /* Navigation and Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d !important;
    }
    
    /* Custom info boxes */
    .custom-info {
        background-color: #1f242c;
        border-left: 4px solid #388bfd;
        border-radius: 4px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Align inputs nicely */
    .stTextInput>div>div>input {
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🧠 Archival-RAG: Advanced Document Intelligence</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-caption">Enterprise-Grade Multimodal Vector Intelligence Studio with Hybrid Cross-Modal Query Synthesis</p>', unsafe_allow_html=True)

text_model, db_manager = load_resources()

# Warm Load DB on Startup
if "kb" not in st.session_state:
    text_idx_path = os.path.join(DB_DIR, "text.index")
    if os.path.exists(text_idx_path):
        meta = db_manager.get_metadata()
        t_idx = faiss.read_index(text_idx_path)
        img_idx_path = os.path.join(DB_DIR, "image.index")
        i_idx = faiss.read_index(img_idx_path) if os.path.exists(img_idx_path) else None
        st.session_state.kb = {
            "chunks": meta["chunks"],
            "images": meta["images"],
            "text_index": t_idx,
            "image_index": i_idx
        }
        st.sidebar.success("⚡ Relational DB and Vector Files Warm Loaded!")

with st.sidebar:
    st.markdown("### 🛠️ Ingestion Control Platform")
    st.markdown("Upload and index technical manuals, schematic layouts, or vector graphics.")
    uploaded = st.file_uploader("Upload Enterprise PDF", type="pdf")
    if uploaded and st.button("Run Ingestion Pipeline", type="primary", use_container_width=True):
        with st.spinner("Processing text elements, computing visual descriptions, and building persistent FAISS stores..."):
            st.session_state.kb = process_pdf(uploaded.getvalue(), text_model, db_manager)
            st.session_state.history = []
            st.success("Pipeline built and stored in local Relational + Vector DB!")

if "kb" not in st.session_state:
    st.markdown("""
    <div class="custom-info">
        <h4>👋 Welcome to Document Intelligence Studio</h4>
        <p>No document database is initialized on this node yet. Use the <b>Ingestion Control Platform</b> in the sidebar to load your enterprise PDF.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

# Main chat rendering
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Submit a technical query to the document storage...")
if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # --- STEP 1: Thread-Safe Semantic Cache Lookup ---
        q_vector = text_model.encode([question], normalize_embeddings=True)[0].astype("float32")
        cached_response = db_manager.cache_lookup(question, q_vector, text_model)

        if cached_response:
            st.markdown(cached_response)
            st.markdown("⚡ **Served from instant Semantic Cache (Sub-5ms response)**")
            st.session_state.history.append({"role": "user", "content": question})
            st.session_state.history.append({"role": "assistant", "content": cached_response})
        else:
            with st.spinner("Scanning database and computing cross-modal synthesis..."):
                text_hits, image_hits = retrieve(question, st.session_state.kb, text_model)
                try:
                    reply = answer(question, text_hits, image_hits)
                    # Write to the semantic cache store
                    db_manager.cache_store(question, q_vector, reply)
                except Exception as e:
                    reply = f"Error: {e}"
            st.markdown(reply)

            # --- Detailed Citation Panel (Sleek UI Cards) ---
            st.markdown("--- ")
            st.markdown("### 🛡️ Verified Verification and Source Citations")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📄 Textual Grounding Evidence")
                for h in text_hits:
                    st.markdown(f"""
                    <div class="citation-card">
                        <div class="citation-header">
                            <span class="badge-page">Page {h['page']}</span>
                            <span class="badge-score">Match Confidence: {h['score']:.4f}</span>
                        </div>
                        <p style="font-size:0.9rem; line-height:1.5; color:#c9d1d9;">{h['text']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            with col2:
                st.markdown("#### 🖼️ Visual Context Extraction")
                if image_hits:
                    for h in image_hits:
                        st.image(h["path"], use_container_width=True)
                        st.markdown(f"""
                        <div class="citation-card" style="margin-top:-10px; border-top-left-radius:0; border-top-right-radius:0;">
                            <div class="citation-header">
                                <span class="badge-page" style="background-color:#8b949e;">Page {h['page']}</span>
                                <span class="badge-score">Description Similarity: {h['score']:.4f}</span>
                            </div>
                            <p style="font-size:0.85rem; color:#8b949e; font-style:italic;"><b>Extracted Metadata:</b> {h['caption'][:220]}...</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("No diagrammatic images met the query similarity threshold validation.")

            st.session_state.history.append({"role": "user", "content": question})
            st.session_state.history.append({"role": "assistant", "content": reply})
