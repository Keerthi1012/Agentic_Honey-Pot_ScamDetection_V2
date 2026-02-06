from fastapi import FastAPI
from agent.controller import handle_agent
from api.routes import router
import pickle
import faiss
import joblib

# from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from contextlib import asynccontextmanager
from fastapi import Request
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "model")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    print("🔄 Loading models into memory...")

    # 1. Load spam classifier
    app.state.spam_model = joblib.load(os.path.join(MODELS_DIR, "spam_model.pkl"))

    # 2. Load TF-IDF vectorizer
    app.state.vectorizer = joblib.load(os.path.join(MODELS_DIR, "vectorizer.pkl"))

    # 3. Load embedder (BERT)
    USE_EMBEDDINGS = False
    if USE_EMBEDDINGS:
        app.state.embedder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    else:
        app.state.embedder = None
        app.state.index = None
        app.state.texts = []

    app.state.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # 4. Load FAISS index (if exists)
    try:
        app.state.index = faiss.read_index(os.path.join(MODELS_DIR, "index.faiss"))
        with open(os.path.join(MODELS_DIR, "texts.pkl"), "rb") as f:
            app.state.texts = pickle.load(f)
        print("✅ FAISS index loaded")
    except:
        app.state.index = faiss.IndexFlatL2(384)
        app.state.texts = []
        print("⚠️ No index found, created new one")

    print("🚀 Startup complete")
    yield
    # Shutdown logic
    print("💾 Saving vector memory...")
    if os.getenv("ENV") == "local":
        faiss.write_index(app.state.index, os.path.join(MODELS_DIR, "index.faiss"))

    with open(os.path.join(MODELS_DIR, "texts.pkl"), "wb") as f:
        pickle.dump(app.state.texts, f)

    print("✅ Saved successfully")


app = FastAPI(title="Agentic Honeypot API", lifespan=lifespan)

# Include our API routes
app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}


def root():
    return {"status": "Agentic Honeypot API running"}
