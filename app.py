from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from scr.gemini_rag import rag

app = FastAPI(title="SIDAS RAG API")

# CORS Ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production'da specific domain'ler eklenecek
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str


# Health check endpoint
@app.get("/")
def root():
    return {
        "message": "SIDAS RAG API is running",
        "status": "ok",
        "endpoints": {
            "chat": "/chat (POST)"
        }
    }

# Chat endpoint - Sadece RAG pipeline
@app.post("/chat")
def chat(req: ChatRequest):
    """
    RAG (Retrieval Augmented Generation) endpoint.
    Konum işlemleri frontend'de yapılıyor, bu endpoint sadece bilgi sorularına cevap verir.
    """
    try:
        print(f"[Soru alındı]: {req.question}")

        # RAG pipeline çalıştır
        answer = rag(req.question, verbose=False)

        if not answer or "cevap" not in answer:
            raise ValueError("Modelden geçerli yanıt alınamadı.")

        cevap_text = answer.get("cevap", "Cevap üretilemedi.")
        kaynak = answer.get("kaynak", "SIDAS RAG")

        print(f"[RAG Yanıtı]: {cevap_text[:100]}...")

        return {
            "answer": cevap_text,
            "sources": kaynak
        }

    except Exception as e:
        print(f"[Hata]: {e}")
        raise HTTPException(status_code=500, detail=f"Hata: {str(e)}")
