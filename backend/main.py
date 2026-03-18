import os
import json
import secrets
from typing import Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np


load_dotenv()


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError("OPENAI_API_KEY non trouvé dans l'environnement")


client = OpenAI(api_key=api_key)

app_access_key = os.getenv("APP_ACCESS_KEY") or ""


rag_store_path = os.path.join(os.path.dirname(__file__), "rag_store.json")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    mode: str | None = "multi"
    use_rag: bool | None = False


class ChatResponse(BaseModel):
    reply: str
    trace: list[dict]


class RagDocumentIn(BaseModel):
    title: str
    content: str


class RagDocumentOut(BaseModel):
    id: str
    title: str
    tokens: int


def load_rag_store() -> list[dict]:
    if not os.path.isfile(rag_store_path):
        return []
    try:
        with open(rag_store_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def save_rag_store(docs: list[dict]) -> None:
    with open(rag_store_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)


def embed_text(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text],
    )
    return response.data[0].embedding


def select_rag_context(query: str, max_docs: int = 3) -> str:
    docs = load_rag_store()
    if not docs:
        return ""
    query_vec = np.array(embed_text(query), dtype="float32")
    scored: list[tuple[float, dict[str, Any]]] = []
    for d in docs:
        vec = np.array(d.get("embedding", []), dtype="float32")
        if vec.size == 0 or vec.shape != query_vec.shape:
            continue
        num = float(np.dot(query_vec, vec))
        den = float(np.linalg.norm(query_vec) * np.linalg.norm(vec))
        if den == 0:
            continue
        scored.append((num / den, d))
    if not scored:
        return ""
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [d for _, d in scored[:max_docs]]
    parts: list[str] = []
    for d in top:
        title = d.get("title") or ""
        content = d.get("content") or ""
        snippet = content[:1200]
        parts.append(f"[{title}]\n{snippet}")
    return "\n\n".join(parts).strip()


def call_agent(name: str, system_prompt: str, history: list[dict], user_content: str) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_content})

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": m["role"],
                "content": m["content"],
            }
            for m in messages
        ],
    )

    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    chunks: list[str] = []
    output = getattr(response, "output", None) or []
    for item in output:
        for part in getattr(item, "content", None) or []:
            t = getattr(part, "text", None)
            if isinstance(t, str):
                if t.strip():
                    chunks.append(t)
                continue
            v = getattr(t, "value", None)
            if isinstance(v, str) and v.strip():
                chunks.append(v)
    return "\n".join(chunks).strip()


@app.get("/")
def root():
    docs = load_rag_store()
    return {"status": "ok", "rag_documents": len(docs)}

def require_app_key(req: Request) -> None:
    if not app_access_key:
        return
    provided = req.headers.get("x-app-key") or ""
    if not secrets.compare_digest(provided, app_access_key):
        raise HTTPException(status_code=401, detail="unauthorized")


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest, req: Request):
    require_app_key(req)
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages requis")

    mode = (request.mode or "multi").lower()
    use_rag = bool(request.use_rag)

    trace: list[dict] = []

    user_message = request.messages[-1].content

    rag_context = ""
    if use_rag:
        rag_context = select_rag_context(user_message)
        if rag_context:
            trace.append({"agent": "rag-retriever", "output": rag_context})

    if mode == "single":
        if rag_context:
            prompt = "Contexte pertinent:\n" + rag_context + "\n\nTu es un assistant unique. Réponds directement à la demande de l'utilisateur, en français, en t'appuyant si possible sur le contexte ci-dessus."
        else:
            prompt = "Tu es un assistant unique. Réponds directement à la demande de l'utilisateur, en français, de manière claire et structurée."
        output = call_agent("single", prompt, [], user_message)
        trace.append({"agent": "single", "output": output})
        return ChatResponse(reply=output, trace=trace)

    if rag_context:
        planner_prompt = "Contexte pertinent:\n" + rag_context + "\n\nTu es un planificateur. Tu décomposes la demande en étapes claires pour une équipe d'agents, en t'appuyant sur le contexte lorsque c'est utile. Réponds en français, en quelques phrases structurées."
    else:
        planner_prompt = "Tu es un planificateur. Tu décomposes la demande en étapes claires pour une équipe d'agents. Réponds en français, en quelques phrases structurées."
    planner_output = call_agent("planner", planner_prompt, [], user_message)
    trace.append({"agent": "planner", "output": planner_output})

    if rag_context:
        researcher_prompt = "Contexte pertinent:\n" + rag_context + "\n\nTu es un agent d'analyse qui réfléchit à la demande et complète le plan du planificateur avec des idées et des détails utiles, en t'appuyant sur le contexte lorsque c'est pertinent. Réponds en français."
    else:
        researcher_prompt = "Tu es un agent d'analyse qui réfléchit à la demande et complète le plan du planificateur avec des idées et des détails utiles. Réponds en français."
    researcher_output = call_agent(
        "researcher",
        researcher_prompt,
        [{"role": "assistant", "content": planner_output}],
        user_message,
    )
    trace.append({"agent": "researcher", "output": researcher_output})

    if rag_context:
        synthesizer_prompt = "Contexte pertinent:\n" + rag_context + "\n\nTu es un agent synthétiseur. En te basant sur le contexte, le plan du planificateur et l'analyse de l'agent d'analyse, rédige une réponse finale claire et structurée pour l'utilisateur, en français."
    else:
        synthesizer_prompt = "Tu es un agent synthétiseur. En te basant sur le plan du planificateur et l'analyse de l'agent d'analyse, rédige une réponse finale claire et structurée pour l'utilisateur, en français."
    synthesizer_history = [
        {"role": "assistant", "content": planner_output},
        {"role": "assistant", "content": researcher_output},
    ]
    final_output = call_agent("synthesizer", synthesizer_prompt, synthesizer_history, user_message)
    trace.append({"agent": "synthesizer", "output": final_output})

    return ChatResponse(reply=final_output, trace=trace)


@app.post("/api/rag/documents", response_model=RagDocumentOut)
def add_rag_document(doc: RagDocumentIn, req: Request):
    require_app_key(req)
    if not doc.title.strip():
        raise HTTPException(status_code=400, detail="title requis")
    if not doc.content.strip():
        raise HTTPException(status_code=400, detail="content requis")
    embedding = embed_text(doc.content)
    store = load_rag_store()
    new_id = f"doc-{len(store) + 1}"
    record = {
        "id": new_id,
        "title": doc.title.strip(),
        "content": doc.content,
        "embedding": embedding,
        "tokens": len(doc.content.split()),
    }
    store.append(record)
    save_rag_store(store)
    return RagDocumentOut(id=new_id, title=record["title"], tokens=record["tokens"])


@app.get("/api/rag/documents", response_model=list[RagDocumentOut])
def list_rag_documents(req: Request):
    require_app_key(req)
    store = load_rag_store()
    result: list[RagDocumentOut] = []
    for d in store:
        result.append(
            RagDocumentOut(
                id=str(d.get("id") or ""),
                title=str(d.get("title") or ""),
                tokens=int(d.get("tokens") or 0),
            )
        )
    return result


static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

if os.path.isdir(static_dir):
    app.mount("/app", StaticFiles(directory=static_dir, html=True), name="frontend")


@app.get("/web")
def serve_frontend():
    index_path = os.path.join(static_dir, "index.html")
    if not os.path.isfile(index_path):
        raise HTTPException(status_code=404, detail="Frontend introuvable")
    return FileResponse(index_path)

