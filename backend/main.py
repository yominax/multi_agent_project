import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI


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


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    mode: str | None = "multi"


class ChatResponse(BaseModel):
    reply: str
    trace: list[dict]


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
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages requis")

    mode = (request.mode or "multi").lower()

    trace: list[dict] = []

    user_message = request.messages[-1].content

    if mode == "single":
        prompt = "Tu es un assistant unique. Réponds directement à la demande de l'utilisateur, en français, de manière claire et structurée."
        output = call_agent("single", prompt, [], user_message)
        trace.append({"agent": "single", "output": output})
        return ChatResponse(reply=output, trace=trace)

    planner_prompt = "Tu es un planificateur. Tu décomposes la demande en étapes claires pour une équipe d'agents. Réponds en français, en quelques phrases structurées."
    planner_output = call_agent("planner", planner_prompt, [], user_message)
    trace.append({"agent": "planner", "output": planner_output})

    researcher_prompt = "Tu es un agent d'analyse qui réfléchit à la demande et complète le plan du planificateur avec des idées et des détails utiles. Réponds en français."
    researcher_output = call_agent(
        "researcher",
        researcher_prompt,
        [{"role": "assistant", "content": planner_output}],
        user_message,
    )
    trace.append({"agent": "researcher", "output": researcher_output})

    synthesizer_prompt = "Tu es un agent synthétiseur. En te basant sur le plan du planificateur et l'analyse de l'agent d'analyse, rédige une réponse finale claire et structurée pour l'utilisateur, en français."
    synthesizer_history = [
        {"role": "assistant", "content": planner_output},
        {"role": "assistant", "content": researcher_output},
    ]
    final_output = call_agent("synthesizer", synthesizer_prompt, synthesizer_history, user_message)
    trace.append({"agent": "synthesizer", "output": final_output})

    return ChatResponse(reply=final_output, trace=trace)


static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

if os.path.isdir(static_dir):
    app.mount("/app", StaticFiles(directory=static_dir, html=True), name="frontend")


@app.get("/web")
def serve_frontend():
    index_path = os.path.join(static_dir, "index.html")
    if not os.path.isfile(index_path):
        raise HTTPException(status_code=404, detail="Frontend introuvable")
    return FileResponse(index_path)

