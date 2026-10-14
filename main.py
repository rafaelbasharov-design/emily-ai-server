import os
from typing import Literal

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field

app = FastAPI(title="Emily AI Server", version="0.1.0")

# Temporary development CORS policy. Tighten allowed origins before production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Emily-App-Token"],
)

SYSTEM_PROMPT = (
    "You are Emily, a warm, thoughtful, friendly virtual companion. "
    "Speak naturally and respectfully. Be honest that you are an AI companion "
    "and do not claim to be a real human. Reply in the language the user uses."
)

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)

class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=30)

class ChatResponse(BaseModel):
    reply: str

def verify_app_token(provided: str | None) -> None:
    expected = os.getenv("EMILY_APP_TOKEN")
    # Require a token so the endpoint is not unintentionally open to everyone.
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Server setup incomplete: EMILY_APP_TOKEN is not configured.",
        )
    if not provided or provided != expected:
        raise HTTPException(status_code=401, detail="Invalid app token.")

@app.get("/")
def root():
    return {"service": "Emily AI Server", "status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    x_emily_app_token: str | None = Header(default=None),
):
    verify_app_token(x_emily_app_token)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Server setup incomplete: OPENAI_API_KEY is not configured.",
        )

    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = OpenAI(api_key=api_key)

    input_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    input_messages.extend(
        {"role": item.role, "content": item.content}
        for item in request.messages
    )

    try:
        result = client.responses.create(
            model=model,
            input=input_messages,
            max_output_tokens=500,
        )
        reply = result.output_text.strip()
        if not reply:
            raise HTTPException(status_code=502, detail="The AI returned an empty response.")
        return ChatResponse(reply=reply)
    except HTTPException:
        raise
    except Exception:
        # Do not return secrets or raw provider errors to the public client.
        raise HTTPException(
            status_code=502,
            detail="The AI request failed. Check server logs and API access settings.",
        )
