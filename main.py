import os
import json
import base64
from typing import Literal

import firebase_admin
from firebase_admin import auth, credentials
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field


app = FastAPI(title="Emily AI Server", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
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


def initialize_firebase():
    try:
        firebase_admin.get_app()
        return
    except ValueError:
        pass

    encoded_key = os.getenv("FIREBASE_SERVICE_ACCOUNT_BASE64")
    if not encoded_key:
        raise RuntimeError(
            "FIREBASE_SERVICE_ACCOUNT_BASE64 is not configured."
        )

    try:
        key_json = base64.b64decode(encoded_key).decode("utf-8")
        service_account_info = json.loads(key_json)
        certificate = credentials.Certificate(service_account_info)
        firebase_admin.initialize_app(certificate)
    except Exception as exc:
        raise RuntimeError(
            "Firebase Admin SDK configuration is invalid."
        ) from exc


initialize_firebase()


def verify_firebase_user(authorization: str | None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Firebase authentication token is required.",
        )

    token = authorization[len("Bearer "):].strip()

    try:
        decoded_token = auth.verify_id_token(token, check_revoked=True)
        return decoded_token["uid"]
    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired Firebase authentication token.",
        )


@app.get("/")
def root():
    return {"service": "Emily AI Server", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    authorization: str | None = Header(default=None),
):
    user_id = verify_firebase_user(authorization)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API is not configured.",
        )

    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = OpenAI(api_key=api_key)

    input_messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
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
            raise HTTPException(
                status_code=502,
                detail="The AI returned an empty response.",
            )

        return ChatResponse(reply=reply)

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="The AI request failed. Check server logs and API access settings.",
        )
