import logging
import os
import uuid
from urllib.parse import urlparse

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from firebase_admin import auth, firestore
from supabase import create_client

router = APIRouter()

SUPABASE_BUCKET = "emily-private-audio"
AUDIO_PREFIX = "audio"
MAX_AUDIO_SIZE = 25 * 1024 * 1024
SIGNED_URL_TTL = 300

ALLOWED_AUDIO_TYPES = {
    "audio/mp4",
    "audio/m4a",
    "audio/x-m4a",
    "audio/mpeg",
    "audio/wav",
    "audio/x-wav",
    "audio/webm",
    "audio/ogg",
}

AUDIO_EXTENSIONS = {
    "audio/mp4": ".m4a",
    "audio/m4a": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
}


def verified_uid(authorization: str | None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required.")

    try:
        token = authorization.removeprefix("Bearer ").strip()
        return auth.verify_id_token(token, check_revoked=True)["uid"]
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid authentication token.")


def get_supabase():
    supabase_url = os.getenv("SUPABASE_URL")
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_url or not service_role_key:
        raise HTTPException(
            status_code=503,
            detail="Audio storage unavailable.",
        )

    try:
        return create_client(supabase_url, service_role_key)
    except Exception:
        logging.exception("Supabase client initialization failed")
        raise HTTPException(
            status_code=503,
            detail="Audio storage unavailable.",
        )


def valid_audio_path(path: object, uid: str) -> bool:
    if not isinstance(path, str) or not path:
        return False

    if path.startswith("/") or "\\" in path:
        return False

    parts = path.split("/")

    if any(part in ("", ".", "..") for part in parts):
        return False

    return (
        len(parts) == 3
        and parts[0] == AUDIO_PREFIX
        and parts[1] == uid
    )


@router.post("/audio/upload")
def upload_private_audio(
    message_id: str = Form(...),
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
):
    uid = verified_uid(authorization)

    if not message_id or len(message_id) > 150:
        raise HTTPException(status_code=400, detail="Invalid message ID.")

    if file.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported audio format.")

    audio = file.file.read(MAX_AUDIO_SIZE + 1)

    if not audio:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    if len(audio) > MAX_AUDIO_SIZE:
        raise HTTPException(status_code=413, detail="Audio file too large.")

    document = (
        firestore.client()
        .collection("users")
        .document(uid)
        .collection("messages")
        .document(message_id)
    )

    snapshot = document.get()

    if not snapshot.exists:
        raise HTTPException(status_code=404, detail="Message not found.")

    message = snapshot.to_dict() or {}

    if message.get("role") != "user":
        raise HTTPException(
            status_code=403,
            detail="Audio can only be attached to user messages.",
        )

    if message.get("audio_file_id") or message.get("audio_file_path"):
        raise HTTPException(
            status_code=409,
            detail="Audio is already attached to this message.",
        )

    object_path = (
        f"{AUDIO_PREFIX}/{uid}/"
        f"{uuid.uuid4().hex}{AUDIO_EXTENSIONS[file.content_type]}"
    )

    storage = get_supabase().storage.from_(SUPABASE_BUCKET)

    try:
        storage.upload(
            object_path,
            audio,
            {
                "content-type": file.content_type,
                "upsert": "false",
            },
        )
    except Exception:
        logging.exception("Supabase private audio upload failed")
        raise HTTPException(status_code=502, detail="Audio upload failed.")

    try:
        document.update({
            "audio_file_id": object_path,
            "audio_file_path": object_path,
            "audio_mime_type": file.content_type,
            "audio_size": len(audio),
            "audio_status": "uploaded",
        })
    except Exception:
        logging.exception("Audio uploaded but Firestore metadata save failed")

        try:
            storage.remove([object_path])
        except Exception:
            logging.exception("Could not remove audio after metadata failure")

        raise HTTPException(
            status_code=502,
            detail="Audio uploaded, but its metadata could not be saved.",
        )

    return {
        "message_id": message_id,
        "status": "uploaded",
    }


@router.post("/audio/url")
def get_private_audio_url(
    message_id: str = Form(...),
    authorization: str | None = Header(default=None),
):
    uid = verified_uid(authorization)

    if not message_id or len(message_id) > 150:
        raise HTTPException(status_code=400, detail="Invalid message ID.")

    document = (
        firestore.client()
        .collection("users")
        .document(uid)
        .collection("messages")
        .document(message_id)
    )

    snapshot = document.get()

    if not snapshot.exists:
        raise HTTPException(status_code=404, detail="Message not found.")

    message = snapshot.to_dict() or {}

    if (
        not message.get("audio_file_id")
        or message.get("audio_status") != "uploaded"
    ):
        raise HTTPException(status_code=404, detail="Audio not found.")

    file_path = message.get("audio_file_path")

    if not valid_audio_path(file_path, uid):
        raise HTTPException(status_code=404, detail="Audio not found.")

    if message.get("audio_file_id") != file_path:
        raise HTTPException(status_code=404, detail="Audio not found.")

    supabase_url = os.getenv("SUPABASE_URL")

    if not supabase_url:
        raise HTTPException(
            status_code=503,
            detail="Audio storage unavailable.",
        )

    try:
        storage = get_supabase().storage.from_(SUPABASE_BUCKET)
        result = storage.create_signed_url(file_path, SIGNED_URL_TTL)

        signed_url = (
            result.get("signedURL")
            or result.get("signedUrl")
            or result.get("signed_url")
        )

        if not isinstance(signed_url, str) or not signed_url:
            raise ValueError("Supabase did not return a signed URL.")

        if signed_url.startswith("/"):
            signed_url = supabase_url.rstrip("/") + signed_url

        parsed_url = urlparse(signed_url)
        expected_host = urlparse(supabase_url).netloc

        if (
            parsed_url.scheme != "https"
            or not parsed_url.netloc
            or parsed_url.netloc != expected_host
        ):
            raise ValueError("Supabase returned an invalid signed URL.")

    except HTTPException:
        raise
    except Exception:
        logging.exception("Private audio URL generation failed")
        raise HTTPException(
            status_code=502,
            detail="Could not prepare audio playback.",
        )

    return {
        "message_id": message_id,
        "url": signed_url,
        "mime_type": message.get("audio_mime_type", "audio/mp4"),
        "expires_in": SIGNED_URL_TTL,
    }
