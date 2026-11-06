import base64
import json
import logging
import os
import uuid
import urllib.error
import urllib.request

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from firebase_admin import auth, firestore

router = APIRouter()

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
MAX_AUDIO_SIZE = 25 * 1024 * 1024


def verified_uid(authorization: str | None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required.")

    try:
        token = authorization.removeprefix("Bearer ").strip()
        return auth.verify_id_token(token, check_revoked=True)["uid"]
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid authentication token.")


@router.post("/audio/upload")
def upload_private_audio(
    message_id: str = Form(...),
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
):
    uid = verified_uid(authorization)

    if not message_id or len(message_id) > 150:
        raise HTTPException(status_code=400, detail="Invalid message ID.")

    messages = (
        firestore.client()
        .collection("users").document(uid)
        .collection("messages")
    )
    document = messages.document(message_id)

    # Only attach audio to an existing user message owned by this UID.
    snapshot = document.get()
    if not snapshot.exists:
        raise HTTPException(status_code=404, detail="Message not found.")

    message = snapshot.to_dict() or {}
    if message.get("role") != "user":
        raise HTTPException(status_code=403, detail="Audio can only be attached to user messages.")

    if message.get("audio_file_id"):
        raise HTTPException(status_code=409, detail="Audio is already attached to this message.")

    if file.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported audio format.")

    private_key = os.getenv("IMAGEKIT_PRIVATE_KEY")
    if not private_key:
        raise HTTPException(status_code=503, detail="Audio storage unavailable.")

    audio = file.file.read(MAX_AUDIO_SIZE + 1)
    if not audio:
        raise HTTPException(status_code=400, detail="Empty audio file.")
    if len(audio) > MAX_AUDIO_SIZE:
        raise HTTPException(status_code=413, detail="Audio file too large.")

    extensions = {
        "audio/mp4": ".m4a",
        "audio/m4a": ".m4a",
        "audio/x-m4a": ".m4a",
        "audio/mpeg": ".mp3",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/webm": ".webm",
        "audio/ogg": ".ogg",
    }
    filename = f"emily_voice_{uuid.uuid4().hex}{extensions[file.content_type]}"
    boundary = f"EmilyAudio{uuid.uuid4().hex}"

    def field(name: str, value: str) -> bytes:
        return (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n"
        ).encode("utf-8")

    body = b"".join([
        f"--{boundary}\r\n".encode(),
        (
            f'Content-Disposition: form-data; name="file"; '
            f'filename="{filename}"\r\n'
            f"Content-Type: {file.content_type}\r\n\r\n"
        ).encode(),
        audio,
        b"\r\n",
        field("fileName", filename),
        field("folder", "/emily_private_audio"),
        field("isPrivateFile", "true"),
        field("useUniqueFileName", "true"),
        f"--{boundary}--\r\n".encode(),
    ])

    credentials = base64.b64encode(f"{private_key}:".encode()).decode()
    request = urllib.request.Request(
        "https://upload.imagekit.io/api/v1/files/upload",
        data=body,
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPException:
        raise
    except Exception:
        logging.exception("Private audio upload failed")
        raise HTTPException(status_code=502, detail="Audio upload failed.")

    if not isinstance(result, dict):
        logging.error("ImageKit upload returned an unexpected response format")
        raise HTTPException(status_code=502, detail="Invalid audio storage response.")

    file_id = result.get("fileId")
    if not isinstance(file_id, str) or not file_id:
        raise HTTPException(status_code=502, detail="Missing audio file ID.")

    file_path = result.get("filePath")
    if (
        not isinstance(file_path, str)
        or not file_path.startswith("/emily_private_audio/")
        or ".." in file_path.split("/")
    ):
        raise HTTPException(status_code=502, detail="Invalid private audio file path.")

    if result.get("isPrivateFile") is not True:
        raise HTTPException(status_code=502, detail="Audio was not marked private.")

    try:
        document.update({
            "audio_file_id": file_id,
            "audio_file_path": file_path,
            "audio_mime_type": file.content_type,
            "audio_size": len(audio),
            "audio_status": "uploaded",
        })
    except Exception:
        logging.exception("Audio uploaded but Firestore metadata save failed")
        raise HTTPException(
            status_code=502,
            detail="Audio uploaded, but its metadata could not be saved.",
        )

    return {"message_id": message_id, "status": "uploaded"}

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
        .collection("users").document(uid)
        .collection("messages").document(message_id)
    )
    snapshot = document.get()

    if not snapshot.exists:
        raise HTTPException(status_code=404, detail="Message not found.")

    message = snapshot.to_dict() or {}

    if not message.get("audio_file_id") or message.get("audio_status") != "uploaded":
        raise HTTPException(status_code=404, detail="Audio not found.")

    file_path = message.get("audio_file_path")
    if (
        not isinstance(file_path, str)
        or not file_path.startswith("/emily_private_audio/")
        or ".." in file_path.split("/")
    ):
        raise HTTPException(status_code=404, detail="Audio not found.")

    private_key = os.getenv("IMAGEKIT_PRIVATE_KEY")
    url_endpoint = os.getenv("IMAGEKIT_URL_ENDPOINT")

    if not private_key or not url_endpoint:
        raise HTTPException(status_code=503, detail="Audio storage unavailable.")

    public_key = os.getenv("IMAGEKIT_PUBLIC_KEY")
    if not public_key:
        raise HTTPException(status_code=503, detail="Audio storage unavailable.")

    try:
        from imagekitio import ImageKit

        imagekit = ImageKit(
            public_key=public_key,
            private_key=private_key,
            url_endpoint=url_endpoint,
        )

        signed_url = imagekit.url({
            "path": file_path,
            "url_endpoint": url_endpoint,
            "signed": True,
            "expire_seconds": 300,
            "private_key": private_key,
        })
    except HTTPException:
        raise
    except Exception:
        logging.exception("Private audio URL generation failed")
        raise HTTPException(status_code=502, detail="Could not prepare audio playback.")

    if not signed_url.startswith(url_endpoint.rstrip("/") + "/"):
        raise HTTPException(status_code=502, detail="Invalid audio URL.")

    return {
        "message_id": message_id,
        "url": signed_url,
        "mime_type": message.get("audio_mime_type", "audio/mp4"),
        "expires_in": 300,
    }