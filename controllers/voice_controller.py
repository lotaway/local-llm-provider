from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import JSONResponse, StreamingResponse
import asyncio
import os
import uuid
import tempfile
import shutil
import sys
import json
from typing import Optional
from pydantic import BaseModel
from schemas import ChatRequest, Message
from controllers.llm_controller import chat_completions as llm_chat_completions
from constants import VIBEVOICE_DIR, VIBEVOICE_MODEL, VIBEVOICE_SCRIPT

router = APIRouter(prefix="/voice", tags=["voice"])


class InputError(Exception):
    pass


class ConfigError(Exception):
    pass


class TranscribeError(Exception):
    pass


class VoiceASRConfig(BaseModel):
    dir: str
    model: str
    script: str


class SessionStore:
    def __init__(self):
        self._store = {}

    def ensure(self, sid: str):
        if sid in self._store:
            return
        tmp = tempfile.mkdtemp(prefix="voice_")
        path = os.path.join(tmp, f"{sid}.raw")
        q = asyncio.Queue()
        self._store[sid] = {"path": path, "queue": q, "tmp": tmp}

    def append(self, sid: str, file: UploadFile) -> str:
        self.ensure(sid)
        p = self._store[sid]["path"]
        return append_file(p, file)

    def queue(self, sid: str):
        self.ensure(sid)
        return self._store[sid]["queue"]

    async def close(self, sid: str):
        d = self._store.get(sid)
        if not d:
            return
        q = d.get("queue")
        if q:
            await q.put(None)
        tmp = d.get("tmp")
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)
        del self._store[sid]


SESSIONS = SessionStore()


def load_config() -> VoiceASRConfig:
    if not VIBEVOICE_SCRIPT:
        raise ConfigError("VibeVoice script missing")
    return VoiceASRConfig(
        dir=VIBEVOICE_DIR, model=VIBEVOICE_MODEL, script=VIBEVOICE_SCRIPT
    )


async def save_file(file: UploadFile) -> str:
    fd, path = tempfile.mkstemp(suffix=f"_{uuid.uuid4().hex}_{file.filename}")
    os.close(fd)
    with open(path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return path


def append_file(path: str, file: UploadFile) -> str:
    with open(path, "ab") as f:
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return path


async def transcribe(path: str, cfg: VoiceASRConfig) -> dict:
    if not os.path.exists(cfg.script):
        raise ConfigError("VibeVoice not configured")
    cmd = [sys.executable, cfg.script, "--model_path", cfg.model, "--audio_files", path]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    out, err = await proc.communicate()
    if proc.returncode != 0:
        raise TranscribeError(err.decode("utf-8", errors="ignore"))
    text = out.decode("utf-8", errors="ignore")
    try:
        lines = [l for l in text.splitlines() if l.strip()]
        parsed = json.loads(lines[-1])
        return parsed
    except Exception:
        return {"text": text}


def stable_payload(payload: dict, sid: Optional[str]) -> dict:
    t = ""
    segs = []
    if "segments" in payload and isinstance(payload["segments"], list):
        segs = payload["segments"]
        t = " ".join([s.get("text", "") for s in segs]).strip()
    if not t:
        t = payload.get("text", "")
    return {"text": t, "segments": segs, "session_id": sid}


async def start_stream(sid: str, request: Request):
    SESSIONS.ensure(sid)
    q = SESSIONS.queue(sid)

    async def gen():
        while True:
            if await request.is_disconnected():
                break
            try:
                item = await asyncio.wait_for(q.get(), timeout=0.5)
                if item is None:
                    break
                yield f"data: {json.dumps(item)}\n\n"
            except asyncio.TimeoutError:
                continue
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/to/text")
async def to_text(
    request: Request,
    audio: Optional[UploadFile] = File(None),
    chunk: Optional[UploadFile] = File(None),
    session_id: Optional[str] = Form(None),
    stream: bool = Form(False),
):
    try:
        cfg = load_config()
        if stream and session_id:
            return await start_stream(session_id, request)
        if audio:
            p = await save_file(audio)
            try:
                raw = await transcribe(p, cfg)
                return JSONResponse(stable_payload(raw, None))
            finally:
                if os.path.exists(p):
                    os.remove(p)
        if chunk and session_id:
            p = SESSIONS.append(session_id, chunk)
            raw = await transcribe(p, cfg)
            data = stable_payload(raw, session_id)
            await SESSIONS.queue(session_id).put(data)
            return JSONResponse(data)
        raise InputError("Invalid parameters")
    except ConfigError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except TranscribeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except InputError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/chat/completions")
async def chat_from_voice(
    request: Request,
    audio: UploadFile = File(...),
    model: Optional[str] = Form(None),
    stream: bool = Form(False),
):
    try:
        cfg = load_config()
        p = await save_file(audio)
        try:
            raw = await transcribe(p, cfg)
            data = stable_payload(raw, None)
            text = data.get("text", "")
            if not text:
                raise TranscribeError("Empty transcription")
            msgs = [Message(role="user", content=text)]
            if not model:
                from model_providers import LocalLLModel

                li = LocalLLModel.get_local_models()
                if not li:
                    raise ConfigError("No local models available")
                model = li[0]
            req = ChatRequest(model=model, messages=msgs, stream=stream)
            return await llm_chat_completions(req, request)
        finally:
            if os.path.exists(p):
                os.remove(p)
    except ConfigError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except TranscribeError as e:
        raise HTTPException(status_code=500, detail=str(e))
