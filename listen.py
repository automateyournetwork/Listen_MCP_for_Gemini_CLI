#!/usr/bin/env python3
"""
Listen MCP — maps your legacy /listen.* TOML to MCP tools
--------------------------------------------------------
Tools provided:
  - listen_start(blocking?: bool=false, queue_dir?: str, language?: str="en",
                 model?: str="base", timeout_ms?: int=900000,
                 delete_after?: bool=false, whisper_bin?: str="whisper",
                 whisper_venv?: str="", poll_ms?: int=500,
                 text?: str | None,
                 http_enable?: bool=True, http_port?: int=8765)
  - listen_status()
  - listen_stop()
  - enqueue_audio(file_path: str) -> { queued_path }
  - listen_logs(lines?: int=50) -> { lines: [..] }
  - listen_clear_logs() -> { status }
  - listen_health() -> { status, running: bool, queue_depth: int }
  - listen_help() -> { text: "..." }

New (for synchronous replies):
  - listen_next(timeout_ms?: int=30000) -> { ok, id, text, source, file } | { ok:false }
  - listen_reply(event_id: str, text: str, audio_path?: str) -> { ok }
HTTP additions:
  - POST /event {"text": "...", "sync": true, "timeout_ms": 30000}  -> waits for reply
  - GET  /wait/{event_id}?timeout_ms=30000                         -> polls for reply
"""

from __future__ import annotations
import os, sys, time, json, shutil, threading, subprocess, logging, contextlib, itertools, base64
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# ---------- logging to stderr + file (NEVER stdout) ----------
LOG_PATH = Path("/tmp/listen_mcp.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

root = logging.getLogger()
for h in list(root.handlers):
    root.removeHandler(h)

fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
sh = logging.StreamHandler(stream=sys.stderr)
sh.setFormatter(fmt)
fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
fh.setFormatter(fmt)
root.addHandler(sh)
root.addHandler(fh)
root.setLevel(logging.INFO)

log = logging.getLogger("listen_mcp")

# ---------- FastMCP wrapper ----------
try:
    from mcp.server.fastmcp import FastMCP
except Exception:
    from fastmcp import FastMCP  # type: ignore

# ---------- Optional HTTP server (to emulate legacy /listen.js) ----------
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
import uvicorn

AUDIO_EXT = {".m4a",".mp3",".wav",".ogg",".flac",".aiff",".wma",".aac"}

class State:
    def __init__(self) -> None:
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.stop_evt = threading.Event()
        self.queue_dir = Path.home() / "listen_queue"
        self.language = "en"
        self.model = "base"
        self.timeout_ms = 15 * 60 * 1000
        self.delete_after = False
        self.poll_ms = 500
        self.whisper_bin = "whisper"
        self.whisper_venv = ""
        self.last_file: Optional[str] = None
        self.last_transcript: Optional[str] = None
        self.processed_count = 0
        self.errors: List[str] = []
        # HTTP server state
        self.http_enable = True
        self.http_port = 8765
        self.http_server: Optional[uvicorn.Server] = None
        self.http_thread: Optional[threading.Thread] = None
        # Event queue (text or transcripts) for CLI/agent consumption
        from collections import deque
        self.events = deque(maxlen=1024)  # each item: {"id": str, "text": str, "source": str, "file": Optional[str]}
        # Reply plumbing for synchronous handoff
        self._id_iter = itertools.count(1)
        self.waiters: Dict[str, Dict[str, Any]] = {}  # id -> {"evt": Event, "reply": Optional[Dict]}
        self.lock = threading.Lock()

    def reset(self) -> None:
        self.running = False
        self.thread = None
        self.stop_evt = threading.Event()
        self.last_file = None
        self.last_transcript = None
        self.processed_count = 0
        self.errors = []
        self.http_server = None
        self.http_thread = None
        with self.lock:
            self.waiters.clear()

STATE = State()
MCP = FastMCP("ListenMCP")

# ---------- helpers ----------

def which(cmd: str, env_path: Optional[str] = None) -> str:
    try:
        env = os.environ.copy()
        if env_path:
            env["PATH"] = env_path
        r = subprocess.run(["which", cmd], capture_output=True, text=True, env=env)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""

def prepare_whisper_environment(whisper_bin: str, whisper_venv: str) -> Tuple[str, str]:
    env_path = os.environ.get("PATH", "")
    if whisper_venv:
        vbin = str(Path(whisper_venv) / "bin")
        if Path(vbin).exists():
            env_path = f"{vbin}:{env_path}"
            log.info("[whisper-env] Added venv bin to PATH: %s", vbin)
        else:
            log.warning("[whisper-env] venv bin not found: %s", vbin)
    found = which(whisper_bin, env_path)
    if not found:
        for p in ["/usr/local/bin/whisper","/opt/homebrew/bin/whisper", str(Path.home()/".local/bin/whisper")]:
            if Path(p).exists():
                found = p
                log.info("[whisper-env] Found whisper at: %s", p)
                break
    log.info("[whisper-env] Final whisper path: %s", found or "NOT FOUND")
    return found, env_path

def transcribe_with_whisper(audio_path: Path, language: str, model: str, timeout_ms: int,
                            whisper_bin: str, whisper_venv: str) -> str:
    cwd = audio_path.parent
    base = audio_path.name
    out_txt = audio_path.with_suffix(".txt")
    wb, env_path = prepare_whisper_environment(whisper_bin, whisper_venv)
    if not wb:
        raise RuntimeError("Whisper CLI not found. Set whisper_bin or whisper_venv.")
    args = [base, "--model", model, "--language", language, "--output_format", "txt"]
    log.info("[whisper] Spawn: %s %s", wb, " ".join(args))
    env = os.environ.copy()
    env["PATH"] = env_path
    p = subprocess.Popen([wb, *args], cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    deadline = time.time() + (timeout_ms/1000.0)
    while True:
        if p.poll() is not None:
            break
        if time.time() > deadline:
            log.warning("[whisper] Timeout after %sms → kill", timeout_ms)
            with contextlib.suppress(Exception):
                p.kill()
            break
        time.sleep(0.05)
    try:
        p.communicate(timeout=2)
    except Exception:
        pass

    code = p.returncode if p.returncode is not None else -1
    log.info("[whisper] Exit code: %s", code)

    if code == 0 and out_txt.exists():
        txt = out_txt.read_text(encoding="utf-8", errors="ignore").strip()
        log.info("[whisper] OK transcript chars=%d", len(txt))
        return txt
    for c in cwd.glob(f"{audio_path.stem}*.txt"):
        with contextlib.suppress(Exception):
            txt = c.read_text(encoding="utf-8", errors="ignore").strip()
            if txt:
                log.info("[whisper] Fallback transcript from %s", c)
                return txt
    raise RuntimeError(f"Transcription failed (code {code}).")

def is_audio(p: Path) -> bool:
    return p.suffix.lower() in AUDIO_EXT

def list_queue_files(qdir: Path) -> List[Path]:
    files = [p for p in qdir.iterdir() if p.is_file() and is_audio(p)]
    files.sort(key=lambda p: p.stat().st_mtime)
    return files

# ---------- reply/event plumbing ----------

def _append_log(line: str):
    try:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        log.exception("append log failed")

def _enqueue_event(text: str, source: str, file: Optional[str]) -> Dict[str, Any]:
    with STATE.lock:
        eid = str(next(STATE._id_iter))
        waiter = {"evt": threading.Event(), "reply": None}
        STATE.waiters[eid] = waiter
    item = {"id": eid, "text": text, "source": source, "file": file}
    STATE.events.append(item)
    return item

def _record_text(text: str):
    STATE.last_file = None
    STATE.last_transcript = text
    STATE.processed_count += 1
    _enqueue_event(text, "http:text", None)
    _append_log(f"[TEXT][{time.strftime('%Y-%m-%d %H:%M:%S')}] {text}")

# ---------- HTTP app ----------

def _build_app() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health_plain():
        return PlainTextResponse("ok")

    @app.get("/health.json")
    async def health_json():
        return {"ok": True, "status": listen_status()}

    @app.post("/event")
    async def event(req: Request):
        try:
            data = await req.json()
        except Exception:
            data = {}

        text = (data or {}).get("text", "")
        attachments = (data or {}).get("attachments", [])
        sync = bool((data or {}).get("sync", False))
        wait_ms = int((data or {}).get("timeout_ms", 30000))

        item = None
        # Accept text immediately (queue it)
        if isinstance(text, str) and text.strip():
            try:
                item = _enqueue_event(text.strip(), "http:text", None)
            except Exception:
                log.exception("/event: failed to enqueue text")

        # Handle attachments -> save into queue_dir (worker will transcribe and enqueue)
        saved: List[Dict[str, Any]] = []
        for att in attachments[:6]:
            fname = att.get("filename") or "upload.bin"
            mime = att.get("mime") or "application/octet-stream"
            b64 = att.get("data_base64") or ""
            try:
                raw = base64.b64decode(b64) if b64 else b""
                dest = STATE.queue_dir / fname
                i = 1
                while dest.exists():
                    stem, ext = dest.stem, dest.suffix
                    dest = STATE.queue_dir / f"{stem}-{i}{ext}"
                    i += 1
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as f:
                    f.write(raw)
                saved.append({"path": str(dest), "bytes": len(raw), "mime": mime})
                log.info("[/event] saved attachment %s (%d bytes, %s)", dest, len(raw), mime)
            except Exception as e:
                log.error("[/event] failed to save attachment %s: %s", fname, e)

        # Non-sync: return immediately with ID if we enqueued text
        if not sync or not item:
            return {"ok": True, "id": item["id"] if item else None, "queued": saved, "status": listen_status()}

        # Sync: wait for the agent to call listen_reply(event_id=item["id"], ...)
        with STATE.lock:
            w = STATE.waiters.get(item["id"])
        if not w:
            return {"ok": False, "error": "no waiter for id", "id": item["id"]}

        if w["evt"].wait(wait_ms/1000.0) and w["reply"]:
            return {"ok": True, "id": item["id"], "queued": saved, "reply": w["reply"]}
        else:
            return {"ok": False, "id": item["id"], "queued": saved, "error": "timeout waiting for reply"}

    @app.get("/wait/{event_id}")
    async def wait_for_reply(event_id: str, timeout_ms: int = 30000):
        with STATE.lock:
            w = STATE.waiters.get(event_id)
        if not w:
            return {"ok": False, "error": "unknown id"}
        if w["evt"].wait(timeout_ms/1000.0) and w["reply"]:
            return {"ok": True, "id": event_id, "reply": w["reply"]}
        return {"ok": False, "id": event_id, "error": "timeout"}

    return app

def _start_http_if_needed():
    if not STATE.http_enable:
        return
    if STATE.http_server is not None:
        return  # already running
    app = _build_app()
    config = uvicorn.Config(app, host="127.0.0.1", port=STATE.http_port, log_level="warning")
    server = uvicorn.Server(config)
    STATE.http_server = server

    def run_server():
        log.info("[http] starting FastAPI on 127.0.0.1:%d", STATE.http_port)
        try:
            server.run()
        except Exception:
            log.exception("[http] server crashed")
        finally:
            log.info("[http] server stopped")

    t = threading.Thread(target=run_server, name="listen-http", daemon=True)
    STATE.http_thread = t
    t.start()

def _stop_http_if_running():
    srv = STATE.http_server
    if srv is not None:
        log.info("[http] stopping FastAPI")
        try:
            srv.should_exit = True
        except Exception:
            pass
        STATE.http_server = None
    th = STATE.http_thread
    if th and th.is_alive():
        th.join(timeout=2)
        STATE.http_thread = None

# ---------- worker ----------

def worker_loop() -> None:
    log.info("[worker] start; queue_dir=%s", STATE.queue_dir)
    while not STATE.stop_evt.is_set():
        try:
            files = list_queue_files(STATE.queue_dir)
            if not files:
                time.sleep(STATE.poll_ms / 1000.0)
                continue
            audio = files[0]
            log.info("[worker] processing: %s", audio)
            try:
                txt = transcribe_with_whisper(
                    audio, STATE.language, STATE.model, STATE.timeout_ms,
                    STATE.whisper_bin, STATE.whisper_venv
                )
                STATE.last_file = str(audio)
                STATE.last_transcript = txt
                STATE.processed_count += 1
                _enqueue_event(txt, "whisper", str(audio))
                _append_log(f"[TRANSCRIPT][{time.strftime('%Y-%m-%d %H:%M:%S')}] {audio.name}: {txt[:240]}{'…' if len(txt)>240 else ''}")
                if STATE.delete_after:
                    with contextlib.suppress(Exception):
                        audio.unlink()
            except Exception as e:
                msg = f"{audio.name}: {e}"
                STATE.errors.append(msg)
                log.error("[worker] %s", msg)
                bad = audio.with_suffix(audio.suffix + ".bad")
                with contextlib.suppress(Exception):
                    shutil.move(str(audio), str(bad))
        except Exception:
            STATE.errors.append("worker loop error")
            log.exception("[worker] loop error")
            time.sleep(0.25)
    log.info("[worker] stop")

# ---------- MCP tools (including event consumption) ----------

@MCP.tool()
def listen_next(timeout_ms: int = 30000) -> Dict[str, Any]:
    """Block (up to timeout_ms) until the next inbound item (text or transcript) arrives, then pop it.
    Returns { ok, id, text, source, file } or { ok: false } on timeout.
    """
    deadline = time.time() + (timeout_ms/1000.0)
    while time.time() < deadline:
        try:
            item = STATE.events.popleft()
            return {"ok": True, **item}  # includes "id"
        except IndexError:
            time.sleep(0.05)
            continue
    return {"ok": False}

@MCP.tool()
def listen_reply(event_id: str, text: str, audio_path: Optional[str] = None) -> Dict[str, Any]:
    """Agent calls this after it generates a reply to a specific event."""
    with STATE.lock:
        w = STATE.waiters.get(event_id)
    if not w:
        return {"ok": False, "error": f"unknown event_id {event_id}"}
    payload = {"text": text}
    if audio_path:
        payload["audio_path"] = audio_path
    w["reply"] = payload
    w["evt"].set()
    return {"ok": True}

# ---------- MCP tools ----------

@MCP.tool()
def listen_start(
    blocking: bool = False,
    queue_dir: Optional[str] = None,
    language: str = "en",
    model: str = "base",
    timeout_ms: int = 15*60*1000,
    delete_after: bool = False,
    whisper_bin: str = "whisper",
    whisper_venv: str = "",
    poll_ms: int = 500,
    text: Optional[str] = None,
    http_enable: bool = True,
    http_port: int = 8765,
) -> Dict[str, Any]:
    """Start background listener and (optionally) the HTTP server for external webhooks."""
    if text is not None and str(text).strip():
        _record_text(str(text))

    STATE.http_enable = bool(http_enable)
    STATE.http_port = int(http_port)
    if STATE.http_enable:
        _start_http_if_needed()

    if STATE.running:
        return listen_status()

    qdir = Path(queue_dir) if queue_dir else STATE.queue_dir
    qdir.mkdir(parents=True, exist_ok=True)

    STATE.queue_dir = qdir
    STATE.language = language
    STATE.model = model
    STATE.timeout_ms = timeout_ms
    STATE.delete_after = delete_after
    STATE.poll_ms = poll_ms
    STATE.whisper_bin = whisper_bin
    STATE.whisper_venv = whisper_venv
    STATE.stop_evt.clear()
    STATE.running = True

    if blocking:
        try:
            files = list_queue_files(qdir)
            for f in files:
                if STATE.stop_evt.is_set():
                    break
                txt = transcribe_with_whisper(f, language, model, timeout_ms, whisper_bin, whisper_venv)
                STATE.last_file = str(f)
                STATE.last_transcript = txt
                STATE.processed_count += 1
                if delete_after:
                    with contextlib.suppress(Exception):
                        f.unlink()
        finally:
            STATE.running = False
            STATE.stop_evt.set()
        return listen_status()

    STATE.thread = threading.Thread(target=worker_loop, name="listen-worker", daemon=True)
    STATE.thread.start()
    return listen_status()

@MCP.tool()
def listen_status() -> Dict[str, Any]:
    qdepth = len(list_queue_files(STATE.queue_dir)) if STATE.queue_dir.exists() else 0
    return {
        "running": STATE.running,
        "queue_dir": str(STATE.queue_dir),
        "queue_depth": qdepth,
        "language": STATE.language,
        "model": STATE.model,
        "timeout_ms": STATE.timeout_ms,
        "delete_after": STATE.delete_after,
        "poll_ms": STATE.poll_ms,
        "whisper_bin": STATE.whisper_bin,
        "whisper_venv": STATE.whisper_venv,
        "last_file": STATE.last_file,
        "last_transcript": (STATE.last_transcript[:400] + "…") if (STATE.last_transcript and len(STATE.last_transcript) > 401) else STATE.last_transcript,
        "processed_count": STATE.processed_count,
        "errors": STATE.errors[-5:],
        "log_path": str(LOG_PATH),
    }

@MCP.tool()
def listen_stop() -> Dict[str, Any]:
    _stop_http_if_running()
    if not STATE.running:
        return listen_status()
    STATE.stop_evt.set()
    t = STATE.thread
    if t and t.is_alive():
        t.join(timeout=3)
    STATE.running = False
    return listen_status()

@MCP.tool()
def enqueue_audio(file_path: str) -> Dict[str, Any]:
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        raise ValueError(f"No such file: {file_path}")
    if not is_audio(p):
        raise ValueError(f"Not an audio file by extension: {p.suffix}")
    STATE.queue_dir.mkdir(parents=True, exist_ok=True)
    dest = STATE.queue_dir / p.name
    i = 1
    while dest.exists():
        dest = STATE.queue_dir / f"{p.stem}-{i}{p.suffix}"
        i += 1
    shutil.copy2(str(p), str(dest))
    log.info("[enqueue] queued %s", dest)
    return {"queued_path": str(dest)}

@MCP.tool()
def listen_logs(lines: int = 50) -> Dict[str, Any]:
    if not LOG_PATH.exists():
        return {"lines": ["No logs found (listener not started yet)."]}
    try:
        with LOG_PATH.open("r", encoding="utf-8", errors="ignore") as f:
            buf = f.readlines()
        tail = buf[-max(1, lines):]
        return {"lines": [s.rstrip("") for s in tail]}
    except Exception as e:
        return {"lines": [f"Error reading logs: {e}"]}

@MCP.tool()
def listen_clear_logs() -> Dict[str, Any]:
    if LOG_PATH.exists():
        try:
            LOG_PATH.write_text("", encoding="utf-8")
            return {"status": "Logs cleared"}
        except Exception as e:
            return {"status": f"Error clearing logs: {e}"}
    return {"status": "No logs found"}

@MCP.tool()
def listen_health() -> Dict[str, Any]:
    qdepth = len(list_queue_files(STATE.queue_dir)) if STATE.queue_dir.exists() else 0
    return {"status": "Healthy", "running": STATE.running, "queue_depth": qdepth}

@MCP.tool()
def listen_help() -> Dict[str, Any]:
    text = (
        "/listen:start   – Start the listener (background)"
        "/listen:status  – Show if it’s running"
        "/listen:stop    – Stop the listener"
        "/listen:logs    – Show the last 50 lines of logs"
        "/listen:clear   – Clear/reset the logs"
        "/listen:health  – Quick MCP health status"
        "/listen:help    – Show this help text"
    )
    return {"text": text}

# ---------- main ----------

if __name__ == "__main__":
    wb, _ = prepare_whisper_environment("whisper", os.environ.get("WHISPER_VENV", ""))
    if wb:
        try:
            r = subprocess.run([wb, "--help"], timeout=5, capture_output=True, text=True)
            log.info("Whisper available: rc=%s", r.returncode)
        except Exception as e:
            log.warning("Whisper check failed: %s", e)
    else:
        log.warning("Whisper not found on PATH; transcription will fail until configured.")

    # MCP stdio server; HTTP starts on demand via listen_start(http_enable=True).
    MCP.run()
