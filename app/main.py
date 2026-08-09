"""AutoAce dashboard.

FastAPI + Jinja2 + a little vanilla JS. Upload, progress, a results table and
two download buttons -- deliberately not a React app (BUILD_SPEC.md 11.1).
"""
from __future__ import annotations

import logging
import mimetypes
from contextlib import asynccontextmanager
import os
import threading
from pathlib import Path

from fastapi import (Depends, FastAPI, File, Form, HTTPException, Request,
                     UploadFile)
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               RedirectResponse, Response)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from voicetone import __version__

from . import auth
from .auth import require_user
from .jobs import (PLAYBACK_ENABLED, RETENTION, WORKERS, BadArchive, JobStore,
                   UploadTooLarge, report_text, results_csv, results_json,
                   stage_upload)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(levelname)s %(name)s %(message)s")
log = logging.getLogger("autoace.app")

BASE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE / "templates"))
store = JobStore()

# How many batches the home page previews before deferring to /batches.
HOME_BATCHES = 3

# Set COOKIE_SECURE=1 behind TLS. Off by default so http://localhost works.
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "").lower() in ("1", "true", "yes")

def _warm_models() -> None:
    """Load every checkpoint into memory before a user asks for one.

    The models are lru_cached and were loaded lazily, on the first file that
    needed them. That put ~2 GB of disk reads and torch initialisation inside
    the first upload of a container's life: measured on Cloud Run, the same
    31-second call took **183 s cold and 19.6 s warm**. Every deploy creates a
    fresh revision, so whoever opened the link first always paid it, and
    concluded the system was slow.

    Runs on a background thread so the port binds immediately -- a platform
    that health-checks the port before routing traffic must not wait for this.
    """
    import time
    t = time.perf_counter()
    try:
        from voicetone.stack import build_predictors
        build_predictors()
        log.info("model warm-up finished in %.1fs", time.perf_counter() - t)
    except Exception as exc:                       # noqa: BLE001
        # Never fatal: a missing checkpoint should degrade the stack, not stop
        # the dashboard from serving.
        log.warning("model warm-up failed (%s); models will load on demand", exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_warm_models, name="warm-models", daemon=True).start()
    yield
    # Never leave customer audio on disk after the process exits.
    store.shutdown()


app = FastAPI(title="AutoAce Voice Tone", version=__version__,
              docs_url=None, redoc_url=None, lifespan=lifespan)

# Fonts are served from here rather than fetched from a CDN: the container has
# no outbound network access, and a webfont that silently fails to load would
# change the whole typographic scale. Quicksand is OFL-1.1, so bundling it is
# permitted -- the licence text ships alongside it (see LICENCES.md).
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")


def _ctx(request: Request, **extra) -> dict:
    """Values every template needs."""
    from voicetone.schema import QUALITY_ORDER, SEVERITY_ORDER, TONE_ORDER
    from voicetone.stack import stack_names
    return {"retention": RETENTION, "version": __version__,
            "insecure": auth.using_default_password(), "workers": WORKERS,
            "playback": PLAYBACK_ENABLED, "stack": stack_names(),
            # The results table sorts ordinal columns by rank, not by the
            # displayed word. The rankings come from the schema so the two
            # cannot drift.
            "tone_order": TONE_ORDER, "severity_order": SEVERITY_ORDER,
            "quality_order": QUALITY_ORDER,
            **extra}


@app.exception_handler(HTTPException)
async def _unauthorised(request: Request, exc: HTTPException):
    """A signed-out browser gets the login page; a signed-out fetch() gets JSON.

    Without this split, an expired session would dump raw JSON into the address
    bar on a normal navigation, and the page scripts would try to parse an HTML
    login page as a status response.
    """
    if exc.status_code == 401:
        wants_html = "text/html" in (request.headers.get("accept") or "")
        if wants_html:
            nxt = request.url.path
            return RedirectResponse(f"/login?next={nxt}", status_code=303)
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


# --------------------------------------------------------------------------
# auth
# --------------------------------------------------------------------------

@app.get("/healthz")
@app.get("/health")
def healthz() -> dict:
    """Unauthenticated so a load balancer can reach it.

    Served at two paths because Google's frontend intercepts `/healthz` before
    it reaches the container on Cloud Run: the request never arrives and the
    caller gets Google's own 404, with no `server: Google Frontend` header to
    explain where it came from. `/health` reaches the app normally. `/healthz`
    is kept for the in-container Docker HEALTHCHECK, which talks to localhost
    and is unaffected.
    """
    return {"status": "ok", "version": __version__, "workers": WORKERS}


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, next: str = "/"):
    if auth.current_user(request):
        return RedirectResponse(next or "/", status_code=303)
    return templates.TemplateResponse(request, "login.html",
                                      _ctx(request, user=None, next_url=next))


@app.post("/login")
def login(request: Request, username: str = Form(""), password: str = Form(""),
          next: str = Form("/")):
    if not auth.check_password(username, password):
        # Deliberately vague: saying which field was wrong helps an attacker
        # enumerate usernames and helps a legitimate user not at all.
        return templates.TemplateResponse(
            request, "login.html",
            _ctx(request, user=None, next_url=next,
                 error="Incorrect username or password."),
            status_code=401)
    # Only ever redirect within this app -- an absolute or protocol-relative
    # "next" would turn the login form into an open redirect.
    target = next if next.startswith("/") and not next.startswith("//") else "/"
    resp = RedirectResponse(target, status_code=303)
    auth.set_cookie(resp, username, secure=COOKIE_SECURE)
    return resp


@app.get("/logout")
def logout():
    resp = RedirectResponse("/login", status_code=303)
    auth.clear_cookie(resp)
    return resp


# --------------------------------------------------------------------------
# pages
# --------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index(request: Request, user: str = Depends(require_user)):
    return templates.TemplateResponse(request, "index.html",
                                      _ctx(request, user=user,
                                           recent=store.recent(limit=HOME_BATCHES),
                                           total_batches=len(store.recent(limit=10_000))))


@app.get("/history", response_class=HTMLResponse)
def history(request: Request, user: str = Depends(require_user)):
    """The full list. The home page shows only the newest few -- an unbounded
    list there pushes everything else off the page as usage builds up."""
    return templates.TemplateResponse(request, "history.html", _ctx(
        request, user=user, recent=store.recent(limit=store.max_jobs),
        limit=store.max_jobs))


@app.post("/upload")
async def upload(request: Request,
                 files: list[UploadFile] = File(...),
                 user: str = Depends(require_user)):
    # Read with a running total so an oversized upload is refused before it is
    # all in memory, rather than after (BUILD_SPEC 11.3).
    from .jobs import MAX_UPLOAD_BYTES
    blobs: list[tuple[str, bytes]] = []
    total = 0
    for f in files:
        chunks: list[bytes] = []
        while True:
            chunk = await f.read(1 << 20)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                return JSONResponse(status_code=413, content={
                    "error": f"Upload exceeds the "
                             f"{MAX_UPLOAD_BYTES // 2**20} MB limit."})
            chunks.append(chunk)
        blobs.append((f.filename or "upload.bin", b"".join(chunks)))
        await f.close()

    if not blobs:
        return JSONResponse(status_code=400, content={"error": "No files received."})

    try:
        staged = stage_upload(blobs)
    except (UploadTooLarge, BadArchive) as exc:
        return JSONResponse(status_code=413, content={"error": str(exc)})
    except Exception as exc:                       # noqa: BLE001
        log.exception("staging failed")
        return JSONResponse(status_code=400,
                            content={"error": f"Could not read the upload: {exc}"})

    if not staged.audio:
        import shutil
        shutil.rmtree(staged.root, ignore_errors=True)
        return JSONResponse(status_code=400, content={
            "error": "No supported audio found in the upload "
                     "(nested folders were searched too)."})

    job = store.submit(staged)
    return {"job": job.id, "total": len(staged.audio),
            "labelled": len(staged.labels),
            "unmatched_labels": staged.unmatched,
            "unlabelled_audio": staged.unlabelled,
            "skipped": staged.skipped,
            "warnings": staged.warnings}


@app.get("/jobs/{job_id}/status")
def job_status(job_id: str, user: str = Depends(require_user)):
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return job.public()


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_page(request: Request, job_id: str, user: str = Depends(require_user)):
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return templates.TemplateResponse(request, "job.html", _ctx(
        request, user=user, job=job, report=report_text(job),
        report_data=job.report))


@app.get("/jobs/{job_id}/audio")
def job_audio(job_id: str, name: str, user: str = Depends(require_user)):
    """Stream one clip back so a prediction can be checked by ear.

    `name` is a row's display name, looked up in a map the server built. It is
    never joined onto a path, so no amount of `../` in the query string reaches
    anything we did not put there ourselves.
    """
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    if job.audio_expired:
        raise HTTPException(410, "audio retention window has closed")
    path = job.audio_paths.get(name)
    if not path or not Path(path).is_file():
        raise HTTPException(404, "no audio retained for that file")

    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    # inline, not attachment: this feeds an <audio> element, it is not a download.
    return FileResponse(path, media_type=mime,
                        headers={"Content-Disposition": "inline",
                                 "Cache-Control": "private, max-age=300"})


@app.get("/jobs/{job_id}/download.json")
def download_json(job_id: str, user: str = Depends(require_user)):
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return Response(results_json(job), media_type="application/json",
                    headers={"Content-Disposition":
                             f'attachment; filename="autoace_{job_id}.json"'})


@app.get("/jobs/{job_id}/download.csv")
def download_csv(job_id: str, user: str = Depends(require_user)):
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return Response(results_csv(job), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition":
                             f'attachment; filename="autoace_{job_id}.csv"'})
