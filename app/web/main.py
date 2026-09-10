"""FastAPI application: auth, upload, job polling, download."""
from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.core.pipeline import TranslationOptions
from app.core.translate import provider_name

from .auth import SESSION_COOKIE, create_session, current_user, require_user, verify_user
from .jobs import store

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("doc-translator")

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")
TEMPLATE_DIR = os.path.join(HERE, "templates")

ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "50")) * 1024 * 1024

app = FastAPI(title="Document Translator", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _page(name: str) -> str:
    with open(os.path.join(TEMPLATE_DIR, name), encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if current_user(request):
        return RedirectResponse("/", status_code=303)
    return HTMLResponse(_page("login.html"))


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    # Every protected route bounces an unauthenticated visitor to the login page.
    if not current_user(request):
        return RedirectResponse("/login", status_code=303)
    return HTMLResponse(_page("app.html"))


# ---------------------------------------------------------------------------
# Auth API
# ---------------------------------------------------------------------------
@app.post("/api/login")
def api_login(username: str = Form(...), password: str = Form(...)):
    if not verify_user(username, password):
        log.warning("Failed login attempt for username=%r", username)
        return JSONResponse(
            {"error": "Incorrect username or password."}, status_code=401
        )
    token = create_session(username)
    response = JSONResponse({"ok": True, "username": username})
    response.set_cookie(
        SESSION_COOKIE,
        token,
        httponly=True,       # not readable from JavaScript
        samesite="lax",
        secure=os.environ.get("COOKIE_SECURE", "").lower() == "true",
        max_age=60 * 60 * 8,
        path="/",
    )
    return response


@app.post("/api/logout")
def api_logout():
    response = JSONResponse({"ok": True})
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


@app.get("/api/me")
def api_me(user: str = Depends(require_user)):
    return {"username": user, "provider": provider_name()}


# ---------------------------------------------------------------------------
# Translation API
# ---------------------------------------------------------------------------
@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    direction: str = Form("en2ar"),
    mirror: str = Form("true"),
    underline: str = Form("true"),
    flip_images: str = Form("true"),
    html_engine: str = Form("true"),
    # Defaults to empty rather than "false": unset means "whatever the
    # environment says", which is how a deployment turns this on globally.
    vision_layout: str = Form(""),
    user: str = Depends(require_user),
):
    filename = os.path.basename(file.filename or "document")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"'{ext or 'this file type'}' is not supported. "
                   f"Upload a PDF or DOCX file.",
        )
    if direction not in ("en2ar", "ar2en"):
        raise HTTPException(status_code=400, detail="Unknown translation direction.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"That file is larger than the "
                   f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )

    def flag(value: str) -> bool:
        return str(value).lower() in ("1", "true", "yes", "on")

    options = TranslationOptions(
        direction=direction,
        mirror=flag(mirror),
        underline=flag(underline),
        flip_directional_images=flag(flip_images),
        html_engine=flag(html_engine),
        # Omitted rather than passed as False, so an unset form field leaves
        # the environment's default in force instead of overriding it off.
        **({"vision_layout": True} if flag(vision_layout) else {}),
    )
    job = store.create(user, filename, data, options)
    log.info("Job %s queued by %s (%s, %s)", job.id, user, filename, direction)
    return JSONResponse(job.public(), status_code=202)


@app.get("/api/jobs")
def list_jobs(user: str = Depends(require_user)):
    return {"jobs": [j.public() for j in store.list_for(user)]}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, user: str = Depends(require_user)):
    job = store.get(job_id, owner=user)
    if job is None:
        raise HTTPException(status_code=404, detail="That job could not be found.")
    return job.public()


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str, user: str = Depends(require_user)):
    job = store.get(job_id, owner=user)
    if job is None:
        raise HTTPException(status_code=404, detail="That job could not be found.")
    if job.status != "done":
        raise HTTPException(
            status_code=409, detail="This document is not finished yet."
        )
    if not os.path.exists(job.output_path):
        log.error("Job %s marked done but %s is missing", job_id, job.output_path)
        raise HTTPException(
            status_code=410,
            detail="The translated file is no longer available. Please run it again.",
        )
    media = (
        "application/pdf" if job.output_path.endswith(".pdf")
        else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    return FileResponse(
        job.output_path,
        media_type=media,
        filename=os.path.basename(job.output_path),
    )


@app.get("/api/jobs/{job_id}/qa")
def job_qa(job_id: str, user: str = Depends(require_user)):
    job = store.get(job_id, owner=user)
    if job is None:
        raise HTTPException(status_code=404, detail="That job could not be found.")
    return job.qa


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(os.path.join(STATIC_DIR, "favicon.svg"),
                        media_type="image/svg+xml")


@app.get("/api/health")
def health():
    return {"status": "ok", "provider": provider_name()}


# ---------------------------------------------------------------------------
# Errors: always JSON with a readable message, never a stack trace
# ---------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    log.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        {"error": "Something went wrong on the server. Please try again."},
        status_code=500,
    )


@app.on_event("shutdown")
def _shutdown():
    store.shutdown()
