"""Background job registry for translation runs.

Jobs run in a thread pool so uploads return immediately - a large PDF takes
minutes and must not block the request. State is in memory, which is right for
a single-process deployment; swapping in Redis means reimplementing JobStore
alone.
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

from app.core.pipeline import TranslationOptions, run_pipeline
from app.core.qa import QAReport

log = logging.getLogger(__name__)

STORAGE_DIR = os.environ.get(
    "STORAGE_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "storage"),
)
MAX_WORKERS = int(os.environ.get("JOB_WORKERS", "2"))
JOB_TTL_SECONDS = int(os.environ.get("JOB_TTL", str(60 * 60 * 6)))

STATUS_QUEUED = "queued"
STATUS_EXTRACTING = "extracting"
STATUS_TRANSLATING = "translating"
STATUS_REBUILDING = "rebuilding"
STATUS_DONE = "done"
STATUS_FAILED = "failed"


@dataclass
class Job:
    id: str
    owner: str
    filename: str
    input_path: str
    output_path: str
    options: TranslationOptions
    status: str = STATUS_QUEUED
    progress: int = 0
    message: str = "Waiting to start…"
    error: Optional[str] = None
    qa: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    def public(self) -> dict[str, Any]:
        """Everything the frontend needs - and nothing about the filesystem."""
        return {
            "id": self.id,
            "filename": self.filename,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "direction": self.options.direction,
            "mirror": self.options.mirror,
            "qa": self.qa,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "download_ready": self.status == STATUS_DONE,
        }


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pool: Optional[ThreadPoolExecutor] = None
        self._pool_lock = threading.Lock()
        os.makedirs(STORAGE_DIR, exist_ok=True)

    def _submit(self, job_id: str) -> None:
        """Get a live pool, recreating it if a previous shutdown closed it.

        The store outlives any single application instance (it is a module-level
        singleton), so a shutdown must not leave it permanently unusable.
        """
        with self._pool_lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(
                    max_workers=MAX_WORKERS, thread_name_prefix="translate"
                )
            try:
                self._pool.submit(self._run, job_id)
                return
            except RuntimeError:
                self._pool = ThreadPoolExecutor(
                    max_workers=MAX_WORKERS, thread_name_prefix="translate"
                )
                self._pool.submit(self._run, job_id)

    # -- lifecycle ---------------------------------------------------------
    def create(self, owner: str, filename: str, data: bytes,
               options: TranslationOptions) -> Job:
        job_id = uuid.uuid4().hex
        job_dir = os.path.join(STORAGE_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)

        ext = os.path.splitext(filename)[1].lower()
        input_path = os.path.join(job_dir, f"input{ext}")
        stem = os.path.splitext(os.path.basename(filename))[0]
        suffix = "ar" if options.direction == "en2ar" else "en"
        output_path = os.path.join(job_dir, f"{stem}_{suffix}{ext}")

        with open(input_path, "wb") as fh:
            fh.write(data)

        job = Job(
            id=job_id,
            owner=owner,
            filename=filename,
            input_path=input_path,
            output_path=output_path,
            options=options,
        )
        with self._lock:
            self._jobs[job_id] = job
        self._submit(job_id)
        return job

    def get(self, job_id: str, owner: Optional[str] = None) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return None
        if owner is not None and job.owner != owner:
            return None  # treated as not-found: never leak another user's jobs
        return job

    def list_for(self, owner: str) -> list[Job]:
        with self._lock:
            jobs = [j for j in self._jobs.values() if j.owner == owner]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    # -- execution ---------------------------------------------------------
    def _update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in fields.items():
                setattr(job, key, value)

    def _run(self, job_id: str) -> None:
        job = self.get(job_id)
        if job is None:
            return

        def progress(stage: str, percent: int, message: str) -> None:
            self._update(job_id, status=stage, progress=percent, message=message)

        qa = QAReport()
        try:
            run_pipeline(job.input_path, job.output_path, job.options, qa, progress)
            qa.save(os.path.join(os.path.dirname(job.output_path), "qa_report.json"))
            self._update(
                job_id,
                status=STATUS_DONE,
                progress=100,
                message="Done",
                qa=qa.to_dict(),
                finished_at=time.time(),
            )
        except Exception as exc:
            # Full detail to the server log; a readable sentence to the user.
            log.error("Job %s failed:\n%s", job_id, traceback.format_exc())
            qa.add("pipeline", "error", str(exc))
            self._update(
                job_id,
                status=STATUS_FAILED,
                progress=100,
                message="Translation failed",
                error=_friendly_error(exc),
                qa=qa.to_dict(),
                finished_at=time.time(),
            )

    # -- housekeeping ------------------------------------------------------
    def cleanup(self) -> int:
        """Drop jobs and their files once they age out."""
        cutoff = time.time() - JOB_TTL_SECONDS
        removed = 0
        with self._lock:
            stale = [
                j for j in self._jobs.values()
                if j.finished_at is not None and j.finished_at < cutoff
            ]
            for job in stale:
                self._jobs.pop(job.id, None)
        for job in stale:
            shutil.rmtree(os.path.join(STORAGE_DIR, job.id), ignore_errors=True)
            removed += 1
        return removed

    def shutdown(self) -> None:
        with self._pool_lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)


def _friendly_error(exc: Exception) -> str:
    """Translate an exception into something a non-technical user can act on.

    Stack traces stay in the log; the UI never shows one.
    """
    text = str(exc)
    lowered = text.lower()
    if isinstance(exc, ValueError) and "unsupported file type" in lowered:
        return text
    if "password protected" in lowered or "encrypted" in lowered:
        return "This file is password protected. Remove the protection and try again."
    if "cannot open" in lowered or "no objects found" in lowered or "damaged" in lowered:
        return "This file appears to be corrupted or is not a valid document."
    if "no usable font" in lowered:
        return ("No Arabic font is installed on the server. Add "
                "NotoNaskhArabic-Regular.ttf to the fonts folder.")
    if "api" in lowered and ("key" in lowered or "auth" in lowered):
        return "The translation service rejected the request. Check the API key."
    if "timeout" in lowered or "timed out" in lowered:
        return "The translation service timed out. Try again in a moment."
    # Nothing matched. The stack trace stays in the log, but the exception's
    # type goes to the user: "Something went wrong" is the same sentence for
    # every fault, so a report of one carries no information back to whoever
    # has to find it. The type names the fault without exposing a path or a
    # line of source.
    return (f"Something went wrong while translating this document "
            f"({type(exc).__name__}). The details have been logged for the "
            f"administrator.")


store = JobStore()
