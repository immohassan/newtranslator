"""Web layer: auth gating, upload validation, job lifecycle, download."""
import os
import time

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("APP_USERNAME", "admin")
os.environ.setdefault("APP_PASSWORD", "testpass123")

from app.web.auth import create_session, read_session, verify_user  # noqa: E402
from app.web.main import app  # noqa: E402


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_client(client):
    response = client.post("/api/login",
                           data={"username": "admin", "password": "testpass123"})
    assert response.status_code == 200
    return client


def wait_for(client, job_id, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.25)
    raise AssertionError("job did not finish in time")


# ---------------------------------------------------------------- auth
def test_password_is_hashed_not_stored_plaintext():
    from app.web.auth import USERS

    stored = USERS["admin"]
    assert stored.startswith("$2"), "password must be bcrypt hashed"
    assert "testpass123" not in stored


def test_verify_user():
    assert verify_user("admin", "testpass123")
    assert not verify_user("admin", "wrong")
    assert not verify_user("ghost", "testpass123")


def test_session_token_round_trip():
    token = create_session("admin")
    assert read_session(token) == "admin"


def test_tampered_session_rejected():
    token = create_session("admin")
    assert read_session(token[:-4] + "aaaa") is None
    assert read_session("garbage") is None


def test_root_redirects_when_signed_out(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_login_page_is_public(client):
    assert client.get("/login").status_code == 200


def test_protected_endpoints_require_auth(client):
    for path in ("/api/me", "/api/jobs"):
        assert client.get(path).status_code == 401


def test_bad_login_returns_message(client):
    response = client.post("/api/login",
                           data={"username": "admin", "password": "nope"})
    assert response.status_code == 401
    assert "Incorrect username or password" in response.json()["error"]


def test_login_sets_httponly_cookie(client):
    response = client.post("/api/login",
                           data={"username": "admin", "password": "testpass123"})
    assert response.status_code == 200
    assert "httponly" in response.headers["set-cookie"].lower()


def test_logout_clears_session(auth_client):
    assert auth_client.get("/api/me").status_code == 200
    auth_client.post("/api/logout")
    assert auth_client.get("/api/me").status_code == 401


# ---------------------------------------------------------------- uploads
def test_rejects_unsupported_file_type(auth_client):
    response = auth_client.post(
        "/api/jobs",
        files={"file": ("notes.txt", b"hello", "text/plain")},
        data={"direction": "en2ar"},
    )
    assert response.status_code == 400
    assert "PDF or DOCX" in response.json()["error"]


def test_rejects_empty_file(auth_client):
    response = auth_client.post(
        "/api/jobs",
        files={"file": ("empty.pdf", b"", "application/pdf")},
        data={"direction": "en2ar"},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["error"].lower()


def test_rejects_unknown_direction(auth_client, sample_pdf):
    with open(sample_pdf, "rb") as fh:
        response = auth_client.post(
            "/api/jobs",
            files={"file": ("sample.pdf", fh.read(), "application/pdf")},
            data={"direction": "fr2de"},
        )
    assert response.status_code == 400


def test_corrupt_pdf_fails_with_readable_message(auth_client):
    response = auth_client.post(
        "/api/jobs",
        files={"file": ("broken.pdf", b"%PDF-1.4 not really a pdf", "application/pdf")},
        data={"direction": "en2ar"},
    )
    assert response.status_code == 202
    job = wait_for(auth_client, response.json()["id"])
    assert job["status"] == "failed"
    assert job["error"], "a failed job must carry a human-readable message"
    assert "Traceback" not in job["error"], "never leak a stack trace to the UI"


# ---------------------------------------------------------------- full flow
def test_full_flow_pdf(auth_client, sample_pdf):
    with open(sample_pdf, "rb") as fh:
        response = auth_client.post(
            "/api/jobs",
            files={"file": ("sample.pdf", fh.read(), "application/pdf")},
            data={"direction": "en2ar", "mirror": "true"},
        )
    assert response.status_code == 202
    job_id = response.json()["id"]

    job = wait_for(auth_client, job_id)
    assert job["status"] == "done"
    assert job["progress"] == 100
    assert job["download_ready"] is True

    download = auth_client.get(f"/api/jobs/{job_id}/download")
    assert download.status_code == 200
    assert download.content[:5] == b"%PDF-"
    assert "sample_ar.pdf" in download.headers["content-disposition"]


def test_full_flow_docx(auth_client, sample_docx):
    with open(sample_docx, "rb") as fh:
        response = auth_client.post(
            "/api/jobs",
            files={"file": ("sample.docx", fh.read(),
                            "application/vnd.openxmlformats-officedocument."
                            "wordprocessingml.document")},
            data={"direction": "en2ar", "mirror": "true"},
        )
    job = wait_for(auth_client, response.json()["id"])
    assert job["status"] == "done"

    download = auth_client.get(f"/api/jobs/{job['id']}/download")
    assert download.status_code == 200
    assert download.content[:2] == b"PK"


def test_job_reports_qa(auth_client, sample_pdf):
    with open(sample_pdf, "rb") as fh:
        response = auth_client.post(
            "/api/jobs",
            files={"file": ("sample.pdf", fh.read(), "application/pdf")},
            data={"direction": "en2ar"},
        )
    job = wait_for(auth_client, response.json()["id"])
    assert job["qa"]["counts"]["info"] > 0

    qa = auth_client.get(f"/api/jobs/{job['id']}/qa").json()
    assert "entries" in qa


def test_download_requires_auth(auth_client, sample_pdf, client):
    with open(sample_pdf, "rb") as fh:
        response = auth_client.post(
            "/api/jobs",
            files={"file": ("sample.pdf", fh.read(), "application/pdf")},
            data={"direction": "en2ar"},
        )
    job = wait_for(auth_client, response.json()["id"])
    auth_client.post("/api/logout")
    assert auth_client.get(f"/api/jobs/{job['id']}/download").status_code == 401


def test_unknown_job_returns_404(auth_client):
    assert auth_client.get("/api/jobs/nosuchjob").status_code == 404


def test_health_is_public(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
