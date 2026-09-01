"""Authentication: bcrypt-hashed credentials + signed session cookies.

v1 is a single shared account, but the check goes through `verify_user` and a
user store, so swapping in a real multi-user table means changing one function
and nothing else.
"""
from __future__ import annotations

import os
import secrets
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from passlib.context import CryptContext

SESSION_COOKIE = "dt_session"
SESSION_MAX_AGE = 60 * 60 * 8  # 8 hours

_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _secret_key() -> str:
    key = os.environ.get("SECRET_KEY")
    if key:
        return key
    # Ephemeral key: sessions do not survive a restart, which is the safe
    # default. Set SECRET_KEY in the environment for a stable deployment.
    return secrets.token_urlsafe(48)


SECRET_KEY = _secret_key()
_serializer = URLSafeTimedSerializer(SECRET_KEY, salt="doc-translator-session")


def hash_password(plain: str) -> str:
    return _pwd.hash(plain)


def _load_users() -> dict[str, str]:
    """username -> bcrypt hash.

    The password is hashed at startup and the plaintext is never stored, logged
    or sent to the browser.
    """
    username = os.environ.get("APP_USERNAME", "admin")
    env_hash = os.environ.get("APP_PASSWORD_HASH")
    if env_hash:
        return {username: env_hash}
    plain = os.environ.get("APP_PASSWORD", "changeme")
    return {username: hash_password(plain)}


USERS = _load_users()


def verify_user(username: str, password: str) -> bool:
    """Constant-time-ish check. Unknown users still run a hash comparison so a
    wrong username and a wrong password take the same time to reject."""
    stored = USERS.get(username)
    if stored is None:
        _pwd.dummy_verify() if hasattr(_pwd, "dummy_verify") else _pwd.hash("x")
        return False
    try:
        return _pwd.verify(password, stored)
    except Exception:
        return False


def create_session(username: str) -> str:
    return _serializer.dumps({"sub": username})


def read_session(token: str) -> Optional[str]:
    try:
        data = _serializer.loads(token, max_age=SESSION_MAX_AGE)
        return data.get("sub")
    except (BadSignature, SignatureExpired, Exception):
        return None


def current_user(request: Request) -> Optional[str]:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    return read_session(token)


def require_user(request: Request) -> str:
    """Dependency for protected API routes. 401 tells the frontend to bounce
    the user back to the login page."""
    user = current_user(request)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Your session has expired. Please sign in again.",
        )
    return user
