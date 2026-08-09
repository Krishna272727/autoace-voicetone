"""Session-cookie auth against two environment variables.

No registration, no password reset, no user database -- the brief asks for a
login, not an identity system (BUILD_SPEC.md 11.2).

HTTP Basic was the first implementation. It works, but the browser's native
credential dialog is the entire login experience, there is no way to sign out
without closing the browser, and a failed attempt is a blank 401 page. This is
a signed session cookie instead, which the spec explicitly allows and which
lets the login be a real page.

The cookie is signed with HMAC-SHA256 from the standard library -- no new
dependency. It carries a username and an expiry, and it is signed rather than
encrypted: nothing secret is inside it, and the signature is what stops a
client editing the username or extending its own session.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time

from fastapi import HTTPException, Request, status

COOKIE_NAME = "autoace_session"
SESSION_HOURS = 12

DEFAULT_USER = "autoace"
DEFAULT_PASS = "changeme"


def credentials() -> tuple[str, str]:
    return (os.getenv("DASHBOARD_USER", DEFAULT_USER),
            os.getenv("DASHBOARD_PASS", DEFAULT_PASS))


def using_default_password() -> bool:
    """Surfaced in the UI. A deployment still on the shipped default is a
    finding, not a convenience -- say so on the page rather than in a log line
    nobody reads."""
    return credentials() == (DEFAULT_USER, DEFAULT_PASS)


def _secret() -> bytes:
    """Cookie-signing key.

    An explicit SESSION_SECRET is preferred. Falling back to a key derived from
    the password means sessions are invalidated whenever the password changes,
    which is the correct behaviour; it also means no secret has to be invented
    for a single-operator deployment.
    """
    explicit = os.getenv("SESSION_SECRET")
    if explicit:
        return explicit.encode()
    user, password = credentials()
    return hashlib.sha256(f"autoace:{user}:{password}".encode()).digest()


def _sign(payload: str) -> str:
    mac = hmac.new(_secret(), payload.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(mac).decode().rstrip("=")


def issue(username: str) -> str:
    """Build a signed cookie value for this user."""
    expires = int(time.time()) + SESSION_HOURS * 3600
    payload = f"{base64.urlsafe_b64encode(username.encode()).decode().rstrip('=')}.{expires}"
    return f"{payload}.{_sign(payload)}"


def verify(token: str | None) -> str | None:
    """Return the username if the cookie is intact and unexpired, else None."""
    if not token:
        return None
    try:
        user_b64, expires, sig = token.rsplit(".", 2)
    except ValueError:
        return None
    payload = f"{user_b64}.{expires}"
    # compare_digest, not ==, so a forged signature cannot be recovered one
    # byte at a time from response timing.
    if not hmac.compare_digest(_sign(payload), sig):
        return None
    try:
        if int(expires) < time.time():
            return None
        pad = "=" * (-len(user_b64) % 4)
        return base64.urlsafe_b64decode(user_b64 + pad).decode()
    except (ValueError, UnicodeDecodeError):
        return None


def check_password(username: str, password: str) -> bool:
    """Constant-time credential check.

    Both comparisons always run: `and` would short-circuit on a wrong username
    and leak, through response timing, which half was wrong.
    """
    want_user, want_pass = credentials()
    ok_user = hmac.compare_digest(username or "", want_user)
    ok_pass = hmac.compare_digest(password or "", want_pass)
    return bool(ok_user & ok_pass)


def current_user(request: Request) -> str | None:
    return verify(request.cookies.get(COOKIE_NAME))


def require_user(request: Request) -> str:
    """Route dependency. Raises 401; the handler turns that into a redirect to
    the login page for browser navigations, and leaves it as JSON for fetch()."""
    user = current_user(request)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="not signed in")
    return user


def set_cookie(response, username: str, secure: bool) -> None:
    response.set_cookie(
        COOKIE_NAME, issue(username),
        max_age=SESSION_HOURS * 3600,
        httponly=True,           # not readable from JavaScript
        samesite="lax",          # survives top-level navigation, blocks CSRF POSTs
        secure=secure,           # HTTPS-only once deployed behind TLS
        path="/",
    )


def clear_cookie(response) -> None:
    response.delete_cookie(COOKIE_NAME, path="/")
