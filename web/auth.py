from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, Query, status


class TokenAuth:
    def __init__(self, token: str | None = None) -> None:
        self.token = token or secrets.token_urlsafe(32)

    def public_url(self, host: str, port: int) -> str:
        display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
        return f"http://{display_host}:{port}/app?token={self.token}"

    def verify_token_value(self, token: str | None) -> None:
        if not token or not secrets.compare_digest(token, self.token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing ModelForge startup token.",
            )

    async def dependency(
        self,
        authorization: str | None = Header(default=None),
        x_modelforge_token: str | None = Header(default=None),
        token: str | None = Query(default=None),
    ) -> None:
        candidate = token or x_modelforge_token
        if authorization and authorization.lower().startswith("bearer "):
            candidate = authorization.split(" ", 1)[1].strip()
        self.verify_token_value(candidate)
