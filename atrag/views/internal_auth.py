import os
import secrets

from fastapi import Header, HTTPException


async def require_internal_service(
    internal_token: str | None = Header(default=None, alias="X-ATRAG-Internal-Token"),
) -> None:
    """Authenticate calls made by trusted ATRAG worker processes."""
    expected_token = os.getenv("ATRAG_INTERNAL_SERVICE_TOKEN")
    if not expected_token:
        raise HTTPException(status_code=503, detail="Internal service authentication is not configured")
    if not internal_token or not secrets.compare_digest(internal_token, expected_token):
        raise HTTPException(status_code=401, detail="Invalid internal service credential")
