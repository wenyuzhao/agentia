"""OAuth 2.1 for remote MCP servers (MCP authorization spec): protected-resource and authorization-server
metadata discovery, dynamic client registration, PKCE (S256), the code exchange and refresh. Stateless
steps: the caller persists everything between them, so a login can survive a restart."""

from __future__ import annotations

import base64
import hashlib
import secrets
import time
from typing import Any
from urllib.parse import urlencode, urlparse

import httpx
from pydantic import BaseModel, ConfigDict

from agentia.mcp.client import resource_metadata_url


def b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


class OAuthError(Exception):
    pass


class AuthMeta(BaseModel):
    model_config = ConfigDict(extra="allow")
    issuer: str | None = None
    authorization_endpoint: str
    token_endpoint: str
    registration_endpoint: str | None = None
    scopes: list[str] | None = None
    # The protected resource (RFC 8707), when the server published metadata for it.
    resource: str | None = None


class Tokens(BaseModel):
    model_config = ConfigDict(extra="allow")
    access_token: str
    token_type: str = "Bearer"
    expires_in: int | None = None
    refresh_token: str | None = None
    scope: str | None = None
    obtained_at: float = 0

    def expired(self) -> bool:
        return (
            bool(self.expires_in)
            and time.time() > self.obtained_at + (self.expires_in or 0) - 30
        )


async def _get_json(http: httpx.AsyncClient, url: str) -> dict[str, Any] | None:
    try:
        r = await http.get(
            url,
            headers={
                "accept": "application/json",
                "mcp-protocol-version": "2025-06-18",
            },
            timeout=15,
        )
    except httpx.HTTPError:
        return None
    if r.status_code != 200:
        return None
    try:
        j = r.json()
    except ValueError:
        return None
    return j if isinstance(j, dict) else None


async def discover(
    http: httpx.AsyncClient, server_url: str, www_authenticate: str | None = None
) -> AuthMeta:
    u = urlparse(server_url)
    origin = f"{u.scheme}://{u.netloc}"
    path = u.path.rstrip("/")
    prm = None
    candidates = [x for x in [resource_metadata_url(www_authenticate)] if x]
    candidates += [
        f"{origin}/.well-known/oauth-protected-resource{path}",
        f"{origin}/.well-known/oauth-protected-resource",
    ]
    for c in dict.fromkeys(candidates):
        prm = await _get_json(http, c)
        if prm:
            break
    issuer = str((prm or {}).get("authorization_servers", [origin])[0]).rstrip("/")
    iu = urlparse(issuer)
    iorigin = f"{iu.scheme}://{iu.netloc}"
    ipath = iu.path.rstrip("/")
    urls = (
        [
            f"{iorigin}/.well-known/oauth-authorization-server{ipath}",
            f"{iorigin}/.well-known/openid-configuration{ipath}",
            f"{issuer}/.well-known/openid-configuration",
        ]
        if ipath
        else [
            f"{iorigin}/.well-known/oauth-authorization-server",
            f"{iorigin}/.well-known/openid-configuration",
        ]
    )
    meta = None
    for x in urls:
        meta = await _get_json(http, x)
        if meta:
            break
    meta = meta or {
        "authorization_endpoint": f"{iorigin}/authorize",
        "token_endpoint": f"{iorigin}/token",
        "registration_endpoint": f"{iorigin}/register",
    }
    scopes = (prm or {}).get("scopes_supported")
    return AuthMeta.model_validate(
        {
            **meta,
            "issuer": meta.get("issuer") or issuer,
            "scopes": scopes if isinstance(scopes, list) else None,
            "resource": (prm or {}).get("resource"),
        }
    )


def client_metadata(redirect_uri: str, client_name: str = "agentia") -> dict[str, Any]:
    return {
        "client_name": client_name,
        "redirect_uris": [redirect_uri],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    }


async def register(
    http: httpx.AsyncClient,
    meta: AuthMeta,
    redirect_uri: str,
    client_name: str = "agentia",
) -> dict[str, Any]:
    if not meta.registration_endpoint:
        raise OAuthError(
            "the authorization server doesn't support dynamic client registration"
        )
    r = await http.post(
        meta.registration_endpoint,
        json=client_metadata(redirect_uri, client_name),
        timeout=15,
    )
    if r.status_code >= 400:
        raise OAuthError(
            f"client registration failed: HTTP {r.status_code} {r.text[:200]}"
        )
    info = r.json()
    if not info.get("client_id"):
        raise OAuthError("client registration returned no client_id")
    return info


def pkce() -> tuple[str, str]:
    verifier = b64url(secrets.token_bytes(48))
    return verifier, b64url(hashlib.sha256(verifier.encode()).digest())


def authorization_url(
    meta: AuthMeta,
    client_info: dict[str, Any],
    redirect_uri: str,
    state: str,
    challenge: str,
) -> str:
    q = {
        "response_type": "code",
        "client_id": client_info["client_id"],
        "redirect_uri": redirect_uri,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    if meta.resource:
        q["resource"] = meta.resource
    if meta.scopes:
        q["scope"] = " ".join(meta.scopes)
    return f"{meta.authorization_endpoint}?{urlencode(q)}"


async def _token(
    http: httpx.AsyncClient,
    meta: AuthMeta,
    client_info: dict[str, Any],
    form: dict[str, str],
) -> Tokens:
    form = {**form, "client_id": client_info["client_id"]}
    if client_info.get("client_secret"):
        form["client_secret"] = client_info["client_secret"]
    if meta.resource:
        form["resource"] = meta.resource
    r = await http.post(
        meta.token_endpoint,
        data=form,
        headers={"accept": "application/json"},
        timeout=20,
    )
    if r.status_code >= 400:
        raise OAuthError(f"token request failed: HTTP {r.status_code} {r.text[:200]}")
    return Tokens.model_validate({**r.json(), "obtained_at": time.time()})


async def exchange(
    http: httpx.AsyncClient,
    meta: AuthMeta,
    client_info: dict[str, Any],
    code: str,
    verifier: str,
    redirect_uri: str,
) -> Tokens:
    return await _token(
        http,
        meta,
        client_info,
        {
            "grant_type": "authorization_code",
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": redirect_uri,
        },
    )


async def refresh(
    http: httpx.AsyncClient, meta: AuthMeta, client_info: dict[str, Any], tokens: Tokens
) -> Tokens:
    if not tokens.refresh_token:
        raise OAuthError("no refresh token")
    t = await _token(
        http,
        meta,
        client_info,
        {"grant_type": "refresh_token", "refresh_token": tokens.refresh_token},
    )
    if not t.refresh_token:
        t.refresh_token = tokens.refresh_token
    return t
