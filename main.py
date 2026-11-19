"""
Free self-hosted AI gateway (OpenRouter-style).

- OpenAI-compatible: POST /v1/chat/completions, GET /v1/models
- Route to a specific model:  "groq/llama-3.3-70b-versatile"
- Or use an alias with auto-fallback: "free-auto", "free-fast", "free-smart"
- Streaming supported, automatic failover on 429 / 5xx / network errors
- Optional gateway API keys so only you (or friends) can use it

Run:  uvicorn main:app --host 0.0.0.0 --port 8000
"""
import os
import time

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

load_dotenv()

# ---------------------------------------------------------------------------
# Providers (all have free tiers or are local). Each exposes an
# OpenAI-compatible /chat/completions endpoint.
# key_env = None means no key needed (e.g. local Ollama).
# ---------------------------------------------------------------------------
PROVIDERS = {
    "groq":       {"base_url": "https://api.groq.com/openai/v1",                            "key_env": "GROQ_API_KEY"},
    "gemini":     {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai",   "key_env": "GEMINI_API_KEY"},
    "cerebras":   {"base_url": "https://api.cerebras.ai/v1",                                "key_env": "CEREBRAS_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1",                              "key_env": "OPENROUTER_API_KEY"},
    "mistral":    {"base_url": "https://api.mistral.ai/v1",                                 "key_env": "MISTRAL_API_KEY"},
    "ollama":     {"base_url": os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),        "key_env": None},
}

# ---------------------------------------------------------------------------
# Aliases: ordered fallback chains of (provider, model).
# Free model names change often, so check each provider's docs and edit these.
# ---------------------------------------------------------------------------
ALIASES = {
    "free-auto": [
        ("groq", "llama-3.3-70b-versatile"),
        ("cerebras", "llama-3.3-70b"),
        ("gemini", "gemini-2.0-flash"),
        ("openrouter", "meta-llama/llama-3.3-70b-instruct:free"),
        ("mistral", "mistral-small-latest"),
        ("ollama", "llama3.2"),
    ],
    "free-fast": [
        ("groq", "llama-3.1-8b-instant"),
        ("cerebras", "llama3.1-8b"),
        ("ollama", "llama3.2"),
    ],
    "free-smart": [
        ("gemini", "gemini-2.0-flash"),
        ("openrouter", "deepseek/deepseek-r1:free"),
        ("groq", "llama-3.3-70b-versatile"),
    ],
}

COOLDOWN_SECONDS = 60          # skip a provider this long after a 429
REQUEST_TIMEOUT = 120.0

GATEWAY_KEYS = {k.strip() for k in os.getenv("GATEWAY_KEYS", "").split(",") if k.strip()}

app = FastAPI(title="Free AI Gateway")
client = httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=10.0))
cooldowns: dict[str, float] = {}


def check_auth(authorization: str | None) -> None:
    if not GATEWAY_KEYS:
        return  # open gateway (fine for localhost, not for the internet)
    token = (authorization or "").removeprefix("Bearer ").strip()
    if token not in GATEWAY_KEYS:
        raise HTTPException(401, "Invalid gateway API key")


def resolve(model: str) -> list[tuple[str, str]]:
    if model in ALIASES:
        return ALIASES[model]
    if "/" in model:
        prov, name = model.split("/", 1)   # "openrouter/meta-llama/x:free" -> openrouter, meta-llama/x:free
        if prov in PROVIDERS:
            return [(prov, name)]
    raise HTTPException(400, f"Unknown model '{model}'. Use an alias or 'provider/model'.")


@app.get("/v1/models")
async def list_models(authorization: str | None = Header(None)):
    check_auth(authorization)
    ids = list(ALIASES) + [f"{p}/{m}" for chain in ALIASES.values() for p, m in chain]
    return {"object": "list", "data": [{"id": i, "object": "model", "owned_by": "gateway"} for i in dict.fromkeys(ids)]}


@app.post("/v1/chat/completions")
async def chat(request: Request, authorization: str | None = Header(None)):
    check_auth(authorization)
    body = await request.json()
    targets = resolve(body.get("model", "free-auto"))
    stream = bool(body.get("stream"))
    errors = []

    for prov, model in targets:
        cfg = PROVIDERS[prov]
        key = os.getenv(cfg["key_env"]) if cfg["key_env"] else "none"
        if not key:
            errors.append(f"{prov}: no API key set")
            continue
        if cooldowns.get(prov, 0) > time.time():
            errors.append(f"{prov}: cooling down")
            continue

        upstream = client.build_request(
            "POST",
            f"{cfg['base_url']}/chat/completions",
            json={**body, "model": model},
            headers={"Authorization": f"Bearer {key}"},
        )
        try:
            resp = await client.send(upstream, stream=True)
        except httpx.HTTPError as e:
            errors.append(f"{prov}: {type(e).__name__}")
            continue

        # Retryable failures -> try the next provider
        if resp.status_code == 429:
            cooldowns[prov] = time.time() + COOLDOWN_SECONDS
        if resp.status_code == 429 or resp.status_code >= 500 or resp.status_code in (401, 403, 404):
            errors.append(f"{prov}: HTTP {resp.status_code}")
            await resp.aclose()
            continue

        # Success (or a client error like 400 that retrying won't fix)
        if stream and resp.status_code == 200:
            async def relay(r=resp):
                try:
                    async for chunk in r.aiter_raw():
                        yield chunk
                finally:
                    await r.aclose()

            return StreamingResponse(relay(), media_type="text/event-stream",
                                     headers={"X-Gateway-Provider": prov, "X-Gateway-Model": model})

        data = await resp.aread()
        await resp.aclose()
        return Response(content=data, status_code=resp.status_code, media_type="application/json",
                        headers={"X-Gateway-Provider": prov, "X-Gateway-Model": model})

    return JSONResponse(status_code=503, content={"error": {"message": "All providers failed", "details": errors}})


@app.get("/health")
async def health():
    return {"ok": True, "providers": {p: bool(c["key_env"] is None or os.getenv(c["key_env"])) for p, c in PROVIDERS.items()}}
