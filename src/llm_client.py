"""Provider-agnostic LLM call layer for answer generation and the groundedness
judge.

The provider is chosen in config.py from the LLM_PROVIDER env var (default:
Google Gemini's free tier via its OpenAI-compatible endpoint). Both the OpenAI
SDK path (gemini/openai/custom) and the Anthropic SDK path are supported behind
one small interface, so src/generate.py and src/judge.py never branch on the
provider themselves.

When no key is available for the selected provider, chat() raises NoApiKeyError
-- callers catch this and continue without fabricating any output.
"""
from __future__ import annotations

import time

import config


class NoApiKeyError(RuntimeError):
    """Raised when an LLM call is requested without an API key."""


_last_call_monotonic = 0.0


def _throttle() -> None:
    """Enforce a minimum gap between consecutive LLM calls (client-side rate
    limiting), so we stay under a provider's requests-per-minute cap instead of
    relying on 429 retries. Controlled by config.LLM_MIN_REQUEST_INTERVAL_S;
    a value of 0 disables it."""
    global _last_call_monotonic
    interval = config.LLM_MIN_REQUEST_INTERVAL_S
    if interval > 0:
        wait = interval - (time.monotonic() - _last_call_monotonic)
        if wait > 0:
            time.sleep(wait)
    _last_call_monotonic = time.monotonic()


def _no_key_message() -> str:
    hint = config.api_key_env_hint()
    return (
        f"No LLM API key found for provider '{config.LLM_PROVIDER}'. "
        f"Set {hint} (or LLM_API_KEY) and re-run to produce real LLM output. "
        f"The default provider is Gemini's free tier: "
        f"`export GEMINI_API_KEY=... && ./run.sh`."
    )


def _build_client(model: str | None):
    """Return (sdk_kind, client, model) for the configured provider.

    Raises NoApiKeyError if no key is set. SDKs are imported lazily so importing
    this module never requires a provider SDK to be installed.
    """
    key = config.resolve_api_key()
    if not key:
        raise NoApiKeyError(_no_key_message())

    model = model or config.LLM_MODEL
    if config.LLM_SDK == "anthropic":
        import anthropic

        return "anthropic", anthropic.Anthropic(api_key=key), model

    # gemini / openai / custom -> OpenAI-compatible SDK pointed at base_url.
    from openai import OpenAI

    # Free tiers (e.g. Gemini) return frequent 429/503s; let the SDK retry with
    # exponential backoff rather than failing a call on the first hiccup.
    client = OpenAI(
        api_key=key,
        base_url=config.LLM_BASE_URL,
        max_retries=config.LLM_MAX_RETRIES,
        timeout=config.LLM_TIMEOUT_S,
    )
    return "openai", client, model


def cost_usd(input_tokens, output_tokens) -> float:
    """Real token-based cost. Returns 0.0 when usage is unavailable or the
    provider is a $0 free tier (both price constants are 0)."""
    if input_tokens is None or output_tokens is None:
        return 0.0
    return (
        input_tokens / 1_000_000 * config.LLM_PRICE_INPUT_PER_MTOK
        + output_tokens / 1_000_000 * config.LLM_PRICE_OUTPUT_PER_MTOK
    )


def chat(user: str, system: str | None = None, max_tokens: int = 512,
         model: str | None = None, response_format: dict | None = None) -> dict:
    """Run a single-turn chat completion through the configured provider.

    Returns {text, input_tokens, output_tokens, cost_usd, model}. Token counts
    are the real values reported by the API (None if the response omits usage),
    never fabricated. Raises NoApiKeyError when no key is available.

    response_format (OpenAI-compatible providers only) can force structured
    output, e.g. {"type": "json_object"}; ignored by the Anthropic path.
    """
    kind, client, model = _build_client(model)
    _throttle()  # client-side pacing to stay under the provider's rate limit

    if kind == "anthropic":
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": user}],
        }
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        text = "".join(
            block.text for block in resp.content
            if getattr(block, "type", "") == "text"
        )
        usage = getattr(resp, "usage", None)
        in_tok = getattr(usage, "input_tokens", None) if usage else None
        out_tok = getattr(usage, "output_tokens", None) if usage else None
    else:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        create_kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if response_format is not None:
            create_kwargs["response_format"] = response_format
        resp = client.chat.completions.create(**create_kwargs)
        text = resp.choices[0].message.content or ""
        # OpenAI-compatible usage: prompt_tokens / completion_tokens (may be
        # absent on some gateways -> fall back to None, cost -> 0.0).
        usage = getattr(resp, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", None) if usage else None
        out_tok = getattr(usage, "completion_tokens", None) if usage else None

    return {
        "text": text,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": cost_usd(in_tok, out_tok),
        "model": model,
    }
