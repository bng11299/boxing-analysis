"""Optional: send the narration prompt to Gemini, and keep its reply only if narrate.check accepts it.

The template summary is the report. This is the one place in the repo that may use the network, and only when
the caller asks for it (`--narrate gemini`). Everything here fails closed: it never raises. It returns the
narrated text, or None and a short reason, and the caller keeps the template.

What is sent: the prompt's `system` and `user` text and nothing else. Those are the report's own sentences
(narrate.build): counts, rates and trust tags. No video, frames, pose, file paths or file names. A prompt marked
`may_leave_machine: false` (participant-derived numbers, vault ISSUES-LOG O63-O65) is never sent.

The key: read from the environment variable GEMINI_API_KEY only, at call time. Never from a file, a CLI argument
or a default. It travels in the `x-goog-api-key` header, not the URL, so it cannot land in a URL, a proxy log or
an exception's text, and no reason string this module returns contains it or anything the server sent back.

Plain HTTPS through the standard library (urllib, certificate-verified by default), with a timeout and a cap on
the reply's size: no SDK to pin.
"""

from __future__ import annotations

import json
import os
import re
import socket
import urllib.error
import urllib.request

from . import narrate

ENV_KEY = "GEMINI_API_KEY"
DEFAULT_MODEL = "gemini-2.5-flash"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
TIMEOUT_S = 30.0
MAX_REPLY_BYTES = 256 * 1024
_MODEL = re.compile(r"gemini-[a-z0-9.-]{1,60}")
# Google API keys are short URL-safe tokens; anything else (spaces, newlines, non-ASCII) is refused unsent, which
# also rules out header injection through the environment.
_KEY = re.compile(r"[A-Za-z0-9_\-]{20,200}")


def _key() -> "str | None":
    k = os.environ.get(ENV_KEY, "")
    return k if _KEY.fullmatch(k) else None


def request_body(prompt: dict) -> bytes:
    """The exact bytes sent: the system and user prompts, and generation settings. Nothing else."""
    return json.dumps({
        "systemInstruction": {"parts": [{"text": prompt["system"]}]},
        "contents": [{"role": "user", "parts": [{"text": prompt["user"]}]}],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
    }, ensure_ascii=False).encode("utf-8")


def _post(url: str, body: bytes, key: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Content-Type": "application/json", "x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=timeout) as r:   # noqa: S310 (fixed https URL)
        raw = r.read(MAX_REPLY_BYTES + 1)
    if len(raw) > MAX_REPLY_BYTES:
        raise ValueError("reply too large")
    return raw


def _reply_text(raw: bytes) -> "str | None":
    try:
        d = json.loads(raw.decode("utf-8"))
        parts = d["candidates"][0]["content"]["parts"]
        text = "".join(p["text"] for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str))
    except (ValueError, KeyError, IndexError, TypeError, UnicodeDecodeError, RecursionError):
        return None
    return text or None


def narrate_with_gemini(prompt: dict, model: str = DEFAULT_MODEL, timeout: float = TIMEOUT_S,
                        post=None) -> "tuple[str | None, str]":
    """(narrated text or None, reason). None means: keep the template. `post` is injectable for tests."""
    try:
        if not isinstance(prompt, dict) or prompt.get("format") != narrate.FORMAT:
            return None, "not a narration prompt"
        if prompt.get("may_leave_machine") is not True:
            return None, "participant-derived numbers never leave this machine; template kept"
        if not isinstance(model, str) or not _MODEL.fullmatch(model):
            return None, "model name refused"
        key = _key()
        if key is None:
            return None, f"no usable {ENV_KEY} in the environment; template kept"
        try:
            raw = (post or _post)(ENDPOINT.format(model=model), request_body(prompt), key, timeout)
        except urllib.error.HTTPError as e:     # status code only: the body may echo the request
            code = e.code if isinstance(e.code, int) else 0
            return None, f"Gemini refused the request (HTTP {code}); template kept"
        except (socket.timeout, TimeoutError):
            return None, "Gemini timed out; template kept"
        except (urllib.error.URLError, OSError, ValueError):
            return None, "network error reaching Gemini; template kept"
        draft = _reply_text(raw)
        if draft is None:
            return None, "Gemini's reply had no text; template kept"
        ok, problems, text = narrate.check(prompt, draft)
        if not ok or text is None:
            return None, f"Gemini's draft failed the fact check ({len(problems)} problem(s)); template kept"
        return text, "narrated by Gemini and accepted by the fact check"
    except Exception:   # fail closed, and say nothing that could carry the key or the reply
        return None, "narration failed; template kept"
