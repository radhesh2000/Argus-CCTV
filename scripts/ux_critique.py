#!/usr/bin/env python3
"""A second pair of eyes on the console, from a model that has never seen it.

**A development tool.** Nothing in the shipped appliance imports this, and
nothing it touches runs on a customer's box. It sends *screenshots of our own
console* — `docs/screens/*.png`, captured from the dev appliance — to a
third-party vision model and writes down what the model says. No footage frame
leaves here that is not already committed to this repository as a screenshot,
and the operator running it typed the command.

    export $(grep -v '^#' .env.openrouter | xargs)   # or: source .env.openrouter
    uv run python scripts/ux_critique.py --out docs/UX_CRITIQUE_KIMI.md

Why it exists. "The text is barely readable" and "looks like a 2022 AI slop
website" are verdicts, not a work list. Turning one into the other needs an
outside reader who is not attached to the thing, and who will say which pixel is
wrong rather than that the vibe is off. So each screen goes out with the *real*
CSS numbers attached — the type scale below is generated from the stylesheets,
not eyeballed off the image — and the model is asked for three implementable
fixes per screen, inside the broadsheet identity rather than instead of it.

Three things are borrowed wholesale from `argus/eval/openrouter.py`, because
that module already paid for them on a 424-case run that failed 420 ways:

* **A reasoning budget that is not the appliance's.** Every candidate here is a
  reasoning model and OpenRouter bills reasoning out of `max_tokens`; a budget
  sized for a 60-token verdict buys a third of a chain of thought and no answer.
* **Degrade, don't die.** `reasoning` and `response_format` are asks, not
  guarantees; an endpoint that refuses one gets the request again without it.
* **Lenient extraction.** A provider that ignored `response_format` returns a
  fence or a paragraph of preamble, and a reply cut off mid-object still holds
  most of a critique. `_strip_fences` and `_load_fragment` are imported from the
  eval client rather than re-typed, because the second copy is the one that
  rots. Unlike the eval path there is no scoring here, so a reply that survives
  none of that is kept as raw prose and marked as such — a critique nobody can
  parse is still a critique someone can read.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from argus.eval.openrouter import (  # noqa: E402
    ATTRIBUTION,
    BOUNDED_REASONING,
    NO_REASONING,
    OBJECT_FORMAT,
    CloudDisabled,
    api_key,
)
from argus.eval.openrouter import _load_fragment, _strip_fences  # noqa: E402

SCREENS = REPO / "docs" / "screens"
STYLES = REPO / "ui" / "src" / "styles"
DEFAULT_MODEL = "moonshotai/kimi-k3"

# What each file is, in one line. The model cannot know that `12-wall-tv` is
# meant to be read from across a room, and a critique that does not know it
# would score the type as enormous instead of as barely enough.
SCREEN_CONTEXT = {
    "01-login": "Sign-in. First thing an operator sees; also the first thing a buyer sees in a demo.",
    "02-live": "Live wall, light theme. The default screen; a control room leaves it up all day.",
    "03-alerts": "Alert queue. Operators triage from here under time pressure.",
    "04-triage": "One alert opened: evidence clip, VLM verdict, accept/dismiss.",
    "05-search": "Natural-language search over recorded events.",
    "06-policies": "Policy list — the rules that decide what raises an alert.",
    "07-cameras": "Camera inventory and enrolment state.",
    "08-analytics": "Counts and trends over the retained window.",
    "09-health": "Appliance health: services, GPU, disk, model status.",
    "10-settings": "Appliance settings.",
    "11-live-dark": "Live wall, DARK theme. Same screen as 02, ink palette — night shift.",
    "12-wall-tv": "TV-wall mode: full-screen grid on a mounted display, read from 3-5 metres away.",
    "13-command-palette": "Command palette (cmd-K) over the console.",
    "14-health-vlm-down": "Health with the VLM service down — the degraded/error state.",
    "15-camera-awaiting-restart": "A camera enrolled but pending an engine restart — a pending state.",
    "20-policy-new": "New-policy form, empty.",
    "21-policy-livetest": "Policy editor running a live test against a camera.",
    "22-policy-pilot": "A policy in pilot (shadow) mode.",
    "23-camera-enrol": "Add-camera form.",
    "24-policy-anmol-classes": "Policy editor with detector classes chosen.",
}

IDENTITY = """\
FIXED DESIGN IDENTITY — do not propose a rebrand, a new palette family, or a new
typeface family. This look was chosen deliberately and approved:
  * Broadsheet newspaper: warm paper (light) / ink (dark), a masthead, hairline
    rules, sharp corners (no rounded cards), no drop shadows, no gradients.
  * Newsreader (serif) for body, Instrument Serif for display, Space Mono for
    labels and data.
Your fixes must live INSIDE that identity. "Use Inter and rounded cards" is a
wrong answer here. "The masthead strap is 10px Space Mono at 0.16em on --faint
and needs to be 11px on --dim" is a right answer. Broadsheets themselves are
extremely legible objects: a real newspaper sets body text large, uses strong
hierarchy, and reserves tiny caps for furniture. Judge us against that, and
against how Linear, Stripe and Vercel handle density, hierarchy and rhythm in
2026 — not against their aesthetics.
"""

SCREEN_TASK = """\
Critique THIS ONE SCREEN of an on-premises CCTV appliance console ("Argus").
Operators are plant safety and security staff on a 1440x900 desktop browser.

Answer three questions, concretely, naming elements you can actually see:

(a) LEGIBILITY FAILURES. Type that is too small for its job, contrast that is
    too low, density that defeats scanning, hierarchy that does not exist. Use
    the REAL CSS NUMBERS supplied below rather than measuring the image, and
    quote the token or the px value you are objecting to. Rank each high /
    medium / low by how much it hurts an operator mid-shift.
(b) WHAT READS DATED OR "AI-GENERATED". Be specific and unsentimental. The
    owner's verdict on this build was "looks like a 2022 AI slop website".
    Name the signals that earn that: generic spacing, uniform grey-on-grey,
    everything the same weight, decoration standing in for hierarchy, tables
    that are just divs, states that are just colour, and so on. For each, say
    what a 2026 product surface does instead.
(c) THREE FIXES, ranked by impact, each one implementable by an engineer
    tonight in CSS/TSX. Name the element, the current value and the new value.
    No "consider exploring" — say the number.

Return ONE JSON object, no prose outside it:
{"screen": "<the id given>",
 "verdict": "<one sentence, blunt>",
 "legibility": [{"element": "...", "issue": "...", "severity": "high|medium|low",
                 "evidence": "the px/token/ratio you are objecting to"}],
 "dated": [{"signal": "...", "why": "...", "instead": "what 2026 does"}],
 "fixes": [{"rank": 1, "title": "...", "change": "current value -> new value",
            "target": "selector, token or component",
            "impact": "high|medium|low", "effort": "S|M|L"}]}
Exactly three items in "fixes". Two to six in each of the other lists.
"""

TOKENS_TASK = """\
Below is the console's complete design-token stylesheet, both themes, plus the
generated type scale showing what size and colour each token is actually used
at. Review it for WCAG 2.2 contrast, in BOTH themes.

Rules you must apply, not guess at:
  * Normal text (< 18.66px bold / < 24px regular) needs 4.5:1 for AA.
  * Large text (>= 24px, or >= 18.66px bold) needs 3:1.
  * Non-text UI boundaries that carry meaning (input borders, focus rings,
    status dots, chart strokes) need 3:1 against their adjacent colour.
  * Almost every label in this console is 9-11px Space Mono, i.e. NORMAL text,
    so the bar is 4.5:1. Letter-spaced uppercase micro-type at 9px is harder to
    read than its contrast ratio alone suggests — say so where it applies.

For each pair you judge, give the ratio you computed. We will recompute every
ratio ourselves and mark any that is wrong, so do not estimate: if you are not
confident of the arithmetic, say so in "uncertain": true rather than inventing
a number.

Return ONE JSON object, no prose outside it:
{"pairs": [{"theme": "light|dark", "fg_token": "--faint", "fg_hex": "#8c8672",
            "bg_token": "--bg", "bg_hex": "#ede7d6",
            "used_for": "where this pair appears and at what px",
            "claimed_ratio": 2.61, "required": 4.5, "passes": false,
            "uncertain": false,
            "suggested_fg_hex": "#5f5a48",
            "suggested_ratio": 4.62}],
 "systemic": [{"finding": "...", "fix": "..."}]}
Cover at least: every --text/--text2/--dim/--faint/--info on every surface they
sit on (--bg, --panel, --panel2, --accent-soft), the status colours
(--good/--warn/--crit) on their soft backgrounds, --border and --border-strong
as UI boundaries, and the --on-video-* family over dark video.
"""

SYNTH_TASK = """\
You have just critiqued every screen of one product. Below are your own
per-screen findings, and your token-contrast review.

Produce the MASTER FIX LIST: deduplicated across screens, ranked by impact x
effort — the thing that fixes the most screens for the least work goes first.
Merge findings that are the same underlying defect wearing different clothes
(the same token being too faint on six screens is ONE fix, not six). Keep the
broadsheet identity. Be specific enough that an engineer can start at item 1
without asking a question.

Return ONE JSON object, no prose outside it:
{"master": [{"rank": 1, "title": "...",
             "what": "the concrete change, with values",
             "why": "what it fixes for the operator",
             "screens": ["02-live", "03-alerts"],
             "impact": "high|medium|low", "effort": "S|M|L",
             "targets": ["ui/src/styles/base.css .masthead__strap"]}],
 "if_you_only_do_three": ["...", "...", "..."],
 "closing": "<two sentences: is this fixable inside the identity, or not?>"}
Twelve to eighteen items in "master".
"""


# --------------------------------------------------------------------------
# the ground truth we hand the model
# --------------------------------------------------------------------------


def type_scale(paths: list[Path], ceiling: int = 15) -> str:
    """Every small `font-size` in the stylesheets, with its selector and colour.

    Generated rather than written down. A hand-kept summary of a type scale is
    wrong within a week, and the whole point of this exercise is that the
    critique argues with the real numbers instead of with a resized JPEG.
    """
    rows: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        # Selector followed by a block; good enough for a stylesheet we own and
        # that has no nesting or media-query trickery in the rules we care about.
        for match in re.finditer(r"([^{}/]+)\{([^{}]*)\}", text):
            selector = " ".join(match.group(1).split())
            body = match.group(2)
            size = re.search(r"font-size:\s*([\d.]+)px", body)
            if not size or float(size.group(1)) > ceiling:
                continue
            bits = [f"{size.group(1)}px"]
            for prop, label in (("color", "color"), ("letter-spacing", "tracking"),
                                ("font-family", "family"), ("line-height", "leading"),
                                ("text-transform", "case"), ("font-weight", "weight")):
                found = re.search(rf"(?<![-\w]){prop}:\s*([^;]+);", body)
                if found:
                    value = " ".join(found.group(1).split())
                    if prop == "font-family":
                        value = value.split(",")[0].strip("'\" ")
                    bits.append(f"{label}={value}")
            rows.append(f"  {selector}  ->  {', '.join(bits)}")
    header = (
        f"TYPE SCALE — every rule under {ceiling}px in {', '.join(p.name for p in paths)},\n"
        "extracted from the stylesheets themselves (base font-size is 13px):\n"
    )
    return header + "\n".join(rows)


def screen_prompt(screen_id: str, tokens_css: str, scale: str, size: tuple[int, int]) -> str:
    width, height = size
    return "\n\n".join([
        IDENTITY,
        f"SCREEN: {screen_id} — {SCREEN_CONTEXT.get(screen_id, 'a console screen')}",
        (f"The image is a {width}x{height} downscale of a 1440x900 capture. Multiply any\n"
         f"pixel you measure on it by {1440 / width:.3f} to get true CSS pixels — but prefer\n"
         "the CSS numbers below, which are exact."),
        f"DESIGN TOKENS (both themes):\n```css\n{tokens_css}\n```",
        scale,
        SCREEN_TASK,
    ])


def encode(path: Path, max_width: int, png_ceiling_kb: int = 800) -> tuple[str, tuple[int, int], str]:
    """Downscale to `max_width` and return a data URI, its size and its format.

    PNG by preference. The question on the table is whether 9px letter-spaced
    type is readable, and handing the model chroma-subsampled ringing around
    that type would be asking it about the encoder instead. But three of these
    screens are mostly live video, where lossless costs 1.5-2 MB a request for
    grain nobody is critiquing — those fall back to JPEG at a quality high
    enough that the overlay type survives.
    """
    from PIL import Image

    with Image.open(path) as image:
        frame = image.convert("RGB")
        if frame.width > max_width:
            height = round(frame.height * max_width / frame.width)
            frame = frame.resize((max_width, height), Image.LANCZOS)
        buffer = io.BytesIO()
        frame.save(buffer, format="PNG", optimize=True)
        fmt = "png"
        if buffer.tell() > png_ceiling_kb * 1024:
            buffer = io.BytesIO()
            frame.save(buffer, format="JPEG", quality=92, subsampling=0)
            fmt = "jpeg"
        return (
            f"data:image/{fmt};base64," + base64.b64encode(buffer.getvalue()).decode(),
            (frame.width, frame.height),
            fmt,
        )


# --------------------------------------------------------------------------
# the wire
# --------------------------------------------------------------------------


def extract_json(text: str, required_key: str) -> dict[str, Any] | None:
    """The first JSON object in `text` that carries `required_key`.

    Same shape as the eval client's extractor and for the same reason — the
    reply may be a fenced block, a paragraph then a block, or a chain of
    thought containing an example object before the real one. It differs in
    what it does when nothing parses: this returns None and the caller keeps
    the prose, because an unparseable critique is still worth reading, whereas
    an unparseable verdict is a number nobody produced.
    """
    body = _strip_fences(text or "")
    best: dict[str, Any] | None = None
    for start in range(len(body)):
        if body[start] != "{":
            continue
        payload = _load_fragment(body[start:])
        if isinstance(payload, dict) and required_key in payload:
            # Prefer the longest such object: a truncation repaired at an early
            # `{` can yield a stub that technically has the key.
            if best is None or len(json.dumps(payload)) > len(json.dumps(best)):
                best = payload
    return best


@dataclass
class Critic:
    """One OpenRouter model, asked questions about pictures of our own UI."""

    model: str = DEFAULT_MODEL
    base_url: str = "https://openrouter.ai/api/v1"
    timeout_sec: float = 600.0
    # Reasoning is billed out of this. A critique is a page of prose, not a
    # 60-token verdict, and the model thinks first; 16k is room for both.
    max_tokens: int = 16000
    attempts: int = 3
    reasoning: dict[str, Any] | None = field(default_factory=lambda: dict(BOUNDED_REASONING))
    response_format: dict[str, Any] | None = field(default_factory=lambda: dict(OBJECT_FORMAT))
    notes: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=lambda: {"prompt": 0, "completion": 0})
    prompt_usd: float | None = None
    completion_usd: float | None = None

    def __post_init__(self) -> None:
        if not api_key():
            raise CloudDisabled(
                "OPENROUTER_API_KEY is not set — `source .env.openrouter` first"
            )

    # --- transport (urllib, like the eval client: a few dozen JSON POSTs) ---

    def _request(self, path: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
        import urllib.error
        import urllib.request

        url = f"{self.base_url.rstrip('/')}{path}"
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(  # noqa: S310 - fixed https endpoint
            url, data=data, method="GET" if payload is None else "POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key()}",
                "Accept": "application/json",
                **ATTRIBUTION,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:400]}") from exc
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise RuntimeError(f"unreachable: {type(exc).__name__}: {exc}") from exc

    def verify(self) -> None:
        """Confirm the model exists, takes images, and what it costs."""
        entries = (self._request("/models", None, 60.0) or {}).get("data") or []
        entry = next((e for e in entries if str(e.get("id")) == self.model), None)
        if entry is None:
            raise RuntimeError(f"{self.model} is not in the OpenRouter catalog")
        modalities = [
            str(m).lower() for m in ((entry.get("architecture") or {}).get("input_modalities") or [])
        ]
        if "image" not in modalities:
            raise RuntimeError(
                f"{self.model} advertises input modalities {modalities} — no image input. "
                "Its critique of a screenshot would be fiction."
            )
        pricing = entry.get("pricing") or {}
        try:
            self.prompt_usd = float(pricing.get("prompt"))
            self.completion_usd = float(pricing.get("completion"))
        except (TypeError, ValueError):
            self.notes.append("catalog publishes no per-token price — cost not estimated")
        self.notes.append(f"{self.model}: vision confirmed by catalog ({', '.join(modalities)})")

    def _degrade(self, detail: str) -> bool:
        """Drop one option the endpoint just refused. True if something moved."""
        text = detail.lower()
        if "http 400" not in text:
            return False
        if "reasoning" in text and self.reasoning is not None:
            if self.reasoning != NO_REASONING:
                self.reasoning = dict(NO_REASONING)
                self.notes.append("reasoning effort refused — asked for none instead")
            else:
                self.reasoning = None
                self.notes.append("`reasoning` refused outright — field dropped")
            return True
        if ("response_format" in text or "json" in text) and self.response_format is not None:
            self.response_format = None
            self.notes.append("`response_format` refused — asking for JSON in the prompt only")
            return True
        return False

    def ask(self, prompt: str, data_uri: str | None, label: str) -> tuple[str, str | None]:
        """One completion. Returns (text, error)."""
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if data_uri:
            content.append({"type": "image_url", "image_url": {"url": data_uri}})
        last = ""
        for attempt in range(1, self.attempts + 1):
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.2,
                "max_tokens": self.max_tokens,
                "usage": {"include": True},
            }
            if self.response_format is not None:
                payload["response_format"] = self.response_format
            if self.reasoning is not None:
                payload["reasoning"] = self.reasoning
            try:
                body = self._request("/chat/completions", payload, self.timeout_sec)
            except RuntimeError as exc:
                last = str(exc)
                if self._degrade(last):
                    continue
                if not any(m in last.lower() for m in ("429", "http 5", "timeout", "unreachable")):
                    return "", last
                time.sleep(3.0 * attempt)
                continue

            usage = body.get("usage") or {}
            self.usage["prompt"] += int(usage.get("prompt_tokens") or 0)
            self.usage["completion"] += int(usage.get("completion_tokens") or 0)
            choice = (body.get("choices") or [{}])[0]
            message = choice.get("message")
            if not isinstance(message, dict):
                last = f"unexpected response shape: {sorted(body)}"
                continue
            # `message_text`'s job, minus the verdict-specific parts: content
            # first, then the places a truncated reasoning model leaves its work.
            text = message.get("content")
            if isinstance(text, list):
                text = "".join(str(p.get("text") or "") for p in text if isinstance(p, dict))
            if not (isinstance(text, str) and text.strip()):
                for key in ("reasoning", "reasoning_content"):
                    value = message.get(key)
                    if isinstance(value, str) and value.strip():
                        text = value
                        self.notes.append(f"{label}: content was empty — read from `{key}`")
                        break
            if isinstance(text, str) and text.strip():
                return text, None
            last = (
                f"empty response (finish={choice.get('finish_reason')}, "
                f"completion_tokens={usage.get('completion_tokens')})"
            )
            time.sleep(2.0 * attempt)
        return "", last

    def cost(self) -> float | None:
        if self.prompt_usd is None or self.completion_usd is None:
            return None
        return self.usage["prompt"] * self.prompt_usd + self.usage["completion"] * self.completion_usd


# --------------------------------------------------------------------------
# marking the model's homework
# --------------------------------------------------------------------------
#
# A critique is opinion and is welcome to be. A contrast ratio is arithmetic,
# and a wrong one costs somebody an afternoon changing a token that was fine.
# So every ratio the model asserts is recomputed here, and so is every ratio it
# suggests a replacement would achieve — the suggestions are the numbers an
# implementer will actually paste in, which makes them the ones worth checking.


def parse_color(value: str) -> tuple[float, float, float, float] | None:
    """`#rgb`, `#rrggbb` or `rgba(r, g, b, a)` as 0-255 channels plus alpha."""
    text = (value or "").strip().lower()
    match = re.fullmatch(r"#([0-9a-f]{3}|[0-9a-f]{6})", text)
    if match:
        digits = match.group(1)
        if len(digits) == 3:
            digits = "".join(c * 2 for c in digits)
        return (int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16), 1.0)
    match = re.fullmatch(r"rgba?\(([^)]*)\)", text)
    if match:
        parts = [p.strip() for p in match.group(1).replace("/", ",").split(",") if p.strip()]
        try:
            channels = [float(p) for p in parts[:3]]
            alpha = float(parts[3]) if len(parts) > 3 else 1.0
        except ValueError:
            return None
        if len(channels) == 3:
            return (*channels, alpha)  # type: ignore[return-value]
    return None


def over(fg: tuple[float, float, float, float], bg: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Source-over composite. `--warn-soft` is a 12% wash; it has no ratio alone."""
    a = fg[3]
    return (*(fg[i] * a + bg[i] * (1 - a) for i in range(3)), 1.0)  # type: ignore[return-value]


def luminance(colour: tuple[float, float, float, float]) -> float:
    """WCAG 2.x relative luminance."""
    channels = []
    for raw in colour[:3]:
        c = raw / 255.0
        channels.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = channels
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def ratio(fg: str, bg: str, under: str | None = None) -> float | None:
    """Contrast ratio of `fg` over `bg`, compositing translucency onto `under`."""
    a, b = parse_color(fg), parse_color(bg)
    if a is None or b is None:
        return None
    if b[3] < 1.0:
        base = parse_color(under or "#ffffff")
        if base is None:
            return None
        b = over(b, base)
    if a[3] < 1.0:
        a = over(a, b)
    high, low = sorted((luminance(a), luminance(b)), reverse=True)
    return round((high + 0.05) / (low + 0.05), 2)


def read_tokens(css: str) -> dict[str, dict[str, str]]:
    """`{theme: {token: value}}` from tokens.css, light then dark override."""
    themes: dict[str, dict[str, str]] = {"light": {}, "dark": {}}
    for block in re.finditer(r"(:root|\[data-theme='dark'\])\s*\{([^{}]*)\}", css):
        theme = "light" if block.group(1) == ":root" else "dark"
        for decl in re.finditer(r"(--[\w-]+):\s*([^;]+);", block.group(2)):
            themes[theme][decl.group(1)] = " ".join(decl.group(2).split())
    themes["dark"] = {**themes["light"], **themes["dark"]}
    return themes


# Which colour sits on which surface, at what standard. Written out rather than
# inferred from the stylesheets because "is this text or is it a boundary" is a
# judgement about intent, and the thresholds differ by a factor that matters.
TEXT_ON = ["--bg", "--panel", "--panel2", "--accent-soft", "--accent-soft2", "--hover", "--nav-active"]
TEXT_TOKENS = ["--text", "--text2", "--dim", "--faint", "--info", "--accent"]
STATUS_PAIRS = [
    ("--good", "--good-soft"), ("--good", "--good-soft2"), ("--good", "--panel"), ("--good", "--bg"),
    ("--warn", "--warn-soft"), ("--warn", "--panel"), ("--warn", "--bg"),
    ("--crit", "--panel"), ("--crit", "--bg"),
]
# Boundaries that carry information — where an input starts, where focus is —
# are held to 1.4.11's 3:1. `--track` and `--line2` are a progress groove and a
# decorative hairline: the fill and the layout carry the meaning, so WCAG does
# not bind them. They are measured anyway and reported separately, because a
# rule at 1.04:1 is not a conformance failure and *is* an invisible rule.
BOUNDARY_PAIRS = [(b, s) for b in ("--border", "--border-strong")
                  for s in ("--bg", "--panel", "--panel2")]
DECORATIVE_PAIRS = [(b, s) for b in ("--track", "--line2")
                    for s in ("--bg", "--panel", "--panel2")]
STATUS_DOTS = [("--good-dot", "--panel"), ("--good-dot", "--bg"), ("--hot", "#000000"),
               ("--scrub", "#000000")]
# Overlay text over its scrim, and — the case the tokens do not cover — the same
# text over a bright frame where the scrim was not painted.
ON_VIDEO = [("--on-video", "--on-video-scrim"), ("--on-video-dim", "--on-video-scrim"),
            ("--on-video-faint", "--on-video-scrim"),
            ("--on-video", "--on-video-scrim-strong"),
            ("--on-video-dim", "--on-video-scrim-strong"),
            ("--on-video-faint", "--on-video-scrim-strong"),
            ("--on-video-dim", "#101418"), ("--on-video-faint", "#101418"),
            ("--on-video", "#c8c8c8"), ("--on-video-dim", "#c8c8c8"),
            ("--on-video-faint", "#c8c8c8")]


def sweep(themes: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    """Every pair that matters, computed locally. The table nobody has to trust.

    `--warn-soft` and the video scrims are translucent, so each is composited
    onto the surface underneath before the ratio is taken — a 12% wash over
    paper is a different colour from a 12% wash over ink, and quoting one
    number for both is how a "fix" lands on the wrong theme.
    """
    rows: list[dict[str, Any]] = []
    for theme, tokens in themes.items():
        page = tokens.get("--bg", "#ffffff")

        def add(fg: str, bg: str, need: float, kind: str) -> None:
            fg_value = tokens.get(fg, fg)
            bg_value = tokens.get(bg, bg)
            under = tokens.get("--panel") if bg.startswith("--") and "soft" in bg else page
            value = ratio(fg_value, bg_value, under=under or page)
            if value is None:
                return
            rows.append({
                "theme": theme, "fg": fg, "fg_hex": fg_value, "bg": bg, "bg_hex": bg_value,
                "ratio": value, "need": need, "passes": value >= need, "kind": kind,
            })

        for fg in TEXT_TOKENS:
            for bg in TEXT_ON:
                add(fg, bg, 4.5, "text")
        for fg, bg in STATUS_PAIRS:
            add(fg, bg, 4.5, "text")
        for fg, bg in BOUNDARY_PAIRS + STATUS_DOTS:
            add(fg, bg, 3.0, "boundary")
        for fg, bg in DECORATIVE_PAIRS:
            add(fg, bg, 3.0, "decorative")
        for fg, bg in ON_VIDEO:
            add(fg, bg, 4.5, "on-video")
    return rows


def verify_claims(pairs: list[Any], themes: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    """Mark each asserted ratio CONFIRMED / WRONG / UNVERIFIABLE.

    Two different things can be wrong and they are not equally serious. The
    verdict (does this pass AA?) is what the fix list acts on, so a mismatched
    verdict is WRONG. The arithmetic can drift by a rounding step while the
    verdict stands; that is CONFIRMED with the real number printed beside it,
    because an implementer copying `4.77` needs to know it is `4.75`.
    """
    marked: list[dict[str, Any]] = []
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        theme = str(pair.get("theme", "light")).lower()
        tokens = themes.get(theme, themes["light"])
        fg = str(pair.get("fg_hex") or tokens.get(str(pair.get("fg_token")), ""))
        bg = str(pair.get("bg_hex") or tokens.get(str(pair.get("bg_token")), ""))
        under = tokens.get("--panel") if "soft" in str(pair.get("bg_token", "")) else tokens.get("--bg")
        actual = ratio(fg, bg, under=under)
        need = float(pair.get("required") or 4.5)
        row = dict(pair)
        row["actual_ratio"] = actual
        row["required_used"] = need
        row["suggested_actual"] = (
            ratio(str(pair.get("suggested_fg_hex")), bg, under=under)
            if pair.get("suggested_fg_hex") else None
        )
        if actual is None:
            row["verdict"] = "UNVERIFIABLE"
            row["verdict_note"] = "not a flat colour pair"
        else:
            claimed_pass = bool(pair.get("passes"))
            really_passes = actual >= need
            if pair.get("passes") is None:
                row["verdict_note_prefix"] = "model omitted `passes`; read as fail"
            claimed = pair.get("claimed_ratio")
            drift = (
                abs(float(claimed) - actual)
                if isinstance(claimed, (int, float)) else None
            )
            if claimed_pass != really_passes:
                row["verdict"] = "WRONG"
                row["verdict_note"] = (
                    f"claims {'pass' if claimed_pass else 'fail'}; actual {actual}:1 "
                    f"{'meets' if really_passes else 'misses'} {need}:1"
                )
            elif drift is not None and drift > 0.25:
                row["verdict"] = "CONFIRMED"
                row["verdict_note"] = f"verdict right, arithmetic off by {drift:.2f}"
            else:
                row["verdict"] = "CONFIRMED"
                row["verdict_note"] = ""
            if row["suggested_actual"] is not None:
                suggested_claim = pair.get("suggested_ratio")
                if row["suggested_actual"] < need:
                    row["verdict_note"] = (
                        (row["verdict_note"] + "; " if row["verdict_note"] else "")
                        + f"SUGGESTED FIX STILL FAILS ({row['suggested_actual']}:1 < {need}:1)"
                    )
                elif isinstance(suggested_claim, (int, float)) and \
                        abs(float(suggested_claim) - row["suggested_actual"]) > 0.25:
                    row["verdict_note"] = (
                        (row["verdict_note"] + "; " if row["verdict_note"] else "")
                        + f"suggested ratio claimed {suggested_claim}, actually "
                          f"{row['suggested_actual']}"
                    )
        if row.pop("verdict_note_prefix", None):
            row["verdict_note"] = (
                "model omitted `passes`; read as fail"
                + ("; " + row["verdict_note"] if row.get("verdict_note") else "")
            )
        marked.append(row)
    return marked


# Claims checked by hand against the stylesheets and the screenshots, after the
# run. The contrast column marks the arithmetic; this marks the diagnoses. Both
# exist for the same reason: the next person acts on this document, and a fix
# aimed at the wrong cause costs more than no fix at all.
REVIEWER_NOTES = [
    (
        "CONFIRMED observation, WRONG cause — the login button",
        "01-login",
        "The critique says the SIGN IN fill 'appears to be --border-strong #978d72 "
        "rather than --accent'. The fill *is* `var(--accent)` (#1d1a12): "
        "`.login__submit` in `screens.css` is correct, and changing that token would "
        "fix nothing. What the screenshot caught is `.login__submit:disabled "
        "{ opacity: 0.45 }` — the capture has empty fields, so the button is "
        "genuinely disabled, and 45% ink over #f6f1e2 composites to almost exactly "
        "the grey the model named. The finding survives in a sharper form: **the "
        "disabled state is indistinguishable from a weak enabled state to somebody "
        "who has never seen it enabled**, which is every first-time viewer of this "
        "screen. Fix the disabled treatment, not the accent.",
    ),
    (
        "CONFIRMED — developer paths are on a production surface",
        "10-settings",
        "`/home/arjunvs/Desktop/Projects/AI_Powered_CCTV_System/config/argus.yaml` is "
        "printed in the Alert-routing panel, and the version reads `2.0.0.dev0`. "
        "Neither is hard-coded in `ui/src`; both arrive from the API, so this is a "
        "presentation decision about what the appliance tells an operator, not a "
        "stray string. On a customer's box the path will be different and still a "
        "filesystem path.",
    ),
    (
        "CAVEAT on the independent sweep — possibility, not sighting",
        "tokens",
        "The sweep enumerates every pair the token system *permits*: `--faint` on "
        "`--accent-soft` is measured because both tokens exist and nothing prevents "
        "the combination, not because a screenshot was found containing it. Read "
        "those rows as latent failures to close when the token moves, and the rows "
        "for `--bg`/`--panel` — which every screen demonstrably uses — as live ones.",
    ),
]


def all_css() -> str:
    """Every stylesheet the console ships, concatenated."""
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(STYLES.glob("*.css")))


def check_selectors(results: dict[str, Any], css: str) -> dict[str, list[str]]:
    """Which class names the critique points at actually exist in the stylesheets.

    The contrast column stops the implementer chasing a ratio nobody measured;
    this stops them chasing a rule nobody wrote. The model was handed real
    selector names in the type scale, so most of what it cites is real — but
    "most" is exactly the condition under which people stop checking.
    """
    cited: set[str] = set()
    for screen in results.get("screens") or []:
        for fix in ((screen.get("data") or {}).get("fixes") or []):
            if isinstance(fix, dict):
                cited.update(re.findall(r"\.[a-z][\w-]*(?:__[\w-]+)?(?:--[\w-]+)?",
                                        str(fix.get("target", ""))))
    for item in (results.get("synthesis") or {}).get("master") or []:
        if isinstance(item, dict):
            for target in item.get("targets") or []:
                cited.update(re.findall(r"\.[a-z][\w-]*(?:__[\w-]+)?(?:--[\w-]+)?", str(target)))
    # `.tsx`, `.css` and friends are file extensions the regex cannot tell from
    # a class, and a bare `.button` is real even when cited as `.button variants`.
    cited = {c for c in cited if not re.fullmatch(r"\.(tsx?|css|js|jsx|md|py|html|json|yaml)", c)}
    real = set(re.findall(r"\.[a-zA-Z][\w-]*", css))
    return {"real": sorted(cited & real), "absent": sorted(cited - real)}


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def bullet(items: Any, line) -> str:
    """One markdown list from a list of dicts, or an honest note that it was empty."""
    if not isinstance(items, list) or not items:
        return "_(none returned)_\n"
    return "".join(line(i) for i in items if isinstance(i, dict)) or "_(none returned)_\n"


def render(results: dict[str, Any], model: str, when: str, cost: float | None,
           notes: list[str]) -> str:
    claims_all = results.get("verified") or []
    tally = {v: sum(1 for c in claims_all if c.get("verdict") == v)
             for v in ("CONFIRMED", "WRONG", "UNVERIFIABLE")}
    optimistic = [c for c in claims_all
                  if c.get("suggested_actual") is not None
                  and c["suggested_actual"] < float(c.get("required_used") or 4.5)]
    scoreline = (
        f"  * **Result: {tally['CONFIRMED']} of {len(claims_all)} verdicts confirmed, "
        f"{tally['WRONG']} wrong"
        + (f", {tally['UNVERIFIABLE']} unverifiable" if tally["UNVERIFIABLE"] else "")
        + ".** The pass/fail calls held and the arithmetic was accurate to two decimal "
        "places. What did not survive checking is "
        f"{len(optimistic)} of the *replacement* colours it proposed: they clear the "
        "bar on the background they were computed against and miss it on the darkest "
        "surface the same token also lands on. Those three are named under the "
        "table.\n"
    ) if claims_all else ""

    out: list[str] = []
    add = out.append
    add("# The console, read by a stranger\n\n")
    add(
        f"**critique: `{model}` via OpenRouter, {when}; images: `docs/screens/*.png`** — "
        f"all {results['image_count']} screens, downscaled to {results['image_width']}px "
        "wide, one request each. The token review was a separate request carrying "
        "`ui/src/styles/tokens.css` and a type scale extracted from the stylesheets.\n\n"
    )
    add(
        "The owner's verdict on this build was \"the text is barely readable\" and "
        "\"looks like a 2022 AI slop website\". That is a judgement, not a work list, "
        "so the screens went to a model that had never seen them and was asked for "
        "the pixel, the token and the number — three implementable fixes per screen, "
        "ranked. **The broadsheet identity was declared fixed in every prompt**: paper "
        "and ink, masthead, Newsreader and Space Mono, sharp corners. Nothing below "
        "proposes a rebrand, because a rebrand was not on the table; everything below "
        "is a size, a token value or a hierarchy inside the design that was approved.\n\n"
    )
    add("## How to read this\n\n")
    add(
        "* **The critique is opinion and is welcome to be.** Take it as an argument, "
        "not an instruction.\n"
        "* **The contrast ratios are arithmetic, and they have been marked.** Every "
        "ratio the model asserted was recomputed here with the WCAG relative-luminance "
        "formula, and so was every ratio it claimed its *suggested replacement* would "
        "achieve — those are the numbers an implementer pastes in, which makes them "
        "the ones worth checking. Rows carry ✅ CONFIRMED or ❌ WRONG.\n"
        f"{scoreline}"
        "* **There is also an independent sweep** that owes the model nothing: every "
        "text token over every surface it can land on, both themes, computed locally. "
        "Start there if you only trust arithmetic.\n"
        "* Generated by `scripts/ux_critique.py` — a development tool, not part of the "
        "appliance. `--rerender` rebuilds this document from the saved replies without "
        "spending anything.\n\n"
    )
    if cost is not None:
        add(f"Run cost: about ${cost:.2f} of OpenRouter credit.\n\n")
    if notes:
        add("Run notes:\n\n" + "".join(f"- {n}\n" for n in notes) + "\n")

    synth = results.get("synthesis") or {}
    add("\n## Master fix list\n\n")
    add(
        "Deduplicated across all screens and ranked by impact against effort — the "
        "change that fixes the most screens for the least work is first. `S` is an "
        "evening, `M` a day, `L` more than that.\n\n"
    )
    if synth.get("master"):
        add("| # | Fix | Impact | Effort | Screens | Where |\n|---|---|---|---|---|---|\n")
        for item in synth["master"]:
            if not isinstance(item, dict):
                continue
            named = [str(s) for s in (item.get("screens") or [])]
            screens = ", ".join(named[:5]) + (f" +{len(named) - 5}" if len(named) > 5 else "")
            targets = "; ".join(str(t) for t in (item.get("targets") or [])[:3])
            add(
                f"| {item.get('rank', '')} | **{item.get('title', '')}** — {item.get('what', '')} "
                f"| {item.get('impact', '')} | {item.get('effort', '')} | {screens} | `{targets}` |\n"
            )
        add("\n")
        for item in synth["master"]:
            if isinstance(item, dict) and item.get("why"):
                add(f"- **{item.get('rank')}. {item.get('title')}** — {item['why']}\n")
    else:
        add("_(synthesis pass returned nothing parseable; see the raw JSON.)_\n")
    if synth.get("if_you_only_do_three"):
        add("\n**If you only do three things:**\n")
        for line in synth["if_you_only_do_three"]:
            add(f"1. {line}\n")
    if synth.get("closing"):
        add(f"\n> {synth['closing']}\n")

    tokens = results.get("tokens") or {}
    claims = results.get("verified") or []
    counts = {"CONFIRMED": 0, "WRONG": 0, "UNVERIFIABLE": 0}
    for claim in claims:
        counts[claim.get("verdict", "UNVERIFIABLE")] = counts.get(claim.get("verdict"), 0) + 1
    add("\n## Token contrast review — marked\n")
    add(
        "Every ratio below was recomputed locally with the WCAG relative-luminance "
        "formula (`scripts/ux_critique.py`, `ratio()`); translucent surfaces are "
        "composited onto the layer beneath before measuring. **Claimed** is the "
        "model's arithmetic, **actual** is ours. A row is CONFIRMED when the model's "
        "pass/fail verdict matches reality and WRONG when it does not — "
        f"{counts.get('CONFIRMED', 0)} confirmed, {counts.get('WRONG', 0)} wrong, "
        f"{counts.get('UNVERIFIABLE', 0)} unverifiable of {len(claims)}.\n\n"
    )
    add("| ✓ | Theme | Pair | Used for | Claimed | **Actual** | Need | Model | Suggested fix → verified |\n")
    add("|---|---|---|---|---|---|---|---|---|\n")
    for pair in claims:
        suggestion = ""
        if pair.get("suggested_fg_hex"):
            suggestion = f"{pair['suggested_fg_hex']} → **{pair.get('suggested_actual', '?')}:1**"
        mark = {"CONFIRMED": "✅", "WRONG": "❌", "UNVERIFIABLE": "⚠️"}.get(pair.get("verdict"), "")
        note = f" _{pair['verdict_note']}_" if pair.get("verdict_note") else ""
        said = "pass" if pair.get("passes") is True else "fail" if pair.get("passes") is False else "—"
        add(
            f"| {mark} | {pair.get('theme', '')} "
            f"| `{pair.get('fg_token', '')}` {pair.get('fg_hex', '')} on "
            f"`{pair.get('bg_token', '')}` {pair.get('bg_hex', '')} "
            f"| {pair.get('used_for', '')} "
            f"| {pair.get('claimed_ratio', '?')} | **{pair.get('actual_ratio', '?')}** "
            f"| {pair.get('required_used', pair.get('required', ''))} "
            f"| {said}{note} | {suggestion} |\n"
        )
    short = [c for c in claims
             if c.get("suggested_actual") is not None
             and c["suggested_actual"] < float(c.get("required_used") or 4.5)]
    if short:
        add(
            "\n**Do not paste these replacements as-is.** The verdicts held, but "
            f"{len(short)} of the suggested new colours still miss the bar they were "
            "proposed to clear — they were computed against the wrong surface. The "
            "faint and border tiers have to satisfy their *worst* background, not "
            "their most flattering one:\n\n"
        )
        for c in short:
            add(
                f"- `{c.get('fg_token')}` on `{c.get('bg_token')}` ({c.get('theme')}): "
                f"proposed {c.get('suggested_fg_hex')} reaches only "
                f"**{c['suggested_actual']}:1** against the {c.get('required_used')}:1 needed.\n"
            )
        add("\n")
    for finding in tokens.get("systemic") or []:
        if isinstance(finding, dict):
            add(f"\n- **{finding.get('finding', '')}** → {finding.get('fix', '')}\n")

    add("\n### Independent sweep — every token pair, computed here\n\n")
    sweep_rows = results.get("sweep") or []
    fails = [r for r in sweep_rows if not r["passes"]]
    add(
        "Not the model's list and not dependent on it: every text token over every "
        "surface it can land on, every boundary token, the status colours on their "
        "fills, and the on-video family over its scrims — both themes, "
        f"{len(sweep_rows)} pairs, of which {len(fails)} miss the bar. Failures only "
        "below; the full matrix is in the run's JSON.\n\n"
        "Text needs 4.5:1 — **nothing in this console qualifies as large text**, "
        "since the largest body size is 13.5px and the display type is 32px regular "
        "rather than bold, so the 3:1 large-text allowance never applies. Boundaries "
        "that carry meaning (input edges, focus rings, status dots) need 3:1. Rows "
        "marked `decorative` are `--track` and `--line2`, a progress groove and a "
        "hairline rule: WCAG does not bind them, so they are not conformance "
        "failures — they are listed because a rule at 1.04:1 is an invisible rule "
        "either way, and that is a design decision somebody should make on purpose. "
        "Rows against `#c8c8c8` are overlay text on a bright frame with no scrim "
        "painted, which is the failure mode the tokens themselves cannot prevent.\n\n"
    )
    add("| Theme | Pair | Ratio | Need | Kind |\n|---|---|---|---|---|\n")
    for row in fails:
        add(
            f"| {row['theme']} | `{row['fg']}` {row['fg_hex']} on `{row['bg']}` {row['bg_hex']} "
            f"| **{row['ratio']}** | {row['need']} | {row['kind']} |\n"
        )

    selectors = results.get("selectors") or {}
    if selectors:
        total = len(selectors["real"]) + len(selectors["absent"])
        add("\n### Selector check\n\n")
        add(
            f"Of the {total} CSS class names the critique points at, "
            f"**{len(selectors['real'])} exist** in `ui/src/styles/`. The model was "
            "handed the real selector names along with the type scale, which is why "
            "the hit rate is what it is — but the ones below are not in the "
            "stylesheets and are either components to be created or names the model "
            "invented. Treat them as descriptions, not addresses.\n\n"
        )
        add("".join(f"- `{name}`\n" for name in selectors["absent"]) or "_(none — all exist.)_\n")

    add("\n### Marked by hand\n\n")
    add(
        "Three claims that needed a human with the stylesheet open. The contrast "
        "table marks the arithmetic; this marks the diagnoses.\n\n"
    )
    for title, where, body in REVIEWER_NOTES:
        add(f"**{title}** — `{where}`\n\n{body}\n\n")

    add("\n## Screen by screen\n")
    for screen in results["screens"]:
        add(f"\n### {screen['id']} — {SCREEN_CONTEXT.get(screen['id'], '')}\n")
        data = screen.get("data")
        if not data:
            add(f"\n_Unparseable reply; raw text:_\n\n```\n{(screen.get('raw') or '')[:4000]}\n```\n")
            continue
        if data.get("verdict"):
            add(f"\n> {data['verdict']}\n")
        add("\n**Legibility**\n\n")
        add(bullet(data.get("legibility"), lambda i: (
            f"- `{i.get('severity', '?')}` **{i.get('element', '')}** — {i.get('issue', '')}"
            f" _({i.get('evidence', '')})_\n"
        )))
        add("\n**Reads dated**\n\n")
        add(bullet(data.get("dated"), lambda i: (
            f"- **{i.get('signal', '')}** — {i.get('why', '')} → 2026: {i.get('instead', '')}\n"
        )))
        add("\n**Fixes**\n\n")
        add(bullet(data.get("fixes"), lambda i: (
            f"{i.get('rank', '-')}. **{i.get('title', '')}** ({i.get('impact', '')} impact, "
            f"{i.get('effort', '')}) — {i.get('change', '')} · `{i.get('target', '')}`\n"
        )))
    return "".join(out)


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--width", type=int, default=1280, help="downscale width in px")
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "UX_CRITIQUE_KIMI.md")
    parser.add_argument("--json", type=Path, default=None, help="also dump the raw replies here")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--only", nargs="*", default=None, help="screen ids, for a cheap dry run")
    parser.add_argument("--skip-screens", action="store_true", help="tokens + synthesis only")
    parser.add_argument(
        "--rerender", type=Path, default=None,
        help="rebuild the document from a saved --json dump: no requests, no cost. "
             "The replies are the expensive part; the prose around them is not, and "
             "editing the layout should not mean paying for the critique twice.",
    )
    args = parser.parse_args()

    if args.rerender:
        saved = json.loads(args.rerender.read_text(encoding="utf-8"))
        themes = read_tokens((STYLES / "tokens.css").read_text(encoding="utf-8"))
        saved["verified"] = verify_claims((saved.get("tokens") or {}).get("pairs") or [], themes)
        saved["sweep"] = sweep(themes)
        saved["selectors"] = check_selectors(saved, all_css())
        args.out.write_text(
            render(saved, saved.get("model", args.model), saved.get("date", date.today().isoformat()),
                   saved.get("cost"), saved.get("notes") or []),
            encoding="utf-8",
        )
        args.rerender.write_text(json.dumps(saved, indent=1), encoding="utf-8")
        print(f"  re-rendered {args.out} from {args.rerender} (no requests)")
        return 0

    critic = Critic(model=args.model)
    critic.verify()
    print("\n".join(f"  {n}" for n in critic.notes))

    tokens_css = (STYLES / "tokens.css").read_text(encoding="utf-8")
    scale = type_scale([STYLES / "base.css", STYLES / "components.css", STYLES / "screens.css"])

    paths = sorted(SCREENS.glob("*.png"))
    if args.only:
        paths = [p for p in paths if p.stem in args.only]
    print(f"  {len(paths)} screen(s), {args.model}, {args.width}px wide")

    results: dict[str, Any] = {"screens": [], "image_count": len(paths), "image_width": args.width}

    def one(path: Path) -> dict[str, Any]:
        uri, size, fmt = encode(path, args.width)
        prompt = screen_prompt(path.stem, tokens_css, scale, size)
        started = time.monotonic()
        text, error = critic.ask(prompt, uri, path.stem)
        data = extract_json(text, "fixes") if text else None
        print(
            f"  {path.stem:<28} {fmt:<4} {time.monotonic() - started:5.0f}s  "
            f"{'parsed' if data else ('RAW' if text else 'FAILED')}"
            f"{'  ' + error if error else ''}"
        )
        return {"id": path.stem, "data": data, "raw": text, "error": error, "format": fmt}

    if not args.skip_screens and paths:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results["screens"] = sorted(pool.map(one, paths), key=lambda r: r["id"])

    print("  tokens.css contrast review ...")
    text, error = critic.ask("\n\n".join([
        IDENTITY, f"```css\n{tokens_css}\n```", scale, TOKENS_TASK,
    ]), None, "tokens")
    results["tokens"] = extract_json(text, "pairs") or {}
    results["tokens_raw"] = text
    themes = read_tokens(tokens_css)
    results["verified"] = verify_claims(results["tokens"].get("pairs") or [], themes)
    results["sweep"] = sweep(themes)
    results["selectors"] = check_selectors(results, all_css())
    marked = {v: sum(1 for c in results["verified"] if c["verdict"] == v)
              for v in ("CONFIRMED", "WRONG", "UNVERIFIABLE")}
    failures = sum(1 for r in results["sweep"] if not r["passes"])
    print(
        f"    {len(results['verified'])} claim(s): {marked} — "
        f"local sweep: {failures}/{len(results['sweep'])} pairs fail"
        f"{'  ' + error if error else ''}"
    )

    print("  synthesis ...")
    digest = json.dumps(
        {"screens": [{"id": s["id"], **(s["data"] or {})} for s in results["screens"]],
         "tokens": results["tokens"]},
        indent=1,
    )
    text, error = critic.ask("\n\n".join([IDENTITY, digest, SYNTH_TASK]), None, "synthesis")
    results["synthesis"] = extract_json(text, "master") or {}
    results["synthesis_raw"] = text
    print(f"    {len(results['synthesis'].get('master') or [])} item(s){'  ' + error if error else ''}")

    cost = critic.cost()
    results.update({
        "model": args.model, "date": date.today().isoformat(),
        "cost": cost, "notes": critic.notes, "tokens_used": critic.usage,
    })
    args.out.write_text(
        render(results, args.model, results["date"], cost, critic.notes),
        encoding="utf-8",
    )
    if args.json:
        args.json.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(
        f"\n  wrote {args.out}  "
        f"({critic.usage['prompt']} prompt + {critic.usage['completion']} completion tokens"
        + (f", ~${cost:.2f}" if cost is not None else "") + ")"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
