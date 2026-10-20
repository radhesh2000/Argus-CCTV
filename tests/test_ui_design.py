"""The console's legibility, checked by arithmetic rather than by eye.

Two properties are asserted here, and both of them regressed once already:

  1. **Contrast.** Every ink the design puts on a surface clears WCAG AA
     (4.5:1) against *the darkest surface it can actually land on*, not against
     the most flattering one. That distinction is the reason this file exists:
     a review of the shipped tokens found 59 failing pairs, several of which
     had been chosen against `--panel` and then used on `--panel2`.

  2. **Type floor.** Nothing in the product is set below 11px. The design's
     original 8px and 9px labels are what "the text is barely readable" meant.

Neither check needs a browser, a screenshot or a network. They read the same
stylesheets the appliance ships and do the sums.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

UI = Path(__file__).resolve().parents[1] / "ui" / "src"
TOKENS = UI / "styles" / "tokens.css"

# --------------------------------------------------------------------------
# WCAG relative luminance
# --------------------------------------------------------------------------


def _channel(value: int) -> float:
    c = value / 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _rgb(colour: str) -> tuple[int, int, int]:
    text = colour.strip()
    if text.startswith("#"):
        h = text[1:]
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    match = re.fullmatch(r"rgba?\(([^)]*)\)", text)
    if not match:  # pragma: no cover - a typo in the stylesheet, not a branch
        raise ValueError(f"not a colour: {colour!r}")
    parts = [p.strip() for p in match.group(1).split(",")]
    return int(parts[0]), int(parts[1]), int(parts[2])


def _alpha(colour: str) -> float:
    match = re.fullmatch(r"rgba\(([^)]*)\)", colour.strip())
    if not match:
        return 1.0
    parts = [p.strip() for p in match.group(1).split(",")]
    return float(parts[3]) if len(parts) > 3 else 1.0


def composite(colour: str, under: str) -> str:
    """Flatten a translucent colour onto the layer beneath it.

    A ratio measured against `rgba(...)` is meaningless; the eye sees the
    composite, so that is what gets measured.
    """
    alpha = _alpha(colour)
    if alpha >= 1.0:
        return colour
    fr, fg, fb = _rgb(colour)
    br, bg, bb = _rgb(under)
    return "#%02x%02x%02x" % (
        round(fr * alpha + br * (1 - alpha)),
        round(fg * alpha + bg * (1 - alpha)),
        round(fb * alpha + bb * (1 - alpha)),
    )


def luminance(colour: str) -> float:
    r, g, b = _rgb(colour)
    return 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)


def ratio(a: str, b: str) -> float:
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


# --------------------------------------------------------------------------
# the tokens, read from the file the appliance ships
# --------------------------------------------------------------------------

_DECL = re.compile(r"(--[a-z0-9-]+)\s*:\s*([^;]+);")


def _themes() -> dict[str, dict[str, str]]:
    """Parse tokens.css into {'light': {...}, 'dark': {...}}.

    The dark block overrides the light one, which is exactly how the cascade
    treats it, so the dark theme is the light theme updated — a token the dark
    block does not mention is still in play there.
    """
    css = TOKENS.read_text(encoding="utf-8")
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    blocks = re.findall(r"(:root|\[data-theme='dark'\])\s*\{(.*?)\}", css, flags=re.S)
    light: dict[str, str] = {}
    dark: dict[str, str] = {}
    for selector, body in blocks:
        target = light if selector == ":root" else dark
        for name, value in _DECL.findall(body):
            target[name] = value.strip()
    assert light, "no :root block in tokens.css"
    assert dark, "no dark block in tokens.css"
    return {"light": light, "dark": {**light, **dark}}


THEMES = _themes()

# Every surface an ink can be painted on. `--track` and `--line2` are bar
# grooves and hairlines, never text backgrounds, so they are not here.
TEXT_SURFACES = (
    "--bg",
    "--panel",
    "--panel2",
    "--hover",
    "--hover2",
    "--accent-soft",
    "--accent-soft2",
)

# Where a boundary is drawn. Not `--accent-soft2`: it is the pressed state of a
# soft button, and nothing is outlined against it.
EDGE_SURFACES = ("--bg", "--panel", "--panel2", "--hover", "--hover2")

# Rules live inside panels and on the page ground; they do not cross hovers.
RULE_SURFACES = ("--bg", "--panel", "--panel2")

# AA for body text. Nothing in this console qualifies for the 3:1 large-text
# allowance: the largest reading matter is 17px regular, and AA's large text
# starts at 18.66px bold or 24px regular.
AA_TEXT = 4.5
# WCAG 1.4.11: the boundary of a control, and any graphic carrying meaning.
AA_NON_TEXT = 3.0
# Not a WCAG threshold. A rule below about 1.2:1 is a rule nobody can see, and
# the shipped values were 1.01–1.11:1 — structure that was paid for and then
# not delivered. These floors are a design decision, asserted so it stays one.
VISIBLE_RULE = 1.22
VISIBLE_GROOVE = 1.45


def _ink_cases() -> list[tuple[str, str, str, float]]:
    cases = []
    for theme in ("light", "dark"):
        for ink in ("--text", "--text2", "--dim", "--faint", "--info"):
            for surface in TEXT_SURFACES:
                cases.append((theme, ink, surface, AA_TEXT))
        # Status inks land on the page, on panels and on their own soft fill.
        for ink, extra in (
            ("--good", ("--good-soft",)),
            ("--warn", ("--warn-soft",)),
            ("--crit", ("--warn-soft",)),
        ):
            for surface in TEXT_SURFACES + extra:
                cases.append((theme, ink, surface, AA_TEXT))
        # Inverse text: the label on a solid ink button.
        for surface in ("--accent", "--accent-h"):
            cases.append((theme, "--bg", surface, AA_TEXT))
        # Control edges and meaning-bearing graphics.
        for surface in EDGE_SURFACES:
            cases.append((theme, "--border-strong", surface, AA_NON_TEXT))
            cases.append((theme, "--good-dot", surface, AA_NON_TEXT))
            cases.append((theme, "--focus", surface, AA_NON_TEXT))
        # The label on a disabled control. WCAG exempts inactive components,
        # so this is a house rule: "inert" should not mean "unreadable".
        cases.append((theme, "--dim", "--track", AA_TEXT))
        # Structural rules and grooves: visible, not conformant.
        for surface in RULE_SURFACES:
            cases.append((theme, "--border", surface, VISIBLE_RULE))
            cases.append((theme, "--line2", surface, VISIBLE_RULE))
            cases.append((theme, "--track", surface, VISIBLE_GROOVE))
    return cases


@pytest.mark.parametrize("theme,ink,surface,need", _ink_cases())
def test_token_pair_is_legible(theme: str, ink: str, surface: str, need: float):
    tokens = THEMES[theme]
    fill = composite(tokens[surface], tokens["--bg"])
    got = ratio(composite(tokens[ink], fill), fill)
    assert got >= need, (
        f"{theme}: {ink} {tokens[ink]} on {surface} {tokens[surface]} "
        f"is {got:.2f}:1, needs {need}:1"
    )


def test_on_video_text_survives_a_bright_frame():
    """The over-video inks are measured against their scrim over *white*.

    A packing line under sodium lamps at noon is the frame that breaks an
    overlay, and the scrim is the only thing between the label and it.
    """
    tokens = THEMES["light"]
    for scrim_name in ("--on-video-scrim", "--on-video-scrim-strong"):
        scrim = composite(tokens[scrim_name], "#ffffff")
        for ink in ("--on-video", "--on-video-dim", "--on-video-faint"):
            got = ratio(tokens[ink], scrim)
            assert got >= AA_TEXT, (
                f"{ink} {tokens[ink]} on {scrim_name} over a white frame "
                f"is {got:.2f}:1"
            )


def test_the_two_themes_declare_the_same_tokens():
    """A token defined in one theme and forgotten in the other is a screen that
    is only styled half the time."""
    css = re.sub(r"/\*.*?\*/", "", TOKENS.read_text(encoding="utf-8"), flags=re.S)
    blocks = dict(
        (selector, body)
        for selector, body in re.findall(
            r"(:root|\[data-theme='dark'\])\s*\{(.*?)\}", css, flags=re.S
        )
    )
    light = {n for n, _ in _DECL.findall(blocks[":root"])}
    dark = {n for n, _ in _DECL.findall(blocks["[data-theme='dark']"])}
    # The dark block deliberately inherits type, fonts and the on-video family.
    inherited = {n for n in light if n.startswith(("--font", "--fs-", "--lh-", "--track-", "--on-video", "--hot", "--scrub"))}
    assert (light - inherited) - dark == set(), "light-only colour tokens"
    assert dark - light == set(), "dark-only tokens"


def test_no_orphan_tokens():
    """A token nothing uses is debris; a token used but never defined is a bug
    that renders as `unset`."""
    declared = {n for n in THEMES["light"] if not n.startswith("--fs-")}
    used: set[str] = set()
    for path in sorted(UI.rglob("*")):
        if path.suffix not in {".css", ".tsx", ".ts"} or path.name == "tokens.css":
            continue
        used |= set(re.findall(r"var\((--[a-z0-9-]+)\)", path.read_text(encoding="utf-8")))
    assert used - set(THEMES["light"]) == set(), "used but never declared"
    assert declared - used == set(), "declared but never used"


# --------------------------------------------------------------------------
# the type floor
# --------------------------------------------------------------------------

_CSS_SIZE = re.compile(r"font-size:\s*([0-9.]+)px")
_TSX_SIZE = re.compile(r"fontSize:\s*([0-9.]+)\b")
FLOOR = 11.0


def _sized_lines():
    for path in sorted(UI.rglob("*")):
        if path.suffix not in {".css", ".tsx"}:
            continue
        pattern = _CSS_SIZE if path.suffix == ".css" else _TSX_SIZE
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for size in pattern.findall(line):
                yield path, number, float(size), line.strip()


def test_nothing_is_set_below_the_type_floor():
    """11px is the floor, and it is for Space Mono furniture only.

    The wall-TV overlays scale with the tile and are exempt by construction —
    they compute their size from the viewport, and their *minimum* is asserted
    in test_wall_overlay_sizes_scale_from_a_legible_minimum below.
    """
    offenders = [
        f"{path.relative_to(UI)}:{number}  {size}px  {line}"
        for path, number, size, line in _sized_lines()
        if size < FLOOR
    ]
    assert not offenders, "text below the 11px floor:\n" + "\n".join(offenders)


def test_wall_overlay_sizes_scale_from_a_legible_minimum():
    """The TV wall is read from three to five metres, so its type scales with
    the viewport rather than sitting at desktop sizes — but a clamp's floor is
    still a floor, and this one is well above the console's.

    Every `font-size` inside the `.tv*` scope must be a clamp, because a fixed
    size there is the bug: the shipped wall set camera ids at 12px and the
    stale-feed line at 9px on a 55-inch panel.
    """
    css = (UI / "styles" / "components.css").read_text(encoding="utf-8")
    scope = css[css.index("--- wall TV ---") :]
    scope = scope[: scope.index("--- evidence viewer ---")]
    values = re.findall(r"font-size:\s*([^;]+);", scope)
    fixed = [v for v in values if not v.strip().startswith("clamp(")]
    assert not fixed, f"wall type set at a fixed size: {fixed}"
    floors = [float(m) for m in re.findall(r"font-size:\s*clamp\(\s*([0-9.]+)px", scope)]
    assert floors, "the wall no longer clamps its type"
    assert min(floors) >= 14, f"wall type floor dropped to {min(floors)}px"
