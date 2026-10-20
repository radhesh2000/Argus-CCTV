"""Camera credentials stay on the appliance.

A camera source is usually ``rtsp://user:password@host/...``, and that one
column was leaving the box four ways: verbatim in every API response that
carried a camera row (so a REVIEW account could read every camera's password),
printed by the console next to "Edit camera", echoed in ``argus status`` and the
engine's start-up lines, and quoted back by ffmpeg and go2rtc in the error text
that becomes ``status_detail`` and ``go2rtc.log``.

The rule this module implements is that the password is **write-only**. It goes
in through the enrolment form and is read back out only by the two things that
open the camera — the gateway config and a direct decoder. Nothing that renders
for a person gets it, whatever their role: not the API, not the CLI, not a log.

That makes editing the interesting case. The console shows
``rtsp://user:***@host/...`` and sends the same string back on save, so
:func:`restore_password` puts the stored password where the ``***`` is — and
only when the URL still points at the same login on the same host. Without that
condition a RESPOND account could change the host to a machine of their own,
keep the ``***``, and have the appliance deliver the password it was never
allowed to read.

``***`` cannot be a real password here: ``*`` is in the gateway's
``_SHELL_UNSAFE`` set, so no source containing one is ever stored.
"""

from __future__ import annotations

import re
from typing import Any, NamedTuple

#: What stands in for a password wherever a source is shown.
MASK = "***"

_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*")

# A URL with a password, found inside other text: an ffmpeg error, a go2rtc log
# line, an exception message. The password runs to the *last* ``@`` before the
# path, which is where ffmpeg, go2rtc and urllib all end the userinfo.
_EMBEDDED = re.compile(r"([A-Za-z][A-Za-z0-9+.\-]*://[^\s/?#@:]*):[^\s/?#]*@")

# A source whose password has a raw ``/`` in it does not parse as a URL at all —
# ``rtsp://user:pa/ss@host`` reads as host ``user``, port ``pa`` — and would be
# shown whole. Nothing can open such a source, but somebody typed a real
# password into it. Hiding too much of a strange URL is the safe direction.
_LOOSE = re.compile(r"^([^\s/?#@:\[]*):.*@", re.S)


class _Parts(NamedTuple):
    scheme: str  # with its "://"
    user: str
    password: str
    hostport: str
    tail: str  # path, query and fragment


def _split(source: str) -> _Parts | None:
    """A URL that carries a password, taken apart; None for anything else.

    Done on the string rather than through ``urllib.parse`` so that putting the
    parts back together returns exactly what was typed — a percent-encoded
    password must survive an edit byte for byte.
    """
    scheme, separator, rest = source.partition("://")
    if not separator or not _SCHEME.fullmatch(scheme):
        return None
    end = min((at for at in (rest.find(char) for char in "/?#") if at != -1), default=len(rest))
    userinfo, at, hostport = rest[:end].rpartition("@")
    user, colon, password = userinfo.partition(":")
    if not at or not colon:
        return None
    return _Parts(scheme + separator, user, password, hostport, rest[end:])


def swap_password(source: str, password: str) -> str:
    """``source`` with its password replaced. Unchanged when it has none."""
    parts = _split(source)
    if parts is None:
        return source
    return f"{parts.scheme}{parts.user}:{password}@{parts.hostport}{parts.tail}"


def redact_source(source: Any) -> str:
    """A camera source as it may be shown: ``rtsp://user:***@host/...``.

    The username stays — it is how an installer tells which NVR account a camera
    is enrolled under. A file path, and a URL with no password, come back as
    they went in.
    """
    text = str(source)
    if _split(text) is not None:
        return swap_password(text, MASK)
    scheme, separator, rest = text.partition("://")
    if separator and _SCHEME.fullmatch(scheme):
        return scheme + separator + _LOOSE.sub(lambda match: f"{match.group(1)}:{MASK}@", rest)
    return text


def redact_text(text: str | None, *sources: Any) -> str | None:
    """Free text with every URL password in it masked.

    ``sources`` are sources the caller knows may be quoted — replaced verbatim
    first, so one that does not parse as a URL is still caught.

    Redact, *then* truncate. ``text[:200]`` on an unredacted line can cut a URL
    in the middle of its password, leaving nothing for the pattern to match.
    """
    if not text:
        return text
    for source in sources:
        raw = str(source) if source else ""
        shown = redact_source(raw)
        if shown != raw:
            text = text.replace(raw, shown)
    return _EMBEDDED.sub(lambda match: f"{match.group(1)}:{MASK}@", text)


def redact_tree(value: Any, *sources: Any) -> Any:
    """:func:`redact_text` over every string in a decoded JSON document.

    For a payload whose shape belongs to somebody else — go2rtc's stream table
    names the camera URL under ``url`` today and may name it elsewhere tomorrow.
    """
    if isinstance(value, str):
        return redact_text(value, *sources)
    if isinstance(value, dict):
        return {key: redact_tree(item, *sources) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_tree(item, *sources) for item in value]
    return value


def redact_camera(camera: dict[str, Any] | None) -> dict[str, Any] | None:
    """A cameras row as the API may return it.

    ``status_detail`` is scrubbed along with ``source``: it is the decoder's
    last error, which is ffmpeg quoting the URL it could not open.
    """
    if camera is None:
        return None
    shown = dict(camera)
    source = shown.get("source")
    if source is not None:
        shown["source"] = redact_source(source)
    if shown.get("status_detail"):
        shown["status_detail"] = redact_text(str(shown["status_detail"]), source)
    return shown


def is_masked(source: str) -> bool:
    """Whether ``source`` carries the placeholder where a password goes."""
    parts = _split(source)
    return parts is not None and parts.password == MASK


class PasswordNotKept(ValueError):
    """A masked source that cannot be given its stored password back."""


def restore_password(submitted: str, stored: str | None) -> str:
    """``submitted`` with a ``***`` password replaced by the stored one.

    A source that is not masked is returned as it is — that is somebody typing
    a new password, or a camera that has none. A masked one keeps the stored
    password only while the scheme, user, host and port are all unchanged; see
    the module docstring for what that is defending.
    """
    new = _split(submitted)
    if new is None or new.password != MASK:
        return submitted
    old = _split(stored) if stored else None
    if old is None:
        raise PasswordNotKept(
            f"this camera has no stored password to keep — type one in place of {MASK}"
        )
    if (new.scheme.lower(), new.user, new.hostport.lower()) != (
        old.scheme.lower(), old.user, old.hostport.lower()
    ):
        raise PasswordNotKept(
            f"the stored password is only kept for the same scheme, user, host and port; "
            f"this source changes one of them, so type the password in place of {MASK}"
        )
    return swap_password(submitted, old.password)


__all__ = [
    "MASK",
    "PasswordNotKept",
    "is_masked",
    "redact_camera",
    "redact_source",
    "redact_text",
    "redact_tree",
    "restore_password",
    "swap_password",
]
