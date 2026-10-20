"""Outbound alert delivery — what makes the ``auto_routed`` stage real.

**Email only.** v1 had SMTP, Slack and generic webhooks; v2 keeps SMTP and
drops the rest. A webhook to a cloud endpoint quietly contradicts the "raw
video never leaves site" claim the whole product is sold on, and every extra
channel is another thing to configure wrong on the day it matters. Plant SMTP
is on-site, already exists, and reaches phones through the same mail the
supervisor already reads.

Three rules this module enforces:

1. **Only ``auto_routed`` events are sent.** Shadow events are being validated
   in silence and reviewed events belong to a human in the triage queue; mail
   about either is exactly the alert fatigue the staged rollout exists to
   prevent. The event's own stage is used, not its policy's current one — the
   stage the event was raised under is what it was promised.
2. **Every attempt is recorded** in ``notifications``, sent or failed, one row
   per recipient. "Was anyone actually told?" must be answerable from the
   database alone, months later.
3. **The snapshot rides along.** An alert without evidence gets argued with; a
   mail with the annotated frame attached is the whole product in one message.

Credentials are never in the config file — ``smtp.password_env`` names the
environment variable that holds the password.
"""

from __future__ import annotations

import mimetypes
import smtplib
import threading
from dataclasses import dataclass, field
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from argus.config import ArgusConfig
from argus.db import Database, decoded_meta

# Cap on an attached snapshot. Annotated JPEGs run ~150 KB; anything far past
# this is a sign something is wrong, and mail servers reject big attachments.
MAX_ATTACHMENT_BYTES = 5_000_000

# The stage that means "a person is not in the loop any more".
AUTO_ROUTED = "auto_routed"


@dataclass
class DeliveryResult:
    """Outcome of one event's delivery, for logs and tests."""

    event_id: int
    sent: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    skipped: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "sent": self.sent,
            "failed": self.failed,
            "skipped": self.skipped,
        }

    def __str__(self) -> str:
        if self.skipped:
            return f"notify: event {self.event_id} not sent ({self.skipped})"
        parts = []
        if self.sent:
            parts.append(f"sent to {', '.join(self.sent)}")
        if self.failed:
            parts.append(f"FAILED for {', '.join(self.failed)}")
        return f"notify: event {self.event_id} " + ("; ".join(parts) or "no recipients")


def format_alert(event: dict[str, Any], *, camera_name: str | None = None,
                 link: str | None = None) -> tuple[str, str]:
    """Subject and plain-text body. Deliberately plain text: it renders in every
    client, including the phone notification preview that is all most people
    will read."""
    severity = str(event.get("severity", "medium")).upper()
    label = str(event.get("detector_name", "alert")).replace("_", " ")
    where = camera_name or event.get("camera_id") or "unknown camera"
    subject = f"[Argus/{severity}] {label} — {where}"

    meta = decoded_meta(event) or {}
    if meta.get("zone_name"):
        where = f"{where} / {meta['zone_name']}"

    lines = [
        str(event.get("description", "")),
        "",
        f"Where:      {where}",
        f"When:       {event.get('created_at', '')}",
        f"Severity:   {event.get('severity', '')}",
        f"Confidence: {float(event.get('confidence') or 0):.2f}",
        f"Routed to:  {event.get('routed_to') or 'unassigned'}",
        f"Event id:   {event.get('id')}",
    ]
    if link:
        lines += ["", f"Evidence:   {link}"]
    lines += [
        "",
        "This alert was routed automatically because its policy has been promoted",
        "to auto_routed. Reply to your Argus administrator to change routing.",
    ]
    return subject, "\n".join(lines)


def build_message(
    event: dict[str, Any],
    recipient: str,
    *,
    from_address: str,
    camera_name: str | None = None,
    link: str | None = None,
    snapshot: Path | None = None,
) -> EmailMessage:
    subject, body = format_alert(event, camera_name=camera_name, link=link)
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = from_address
    message["To"] = recipient
    message.set_content(body)
    if snapshot is not None and snapshot.is_file():
        try:
            if snapshot.stat().st_size <= MAX_ATTACHMENT_BYTES:
                guessed = mimetypes.guess_type(snapshot.name)[0] or "image/jpeg"
                maintype, _, subtype = guessed.partition("/")
                message.add_attachment(
                    snapshot.read_bytes(), maintype=maintype, subtype=subtype, filename=snapshot.name
                )
        except OSError:
            # A missing or unreadable snapshot must never stop the alert: the
            # text of the message is the part that matters.
            pass
    return message


class Notifier:
    """Delivers auto-routed events by email and records every attempt."""

    def __init__(self, config: ArgusConfig, database: Database, *, log=print):
        self.config = config
        self.db = database
        self.log = log

    # --- policy -----------------------------------------------------------

    def _skip_reason(self, event: dict[str, Any]) -> str | None:
        """Why this event is not being emailed, or None to send it."""
        if event.get("stage") != AUTO_ROUTED:
            return f"stage is {event.get('stage')!r}, not {AUTO_ROUTED!r}"
        if not event.get("routed_to"):
            return "policy has no routed_to role"
        if not self.config.smtp.recipients_for(event["routed_to"]):
            return f"no recipients configured for role {event['routed_to']!r}"
        return None

    # --- transport --------------------------------------------------------

    def _connect(self) -> smtplib.SMTP:
        smtp_config = self.config.smtp
        if smtp_config.port == 465:
            # Implicit TLS. Common enough on hosted mail that failing here with
            # a bare handshake error would waste an afternoon.
            server: smtplib.SMTP = smtplib.SMTP_SSL(smtp_config.host, smtp_config.port, timeout=20)
        else:
            server = smtplib.SMTP(smtp_config.host, smtp_config.port, timeout=20)
            if smtp_config.starttls:
                server.starttls()
        if smtp_config.username:
            server.login(smtp_config.username, smtp_config.password)
        return server

    # --- delivery ---------------------------------------------------------

    def deliver(self, event: dict[str, Any]) -> DeliveryResult:
        """Send one event synchronously. Safe to call from a test."""
        event_id = int(event["id"])
        result = DeliveryResult(event_id=event_id)

        reason = self._skip_reason(event)
        if reason is not None:
            result.skipped = reason
            # A promoted policy pointing at a role nobody configured is a
            # misconfiguration a human must see, so it is logged as a failed
            # attempt. A shadow event is working as designed and is not.
            if event.get("stage") == AUTO_ROUTED:
                self.db.log_notification(event_id, "none", str(event.get("routed_to") or "?"),
                                         "failed", reason)
                self.log(str(result))
            return result

        recipients = self.config.smtp.recipients_for(event["routed_to"])
        camera = self.db.get_camera(str(event["camera_id"]))
        snapshot = (
            self.config.snapshots_dir / str(event["snapshot_path"])
            if event.get("snapshot_path") else None
        )
        link = self.config.api.event_url(event_id)

        try:
            server = self._connect()
        except (OSError, smtplib.SMTPException) as exc:
            detail = f"{type(exc).__name__}: {exc}"[:300]
            for recipient in recipients:
                self.db.log_notification(event_id, "smtp", recipient, "failed", detail)
                result.failed.append(recipient)
            self.log(str(result))
            return result

        try:
            for recipient in recipients:
                # One message per recipient so a single bad address cannot sink
                # the batch, and so each gets its own auditable row — all on the
                # one connection.
                try:
                    server.send_message(build_message(
                        event, recipient,
                        from_address=self.config.smtp.from_address,
                        camera_name=camera["name"] if camera else None,
                        link=link,
                        snapshot=snapshot,
                    ))
                except (OSError, smtplib.SMTPException) as exc:
                    self.db.log_notification(event_id, "smtp", recipient, "failed",
                                             f"{type(exc).__name__}: {exc}"[:300])
                    result.failed.append(recipient)
                else:
                    self.db.log_notification(event_id, "smtp", recipient, "sent")
                    result.sent.append(recipient)
        finally:
            try:
                server.quit()
            except (OSError, smtplib.SMTPException):
                pass

        self.log(str(result))
        return result

    def dispatch(self, event_id: int) -> threading.Thread | None:
        """Fire-and-forget delivery on a daemon thread.

        The vision loop calls this. An SMTP server that takes 20 seconds to
        answer must not cost 20 seconds of dropped frames, and an alert that
        fails to send must not take the engine down with it.
        """
        event = self.db.get_event(event_id)
        if not event:
            return None
        if event.get("stage") != AUTO_ROUTED:
            # Cheap early exit on the hot path: most events are shadow, and
            # spawning a thread per shadow event is pure waste.
            return None

        def run() -> None:
            try:
                self.deliver(event)
            except Exception as exc:  # noqa: BLE001 - delivery must never crash the engine
                self.log(f"notify: event {event_id} delivery crashed "
                         f"({type(exc).__name__}: {exc})")

        thread = threading.Thread(target=run, daemon=True, name=f"notify-{event_id}")
        thread.start()
        return thread


__all__ = ["AUTO_ROUTED", "DeliveryResult", "Notifier", "build_message", "format_alert"]
