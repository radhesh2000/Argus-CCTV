/**
 * The small repeated marks: severity glyphs, stage badges, and the two states
 * this appliance has to be able to say honestly — "not enabled yet" and
 * "nothing here".
 */

import type { ReactNode } from 'react';
import type { ArgusEvent, Severity, Stage } from '../api/types';
import { conditionLook, originLook, severityLook, stageLook } from '../lib/format';

export function SeverityGlyph({ severity, size = 10 }: { severity: Severity; size?: number }) {
  const look = severityLook(severity);
  return (
    <span style={{ color: look.color, fontSize: size }} title={look.label}>
      {look.glyph}
    </span>
  );
}

export function StageBadge({ stage }: { stage: Stage }) {
  const look = stageLook(stage);
  return (
    <span className={look.className} title={look.title}>
      {look.glyph} {look.label}
    </span>
  );
}

/**
 * Which engine raised an event — a prompt policy or the fast detector.
 *
 * Shown wherever events are listed, because the two are different kinds of
 * claim: the detector counted people in a polygon and can prove it with boxes;
 * the VLM read a sentence off a frame and can only prove it by quoting itself.
 * `short` for dense rows, the full label where there is room.
 */
export function OriginBadge({
  type,
  short = false,
}: {
  type: 'detector' | 'vlm';
  short?: boolean;
}) {
  const look = originLook(type);
  return (
    <span className={look.className} title={look.title}>
      {short ? look.short : look.label}
    </span>
  );
}

/**
 * An ongoing condition, said in one line.
 *
 * A prompt policy's event is a *span*, not a moment: the appliance keeps folding
 * re-sightings into the row that opened rather than raising a new alert every
 * cooldown. So the card has to say when it started, how long it has run and how
 * many times the model has said yes — otherwise "1 alert" and "the same mess for
 * three hours" look identical, which is the failure this replaces from the other
 * direction.
 *
 * Renders nothing at all for a detector event or a pre-migration-7 row: there is
 * no span to report and an empty "ongoing since —" would be worse than silence.
 */
export function ConditionLine({ event }: { event: ArgusEvent }) {
  const look = conditionLook(event);
  if (!look) return null;
  return (
    <div
      className={`condition ${look.ongoing ? 'condition--live' : ''}`}
      title={look.title}
    >
      <span className={`badge ${look.ongoing ? 'badge--ongoing' : 'badge--ended'}`}>
        {look.ongoing ? <span className="condition__dot" /> : null}
        {look.badge}
      </span>
      <span className="mono condition__line">{look.line}</span>
    </div>
  );
}

/**
 * A block the design specifies but this appliance cannot yet fill.
 *
 * Rendered rather than hidden, and empty rather than mocked: the screen a buyer
 * was shown is the screen they get, and it says plainly what turns it on. A
 * fabricated number here would be the one thing that makes every other number
 * on the page suspect.
 *
 * `phase` is the eyebrow: a roadmap phase where one is what is missing, or the
 * name of the missing piece where it is something more specific than that.
 */
export function NotEnabled({
  title,
  phase,
  children,
}: {
  title: string;
  phase: string;
  children: ReactNode;
}) {
  return (
    <div className="notice">
      <div className="notice__phase">{phase}</div>
      <div className="notice__title">{title}</div>
      <div className="notice__body">{children}</div>
    </div>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="empty">
      <div className="screen-title">{title}</div>
      {children ? <div style={{ color: 'var(--faint)', fontSize: 'var(--fs-meta)' }}>{children}</div> : null}
    </div>
  );
}

export function ErrorBar({ error }: { error: string | null }) {
  if (!error) return null;
  return <div className="error-bar">API ERROR · {error}</div>;
}
