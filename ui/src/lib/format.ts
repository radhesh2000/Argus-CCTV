/**
 * Formatting, and the design's severity/stage vocabulary.
 *
 * The design speaks in critical/warning/info and LIVE/PILOT/SHADOW; Argus
 * speaks in high/medium/low and shadow/reviewed/auto_routed. The marks (glyph,
 * colour, badge shape) are the design's and are kept exactly; the words are the
 * product's, because "PILOT" on an alert that is simply waiting for a human
 * would be a different claim from the one the staged rollout makes.
 */

import type { ArgusEvent, Severity, Stage } from '../api/types';

export interface SeverityLook {
  glyph: string;
  color: string;
  label: string;
}

const SEVERITY: Record<Severity, SeverityLook> = {
  high: { glyph: '▲', color: 'var(--crit)', label: 'Critical' },
  medium: { glyph: '◆', color: 'var(--warn)', label: 'Warning' },
  low: { glyph: '●', color: 'var(--info)', label: 'Info' },
};

export function severityLook(severity: Severity | string): SeverityLook {
  return SEVERITY[severity as Severity] ?? SEVERITY.low;
}

export interface StageLook {
  glyph: string;
  label: string;
  className: string;
  title: string;
}

const STAGE: Record<Stage, StageLook> = {
  auto_routed: {
    glyph: '●',
    label: 'ROUTED',
    className: 'badge badge--live',
    title: 'Auto-routed: confirmed violations notify their role without waiting for a reviewer',
  },
  reviewed: {
    glyph: '◐',
    label: 'REVIEW',
    className: 'badge badge--pilot',
    title: 'Human-reviewed: every alert waits for a verdict before anyone is notified',
  },
  shadow: {
    glyph: '○',
    label: 'SHADOW',
    className: 'badge badge--shadow',
    title: 'Shadow: logged silently, no notification, not counted towards precision',
  },
};

export function stageLook(stage: Stage | string): StageLook {
  return STAGE[stage as Stage] ?? STAGE.shadow;
}

/**
 * What raised this — the design's own vocabulary ("VLM PROMPT" / "FAST
 * DETECTOR", from the Policy Studio cards), used everywhere an event's origin
 * is shown so the feed, triage and analytics all name the same thing the same
 * way. A prompt policy's event is the model's sentence about a frame; a
 * detector's event is a count inside a polygon. A reviewer weighs them
 * differently and has to be able to tell which is in front of them.
 */
export interface OriginLook {
  label: string;
  short: string;
  className: string;
  title: string;
}

const ORIGIN: Record<'detector' | 'vlm', OriginLook> = {
  vlm: {
    label: 'VLM PROMPT',
    short: 'VLM',
    className: 'badge badge--vlm',
    title:
      'Raised by a plain-English prompt policy, judged frame by frame by the vision-language model on this box',
  },
  detector: {
    label: 'FAST DETECTOR',
    short: 'DET',
    className: 'badge badge--detector',
    title: 'Raised by the fast person detector — boxes, track ids and a zone count',
  },
};

export function originLook(type: string | null | undefined): OriginLook {
  return ORIGIN[type as 'detector' | 'vlm'] ?? ORIGIN.detector;
}

/**
 * The decoder's rung, in words an operator can act on.
 *
 * Decode is a ladder, not a flag: `nvdec` keeps the frame on the GPU until the
 * sampling filter has dropped four in five, `nvdec-download` copies every frame
 * at full rate and then throws most of them away, and `cpu` is software. A
 * camera sitting on the middle rung is working but paying about ten times the
 * host cost it needs to, which is worth saying out loud rather than rendering
 * as "NVDEC-DOWNLOAD DECODE".
 */
export function decodeLabel(mode: string | null | undefined): string | null {
  if (!mode) return null;
  if (mode === 'nvdec') return 'NVDEC decode';
  if (mode === 'nvdec-download') return 'NVDEC decode, full-rate copy';
  if (mode === 'cpu') return 'software decode';
  return `${mode} decode`;
}

export const LIFECYCLE_LABEL: Record<string, string> = {
  draft: 'DRAFT',
  pilot: 'PILOT',
  production: 'PRODUCTION',
  archived: 'ARCHIVED',
};

/** 24-hour clock, the way a plant log writes it. */
export function clockTime(value: number | Date): string {
  const date = typeof value === 'number' ? new Date(value * 1000) : value;
  return date.toLocaleTimeString('en-GB', { hour12: false });
}

export function shortTime(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString('en-GB', {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

export function dayAndTime(ts: number): string {
  const date = new Date(ts * 1000);
  const today = new Date();
  const sameDay = date.toDateString() === today.toDateString();
  const time = clockTime(date);
  return sameDay ? time : `${date.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' })} ${time}`;
}

export function mastheadDate(date: Date): string {
  return date
    .toLocaleDateString('en-GB', {
      weekday: 'short',
      month: 'short',
      day: '2-digit',
      year: 'numeric',
    })
    .toUpperCase()
    .replace(/,/g, '');
}

export function relativeTime(ts: number, now = Date.now() / 1000): string {
  const seconds = Math.max(0, now - ts);
  if (seconds < 60) return `${Math.round(seconds)} s ago`;
  const minutes = seconds / 60;
  if (minutes < 60) return `${Math.round(minutes)} min ago`;
  const hours = minutes / 60;
  if (hours < 24) return `${Math.round(hours)} h ago`;
  const days = hours / 24;
  return `${Math.round(days)} d ago`;
}

export function duration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return '—';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60);
  if (minutes < 60) return `${minutes}m ${String(rest).padStart(2, '0')}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${String(minutes % 60).padStart(2, '0')}m`;
}

export function percent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return `${(value * 100).toFixed(digits)}%`;
}

export function count(value: number | null | undefined): string {
  if (value === null || value === undefined) return '—';
  return value.toLocaleString('en-GB');
}

export function bytesFromMb(mb: number): string {
  if (mb >= 1000) return `${(mb / 1000).toFixed(1)} GB`;
  return `${Math.round(mb)} MB`;
}

/**
 * An event that is an ongoing condition rather than a moment.
 *
 * A prompt policy asks about a state — "is there loose laminate on the wrapper
 * apron" — and a state persists. Since schema migration 7 the appliance folds
 * every re-sighting into the row that opened, so one of these carries a start,
 * a last-seen, and a count of how many times the model said yes. Before that it
 * carried none of them and wrote a new row every cooldown instead: forty
 * near-identical cards for one offcut, overnight, on this appliance's own
 * feed.
 *
 * `null` for anything that is not one — every detector event, and every VLM
 * event raised before the migration. Those render exactly as they always did.
 */
export interface ConditionLook {
  ongoing: boolean;
  observations: number;
  /** Seconds from the first observation to the last. */
  spanSec: number;
  /** The one line the card shows. */
  line: string;
  /** Small-caps mark: ONGOING while it is still true, ENDED once it is not. */
  badge: string;
  title: string;
}

const CLOSE_REASON: Record<string, string> = {
  cleared: 'the model looked again and the condition was gone',
  stale: 'nobody observed it for long enough that the appliance stopped claiming it was ongoing — a camera or the model may have dropped, so this is not evidence it was cleared',
  moved: 'a triggered verdict landed somewhere else in the view and started its own condition',
};

export function conditionLook(event: ArgusEvent): ConditionLook | null {
  if (!event.condition_key) return null;
  const observations = event.observations ?? 1;
  const lastSeen = event.last_seen_ts ?? event.ts;
  const ongoing = event.condition_state === 'open';
  const spanSec = Math.max(0, (ongoing ? lastSeen : event.closed_at ?? lastSeen) - event.ts);
  // "N observations" is spelled out rather than abbreviated because the number
  // is the point of the card: it is the difference between "somebody dropped a
  // wrapper" and "this has been on the floor for three hours".
  const seen = `${observations} observation${observations === 1 ? '' : 's'}`;
  const line = ongoing
    ? `Ongoing since ${shortTime(event.ts)} · ${seen}${spanSec >= 60 ? ` · ${duration(spanSec)}` : ''}`
    : `${shortTime(event.ts)} → ${shortTime(event.closed_at ?? lastSeen)} · ${duration(spanSec)} · ${seen}`;
  const ended = event.close_reason ? CLOSE_REASON[event.close_reason] ?? event.close_reason : null;
  return {
    ongoing,
    observations,
    spanSec,
    line,
    badge: ongoing ? 'ONGOING' : 'ENDED',
    title: ongoing
      ? `Still true as of ${shortTime(lastSeen)}. The appliance has folded ${seen} into this one alert instead of raising ${observations} of them.`
      : `Ended${ended ? ` — ${ended}` : ''}.`,
  };
}

/** ISO day (`2026-08-09`) as the compact `AUG 09` the design's axes use. */
export function dayLabel(day: string): string {
  const [year, month, date] = day.split('-').map(Number);
  if (!year || !month || !date) return day;
  return new Date(year, month - 1, date)
    .toLocaleDateString('en-GB', { month: 'short', day: '2-digit' })
    .toUpperCase();
}

/** The event's headline, kept short enough for a card. Falls back to the
 *  description the engine wrote when a policy has been deleted since. */
export function eventTitle(
  detectorName: string,
  policyLabels: Record<string, string>,
  description: string,
): string {
  return policyLabels[detectorName] ?? description.split(':')[0] ?? detectorName;
}
