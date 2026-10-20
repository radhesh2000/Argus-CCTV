/**
 * Alert Feed — the reviewed record, filterable, with evidence on every card.
 *
 * The one interaction that matters here is what happens when an alert arrives
 * while somebody is reading. Nothing moves: new events collect behind a pill
 * ("↑ 3 NEW ALERTS"), and the list only changes when the reader says so. A feed
 * that reflows under the cursor is a feed that gets misclicked, and the misclick
 * is a safety verdict.
 *
 * **A card can change without the list changing.** Since schema migration 7 a
 * prompt policy's event is an ongoing *condition*: while the mess is still
 * there the appliance folds each re-sighting into the row that opened rather
 * than raising another alert. Those arrive as SSE `update` frames and are
 * merged straight over the row in place — same position, same order, a higher
 * observation count and a fresher picture. That is the opposite handling from
 * an arrival, and deliberately: an arrival is news the reader has not asked to
 * see yet, an update is the card in front of them becoming true again.
 */

import { useEffect, useMemo, useState } from 'react';
import { api } from '../api/client';
import type { EventFilters } from '../api/client';
import { evidence } from '../api/client';
import type { ArgusEvent, Severity, Stage, Verdict } from '../api/types';
import { EvidenceViewer } from '../components/EvidenceViewer';
import { ConditionLine, OriginBadge, SeverityGlyph, StageBadge, ErrorBar } from '../components/marks';
import { useResource } from '../lib/hooks';
import { dayAndTime, originLook, percent, severityLook, stageLook } from '../lib/format';
import { shiftWindowLabel } from '../lib/shifts';
import { useAppState } from '../state/AppState';

type Filter = {
  severity: Severity | null;
  stage: Stage | null;
  verdict: Verdict | 'unreviewed' | null;
  detector: string | null;
  /** Which engine raised it — the API filters this server-side. */
  origin: 'detector' | 'vlm' | null;
  shift: string | null;
};

const EMPTY: Filter = {
  severity: null,
  stage: null,
  verdict: null,
  detector: null,
  origin: null,
  shift: null,
};

const PAGE = 40;

export function AlertFeed() {
  const { policyLabels, config, arrivals, updates, drainArrivals, noteReview } = useAppState();
  const [filter, setFilter] = useState<Filter>(EMPTY);
  const [held, setHeld] = useState<ArgusEvent[]>([]);
  const [open, setOpen] = useState<ArgusEvent | null>(null);
  const [limit, setLimit] = useState(PAGE);

  const query: EventFilters = useMemo(
    () => ({
      severity: filter.severity,
      stage: filter.stage,
      verdict: filter.verdict,
      detector: filter.detector,
      detector_type: filter.origin,
      limit,
    }),
    [filter, limit],
  );

  const page = useResource(() => api.events(query), [query]);

  // Arrivals are parked, not merged: see the module docstring.
  useEffect(() => {
    if (arrivals.length > 0) setHeld(drainArrivals());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [arrivals.length]);

  const shifts = config?.site.shifts ?? [];

  const events = useMemo(() => {
    // Merged, not re-sorted, and not re-filtered. A live update is the same
    // row with a higher observation count and a fresher snapshot; moving it
    // would move every card below it under the reader's cursor, which is the
    // one thing this screen promises never to do.
    const rows = (page.data?.events ?? []).map((event) => updates[event.id] ?? event);
    if (!filter.shift) return rows;
    const shift = shifts.find((entry) => entry.id === filter.shift);
    if (!shift) return rows;
    return rows.filter((event) => {
      const hour = new Date(event.ts * 1000).getHours();
      return shift.start_hour <= shift.end_hour
        ? hour >= shift.start_hour && hour < shift.end_hour
        : hour >= shift.start_hour || hour < shift.end_hour;
    });
  }, [page.data, updates, filter.shift, shifts]);

  const showHeld = () => {
    setHeld([]);
    page.reload();
  };

  const counts = useMemo(() => {
    const rows = page.data?.events ?? [];
    const tally = (key: (event: ArgusEvent) => string) => {
      const map: Record<string, number> = {};
      for (const event of rows) map[key(event)] = (map[key(event)] ?? 0) + 1;
      return map;
    };
    return {
      severity: tally((event) => event.severity),
      stage: tally((event) => event.stage),
      detector: tally((event) => event.detector_name),
      origin: tally((event) => event.detector_type),
      verdict: tally((event) => event.review_verdict ?? 'unreviewed'),
    };
  }, [page.data]);

  const toggle = <K extends keyof Filter>(key: K, value: Filter[K]) => {
    setFilter((current) => ({ ...current, [key]: current[key] === value ? null : value }));
    setLimit(PAGE);
  };

  return (
    <div className="rise feed">
      <aside className="feed__filters">
        <div className="screen-title">Alert Feed</div>

        <FilterGroup label="Severity">
          {(['high', 'medium', 'low'] as Severity[]).map((severity) => (
            <FilterRow
              key={severity}
              on={filter.severity === severity}
              label={`${severityLook(severity).glyph} ${severityLook(severity).label}`}
              n={counts.severity[severity] ?? 0}
              onClick={() => toggle('severity', severity)}
            />
          ))}
        </FilterGroup>

        <FilterGroup label="Stage">
          {(['auto_routed', 'reviewed', 'shadow'] as Stage[]).map((stage) => (
            <FilterRow
              key={stage}
              on={filter.stage === stage}
              label={`${stageLook(stage).glyph} ${stageLook(stage).label}`}
              n={counts.stage[stage] ?? 0}
              onClick={() => toggle('stage', stage)}
            />
          ))}
        </FilterGroup>

        <FilterGroup label="Verdict">
          {([
            ['unreviewed', 'Awaiting review'],
            ['confirmed', 'Confirmed'],
            ['false_positive', 'False positive'],
          ] as [Verdict | 'unreviewed', string][]).map(([verdict, label]) => (
            <FilterRow
              key={verdict}
              on={filter.verdict === verdict}
              label={label}
              n={counts.verdict[verdict] ?? 0}
              onClick={() => toggle('verdict', verdict)}
            />
          ))}
        </FilterGroup>

        {/* Which engine raised it. Kept apart from "Type" because a plant runs
            both kinds against the same zone, and "show me only what the model
            claimed" is the question asked when the model is on trial. */}
        <FilterGroup label="Raised by">
          {(['detector', 'vlm'] as const).map((origin) => (
            <FilterRow
              key={origin}
              on={filter.origin === origin}
              label={originLook(origin).label}
              n={counts.origin[origin] ?? 0}
              onClick={() => toggle('origin', origin)}
            />
          ))}
        </FilterGroup>

        <FilterGroup label="Type">
          {(page.data?.detectors ?? []).map((detector) => (
            <FilterRow
              key={detector}
              on={filter.detector === detector}
              label={policyLabels[detector] ?? detector}
              n={counts.detector[detector] ?? 0}
              onClick={() => toggle('detector', detector)}
            />
          ))}
        </FilterGroup>

        {shifts.length > 0 ? (
          <FilterGroup label="Shift">
            {shifts.map((shift) => (
              <FilterRow
                key={shift.id}
                on={filter.shift === shift.id}
                label={`Shift ${shift.id} · ${shiftWindowLabel(shift)}`}
                n={null}
                onClick={() => toggle('shift', shift.id)}
              />
            ))}
          </FilterGroup>
        ) : null}
      </aside>

      <div className="feed__list">
        <ErrorBar error={page.error} />
        <div className="feed__column">
          {held.length > 0 ? (
            <button className="feed__pill slidein" onClick={showHeld}>
              ↑ {held.length} NEW ALERT{held.length === 1 ? '' : 'S'} — CLICK TO SHOW
            </button>
          ) : null}

          {events.length === 0 && !page.loading ? (
            <div className="notice">
              <div className="notice__title">No alerts match</div>
              <div className="notice__body">
                {page.data?.total === 0
                  ? 'This appliance has not raised an event yet. Start the engine with `argus run` and the feed fills as policies fire.'
                  : 'Nothing in the current filter. Clear a filter on the left to widen the search.'}
              </div>
            </div>
          ) : null}

          {events.map((event) => (
            <AlertCard
              key={event.id}
              event={event}
              title={policyLabels[event.detector_name] ?? event.detector_name}
              onOpen={() => setOpen(event)}
              onReviewed={() => {
                page.reload();
                noteReview();
              }}
            />
          ))}

          {page.data && page.data.total > events.length ? (
            <button
              className="button button--quiet"
              style={{ alignSelf: 'center' }}
              onClick={() => setLimit((value) => value + PAGE)}
            >
              Show more — {events.length} of {page.data.total}
            </button>
          ) : null}
        </div>
      </div>

      {open ? (
        <EvidenceViewer
          eventId={open.id}
          title={policyLabels[open.detector_name] ?? open.detector_name}
          onClose={() => setOpen(null)}
          onReviewed={() => {
            page.reload();
            noteReview();
          }}
        />
      ) : null}
    </div>
  );
}

function FilterGroup({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="eyebrow" style={{ marginBottom: 8 }}>
        {label}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>{children}</div>
    </div>
  );
}

function FilterRow({
  on,
  label,
  n,
  onClick,
}: {
  on: boolean;
  label: string;
  n: number | null;
  onClick: () => void;
}) {
  return (
    <button className={`filter ${on ? 'filter--on' : ''}`} onClick={onClick}>
      <span className="filter__box" style={{ background: on ? 'var(--accent)' : 'transparent' }} />
      <span style={{ flex: 1, textAlign: 'left' }}>{label}</span>
      {n !== null ? (
        <span className="mono" style={{ fontSize: 'var(--fs-note)', color: 'var(--faint)' }}>
          {n}
        </span>
      ) : null}
    </button>
  );
}

function AlertCard({
  event,
  title,
  onOpen,
  onReviewed,
}: {
  event: ArgusEvent;
  title: string;
  onOpen: () => void;
  onReviewed: () => void;
}) {
  const [busy, setBusy] = useState(false);
  // Versioned on `last_seen_ts`: an ongoing condition rewrites its snapshot in
  // place, so the URL is stable and only the cache key moves.
  const snapshot = evidence.snapshot(event.snapshot_path, event.last_seen_ts);
  const verdict = event.review_verdict ?? null;
  const fromVlm = event.detector_type === 'vlm';
  // The VLM verification pass held this alert back. The row and its evidence
  // survive on purpose; what must not survive is a routing line claiming the
  // alert went out.
  const held = Boolean(event.suppressed);

  const rule = async (value: Verdict) => {
    setBusy(true);
    try {
      await api.review(event.id, value, 'console');
      onReviewed();
    } finally {
      setBusy(false);
    }
  };

  const routing = held
    ? `Held by the verifier — ${event.suppressed_reason ?? 'no reason recorded'}${
        event.routed_to ? ` · not sent to ${event.routed_to}` : ''
      }`
    : verdict
      ? verdict === 'confirmed'
        ? `Confirmed${event.routed_to ? ` · routed → ${event.routed_to}` : ''}`
        : 'Ruled a false positive — feeds retraining'
      : event.stage === 'auto_routed'
        ? `Routed → ${event.routed_to ?? 'nobody configured'}`
        : event.stage === 'shadow'
          ? 'Logged silently — shadow stage, no notification'
          : 'Pending human review';

  return (
    <div className="alert">
      <button className="alert__thumb" onClick={onOpen} title="Open evidence">
        {snapshot ? (
          <img src={snapshot} alt="" loading="lazy" />
        ) : (
          <span className="mono" style={{ fontSize: 'var(--fs-label)', color: 'var(--on-video-faint)' }}>
            NO SNAPSHOT
          </span>
        )}
        {event.clip_path ? <span className="alert__clip mono">CLIP</span> : null}
      </button>

      <div className="alert__body">
        <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
          <SeverityGlyph severity={event.severity} />
          <span style={{ fontWeight: 600, fontSize: 'var(--fs-body)' }}>{title}</span>
          <OriginBadge type={event.detector_type} />
          <StageBadge stage={event.stage} />
        </div>
        <div className="mono alert__meta">
          {event.camera_id}
          {event.meta?.zone_name ? ` · ${event.meta.zone_name}` : ''} · {dayAndTime(event.ts)}
          <span style={{ color: 'var(--faint)' }}> · conf {percent(event.confidence, 0)}</span>
          {event.meta?.count !== undefined ? (
            <span style={{ color: 'var(--faint)' }}>
              {' '}
              · {event.meta.count} in zone (limit {event.meta.limit ?? '—'})
            </span>
          ) : null}
          {fromVlm && event.meta?.model ? (
            <span style={{ color: 'var(--faint)' }} title="the model that answered">
              {' '}
              · {event.meta.model}
            </span>
          ) : null}
        </div>
        {/* A prompt policy's event *is* a sentence the model wrote about this
            frame. Showing the policy label alone would attribute the claim to
            the appliance rather than to the model that made it. */}
        {fromVlm ? (
          <div className="alert__reason">“{event.meta?.reason ?? event.description}”</div>
        ) : null}
        {/* One card per condition, not per sighting. See ConditionLine. */}
        <ConditionLine event={event} />
        <div
          style={{
            fontSize: 'var(--fs-note)',
            color: held
              ? 'var(--warn)'
              : verdict === 'confirmed' || event.stage === 'auto_routed'
                ? 'var(--good)'
                : 'var(--dim)',
          }}
        >
          {routing}
        </div>
      </div>

      <div className="alert__actions">
        {verdict || event.stage === 'shadow' ? (
          <span className="mono" style={{ fontSize: 'var(--fs-note)', color: 'var(--faint)', textAlign: 'center' }}>
            {verdict ? (verdict === 'confirmed' ? '✓ CONFIRMED' : '✕ FALSE POSITIVE') : 'NOT REVIEWABLE'}
          </span>
        ) : (
          <>
            {/* Confirm is the only filled control on the card; false positive
                is its outlined peer. They were two soft fills a shade apart,
                on a surface where one of them files a safety incident. */}
            <button className="button" disabled={busy} onClick={() => rule('confirmed')}>
              Confirm
            </button>
            <button
              className="button button--soft"
              disabled={busy}
              onClick={() => rule('false_positive')}
            >
              False positive
            </button>
          </>
        )}
        <button className="button button--ghost" onClick={onOpen}>
          Open ↗
        </button>
      </div>
    </div>
  );
}
