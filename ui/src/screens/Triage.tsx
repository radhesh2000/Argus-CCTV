/**
 * Review Triage — keyboard-first, one alert at a time.
 *
 * C confirms, F marks a false positive, S skips. The verdicts are real writes
 * to /api/events/{id}/review, and the precision they produce is what gates a
 * policy's promotion to auto-routed, so the queue advances *optimistically*:
 * the next alert is on screen before the request lands, and a failure puts the
 * card back with the reason. Waiting for a round trip between keystrokes is
 * what makes a reviewer stop using the keyboard.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api, evidence } from '../api/client';
import type { ArgusEvent, Verdict } from '../api/types';
import { ConditionLine, OriginBadge, SeverityGlyph, StageBadge, ErrorBar } from '../components/marks';
import { TrackBoxes } from '../components/Overlay';
import { useKeydown, useResource } from '../lib/hooks';
import { dayAndTime, duration, originLook, percent, shortTime } from '../lib/format';
import { useAppState } from '../state/AppState';

interface Props {
  onQueueChanged: () => void;
}

export function Triage({ onQueueChanged }: Props) {
  const { policyLabels, updates, noteReview } = useAppState();
  const [order, setOrder] = useState<'oldest' | 'newest'>('oldest');
  const queue = useResource(() => api.triage({ order }), [order]);
  const [index, setIndex] = useState(0);
  const [ruled, setRuled] = useState<Record<number, Verdict>>({});
  const [cleared, setCleared] = useState(0);
  const [mode, setMode] = useState<'frame' | 'clip'>('frame');
  const [failure, setFailure] = useState<string | null>(null);
  const startedAt = useRef<number>(Date.now());
  const [times, setTimes] = useState<number[]>([]);

  // Merged in place, never re-sorted: an ongoing condition the reviewer is
  // looking at can gain observations while they read it, and the queue moving
  // under a keyboard-first screen is how the wrong alert gets a verdict.
  const all = useMemo(
    () => (queue.data?.events ?? []).map((event) => updates[event.id] ?? event),
    [queue.data, updates],
  );
  const backlog = queue.data?.total ?? 0;
  const pending = useMemo(() => all.filter((event) => !(event.id in ruled)), [all, ruled]);
  const current = pending[Math.min(index, Math.max(0, pending.length - 1))] ?? null;

  useEffect(() => {
    startedAt.current = Date.now();
    setMode('frame');
  }, [current?.id]);

  const rule = useCallback(
    (verdict: Verdict) => {
      if (!current) return;
      const event = current;
      const seconds = (Date.now() - startedAt.current) / 1000;
      setRuled((current_) => ({ ...current_, [event.id]: verdict }));
      setCleared((value) => value + 1);
      setTimes((values) => [...values, seconds]);
      setIndex(0);
      void api
        .review(event.id, verdict, 'console')
        .then(() => {
          noteReview();
          onQueueChanged();
        })
        .catch((error: unknown) => {
          // Put it back: an unrecorded verdict must not look recorded.
          setRuled((current_) => {
            const next = { ...current_ };
            delete next[event.id];
            return next;
          });
          setCleared((value) => Math.max(0, value - 1));
          setFailure(
            `Verdict for #${event.id} was not saved (${error instanceof Error ? error.message : String(error)})`,
          );
        });
    },
    [current, noteReview, onQueueChanged],
  );

  useKeydown((event) => {
    if (
      event.target instanceof HTMLElement &&
      ['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target.tagName)
    ) {
      return;
    }
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    const key = event.key.toLowerCase();
    if (key === 'c') {
      event.preventDefault();
      rule('confirmed');
    } else if (key === 'f') {
      event.preventDefault();
      rule('false_positive');
    } else if (key === 's') {
      event.preventDefault();
      setIndex((value) => (pending.length ? (value + 1) % pending.length : 0));
    }
  });

  const median = useMemo(() => {
    if (times.length === 0) return null;
    const sorted = [...times].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle]! : ((sorted[middle - 1]! + sorted[middle]!) / 2);
  }, [times]);

  return (
    <div className="rise triage">
      <div className="triage__strip">
        {/*
          The backlog, not the page. A queue is drained from the front, so
          oldest stays the default — but on an appliance nobody has kept up
          with, the front is a fortnight old and today's alert is thousands of
          rows out of reach. `NEWEST` is how a reviewer gets to the shift they
          just worked; the count says how much is behind them either way.
        */}
        <div className="mono" style={{ fontSize: 'var(--fs-note)', color: 'var(--faint)', textAlign: 'center' }}>
          QUEUE {pending.length ? Math.min(index + 1, pending.length) : 0}/{pending.length}
          {backlog > pending.length ? ` · ${backlog.toLocaleString()} WAITING` : null}
        </div>
        <div className="triage__order">
          {(['oldest', 'newest'] as const).map((option) => (
            <button
              key={option}
              className={`triage__orderbtn ${order === option ? 'triage__orderbtn--on' : ''}`}
              onClick={() => {
                setOrder(option);
                setIndex(0);
              }}
            >
              {option.toUpperCase()}
            </button>
          ))}
        </div>
        {all.map((event) => {
          const verdict = ruled[event.id];
          const position = pending.findIndex((entry) => entry.id === event.id);
          const isCurrent = current?.id === event.id;
          const snapshot = evidence.snapshot(event.snapshot_path, event.last_seen_ts);
          return (
            <button
              key={event.id}
              className={`triage__thumb ${isCurrent ? 'triage__thumb--on' : ''}`}
              style={{ opacity: verdict ? 0.38 : 1 }}
              onClick={() => position >= 0 && setIndex(position)}
            >
              <div className="triage__thumbimg">
                {snapshot ? <img src={snapshot} alt="" loading="lazy" /> : null}
              </div>
              <div className="triage__thumbmeta">
                <SeverityGlyph severity={event.severity} size={8} />
                <span className="mono" style={{ fontSize: 'var(--fs-label)', color: 'var(--dim)' }}>
                  {shortTime(event.ts)}
                </span>
                <span style={{ marginLeft: 'auto', fontSize: 'var(--fs-label)' }}>
                  {verdict === 'confirmed' ? '✓' : verdict === 'false_positive' ? '✕' : ''}
                </span>
              </div>
            </button>
          );
        })}
      </div>

      {current ? (
        <>
          <div className="triage__main">
            <ErrorBar error={failure ?? queue.error} />
            <div className="triage__head">
              <SeverityGlyph severity={current.severity} size={11} />
              <span className="screen-title">
                {policyLabels[current.detector_name] ?? current.detector_name}
              </span>
              <OriginBadge type={current.detector_type} />
              <StageBadge stage={current.stage} />
              <span
                className="mono"
                style={{ marginLeft: 'auto', fontSize: 'var(--fs-note)', color: 'var(--dim)' }}
              >
                #{current.id} · {current.camera_id} · {dayAndTime(current.ts)}
              </span>
            </div>

            <div className="triage__stage">
              <div className="evidence__toggle">
                <button
                  className={`media-toggle ${mode === 'frame' ? 'media-toggle--on' : ''}`}
                  onClick={() => setMode('frame')}
                  disabled={!current.snapshot_path}
                >
                  FRAME
                </button>
                <button
                  className={`media-toggle ${mode === 'clip' ? 'media-toggle--on' : ''}`}
                  onClick={() => setMode('clip')}
                  disabled={!current.clip_path}
                  title={current.clip_path ? 'Evidence clip' : 'No clip on disk for this event'}
                >
                  CLIP
                </button>
              </div>
              {mode === 'clip' && current.clip_path ? (
                <video src={evidence.clip(current.clip_path)!} controls autoPlay muted playsInline />
              ) : (
                /* The overlay goes over the snapshot on the same terms as the
                   Evidence Viewer's, so the two screens do not disagree about
                   what a detection looks like. It is safe to draw it over an
                   older event whose boxes are burned into the JPEG: the quiet
                   marks land on the burned-in ones rather than beside them,
                   and carry no label of their own to stack on top of the
                   burned-in tags. Events written from today on have a clean
                   frame and only this overlay. */
                <div className="triage__frame">
                  {current.snapshot_path ? (
                    <img
                      src={evidence.snapshot(current.snapshot_path, current.last_seen_ts)!}
                      alt="Violation frame"
                    />
                  ) : (
                    <div className="tile__signal">
                      <span className="mono" style={{ fontSize: 'var(--fs-note)' }}>
                        NO SNAPSHOT ON DISK
                      </span>
                    </div>
                  )}
                  {current.meta?.boxes?.length ? <TrackBoxes boxes={current.meta.boxes} /> : null}
                </div>
              )}
            </div>

            <div className="triage__verdicts">
              <button className="verdict verdict--confirm" onClick={() => rule('confirmed')}>
                Confirm{current.routed_to ? ` — route to ${current.routed_to}` : ''}
                <span className="key-hint">C</span>
              </button>
              <button className="verdict verdict--false" onClick={() => rule('false_positive')}>
                False positive — feeds retraining
                <span className="key-hint">F</span>
              </button>
              <button
                className="verdict verdict--skip"
                onClick={() => setIndex((value) => (pending.length ? (value + 1) % pending.length : 0))}
              >
                Skip
                <span className="key-hint">S</span>
              </button>
            </div>
          </div>

          <aside className="triage__side">
            <div>
              <div className="eyebrow" style={{ marginBottom: 8 }}>
                Policy fired
              </div>
              <div style={{ fontSize: 'var(--fs-body)', fontWeight: 600 }}>
                {policyLabels[current.detector_name] ?? current.detector_name}
              </div>
              <div className="mono" style={{ fontSize: 'var(--fs-note)', color: 'var(--dim)', marginTop: 5 }}>
                confidence {percent(current.confidence, 0)} ·{' '}
                <span title={originLook(current.detector_type).title}>
                  {originLook(current.detector_type).label}
                </span>
              </div>
            </div>

            {/* How long this has been true, and how many times the model has
                said so. On a condition that has run for hours the count is the
                thing a reviewer needs before ruling — one sighting and forty
                are not the same claim. */}
            <ConditionLine event={current} />

            <Grounds event={current} />

            <div>
              <div className="eyebrow" style={{ marginBottom: 8 }}>
                {current.detector_type === 'vlm' ? 'What the model answered' : 'What the engine wrote'}
              </div>
              <div style={{ fontSize: 'var(--fs-meta)', color: 'var(--dim)', lineHeight: 1.6 }}>
                {current.description}
              </div>
            </div>

            {current.suppressed ? (
              <div>
                <div className="eyebrow" style={{ marginBottom: 8 }}>
                  Held by the verifier
                </div>
                <div style={{ fontSize: 'var(--fs-meta)', color: 'var(--warn)', lineHeight: 1.6 }}>
                  {current.suppressed_reason ?? 'no reason recorded'}
                </div>
                <div className="mono" style={{ fontSize: 'var(--fs-note)', color: 'var(--faint)', marginTop: 5 }}>
                  {current.verified_by} · {percent(current.verify_confidence, 0)} · alert not sent
                </div>
              </div>
            ) : null}

            <CameraHistory cameraId={current.camera_id} policyLabels={policyLabels} />

            <div
              className="mono triage__stats"
              style={{ marginTop: 'auto', fontSize: 'var(--fs-note)', color: 'var(--faint)', lineHeight: 1.9 }}
            >
              {cleared} CLEARED THIS SESSION
              <br />
              {median !== null ? `MEDIAN ${duration(median)} / ALERT` : 'MEDIAN — / ALERT'}
            </div>
          </aside>
        </>
      ) : (
        <div className="empty" style={{ flex: 1 }}>
          <div style={{ fontSize: 'var(--fs-vital)', color: 'var(--good-dot)' }}>✓</div>
          <div className="screen-title">Queue clear</div>
          <div style={{ color: 'var(--faint)', fontSize: 'var(--fs-meta)' }}>
            {cleared > 0
              ? `${cleared} alert${cleared === 1 ? '' : 's'} reviewed this session.`
              : 'Nothing is waiting on a human verdict.'}
          </div>
          <button className="button button--quiet" onClick={queue.reload}>
            Check again
          </button>
        </div>
      )}
    </div>
  );
}

/**
 * What the claim rests on, which is a different thing for each engine.
 *
 * A detector event stands on a zone, a limit and a count. A prompt-policy event
 * stands on the question that was asked — so showing "Zone — · limit — · —
 * present at capture" for one, as the screen did before, is three dashes where
 * the actual grounds belong.
 */
function Grounds({ event }: { event: ArgusEvent }) {
  if (event.detector_type === 'vlm') {
    return (
      <div>
        <div className="eyebrow" style={{ marginBottom: 8 }}>
          Prompt asked of this frame
        </div>
        <div className="studio__prompt" style={{ fontSize: 'var(--fs-note)' }}>
          {event.meta?.prompt ?? 'not recorded with this event'}
        </div>
        <div className="mono" style={{ fontSize: 'var(--fs-note)', color: 'var(--faint)', marginTop: 6 }}>
          {event.meta?.model ?? 'unknown model'}
          {event.meta?.latency_sec != null ? ` · ${event.meta.latency_sec.toFixed(2)} s` : ''}
          {event.meta?.box_2d ? ' · answer grounded to a box' : ''}
        </div>
      </div>
    );
  }
  return (
    <div>
      <div className="eyebrow" style={{ marginBottom: 8 }}>
        Zone
      </div>
      <div style={{ fontSize: 'var(--fs-meta)', color: 'var(--dim)' }}>
        {event.meta?.zone_name ?? event.zone_id ?? '—'} · limit {event.meta?.limit ?? '—'} ·{' '}
        {event.meta?.count ?? '—'} present at capture
      </div>
    </div>
  );
}

function CameraHistory({
  cameraId,
  policyLabels,
}: {
  cameraId: string;
  policyLabels: Record<string, string>;
}) {
  const history = useResource(
    () => api.events({ camera_id: cameraId, limit: 6 }),
    [cameraId],
  );
  return (
    <div>
      <div className="eyebrow" style={{ marginBottom: 8 }}>
        This camera · recent
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
        {(history.data?.events ?? []).map((event) => (
          <div key={event.id} className="focus__event">
            <SeverityGlyph severity={event.severity} size={8} />
            <span
              className="mono"
              style={{ fontSize: 'var(--fs-label)', color: 'var(--faint)' }}
              title={originLook(event.detector_type).title}
            >
              {originLook(event.detector_type).short}
            </span>
            <span style={{ flex: 1, color: 'var(--dim)' }}>
              {policyLabels[event.detector_name] ?? event.detector_name}
            </span>
            <span className="mono" style={{ fontSize: 'var(--fs-note)', color: 'var(--faint)' }}>
              {dayAndTime(event.ts)}
            </span>
          </div>
        ))}
        {history.data?.events.length === 0 ? (
          <span style={{ fontSize: 'var(--fs-note)', color: 'var(--faint)' }}>No other events.</span>
        ) : null}
      </div>
    </div>
  );
}
