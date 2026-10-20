/**
 * One event, with its evidence.
 *
 * FRAME is the annotated snapshot the engine wrote at the moment of the
 * violation; CLIP is the ±8 s of real footage around it, served with Range
 * support so scrubbing works. The toggle is disabled rather than hidden when a
 * clip does not exist (yet, or any more — clips are kept 14 days, events 180),
 * and says which of those it is.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { api, evidence } from '../api/client';
import type { ArgusEvent, BoxCorrection, Camera, Notification, Review, Zone } from '../api/types';
import { useResource } from '../lib/hooks';
import { dayAndTime, duration, originLook, percent, stageLook } from '../lib/format';
import { DetectionPanel } from './DetectionPanel';
import { ConditionLine, OriginBadge, SeverityGlyph, StageBadge } from './marks';
import { TrackBoxes } from './Overlay';

interface Props {
  eventId: number;
  title: string;
  onClose: () => void;
  onReviewed?: () => void;
}

export function EvidenceViewer({ eventId, title, onClose, onReviewed }: Props) {
  const detail = useResource(() => api.event(eventId), [eventId]);
  const [mode, setMode] = useState<'frame' | 'clip'>('frame');
  const [busy, setBusy] = useState(false);

  // Per-box review state. Reset whenever the viewer is pointed at a different
  // event — otherwise a box hidden or highlighted on event #40 would still
  // look that way when a reviewer opens event #41 a moment later.
  const [hoveredTrackId, setHoveredTrackId] = useState<number | null>(null);
  const [selectedTrackId, setSelectedTrackId] = useState<number | null>(null);
  const [hiddenTrackIds, setHiddenTrackIds] = useState<Set<number>>(new Set());
  const [boxCorrections, setBoxCorrections] = useState<BoxCorrection[]>([]);
  // The box overlay has to land on the same pixels `object-fit: contain`
  // draws the frame on — and the stage's own box does not reliably have the
  // frame's aspect ratio, even with the `aspectRatio` style below: this stage
  // also carries a `max-height` (see components.css), and a block box with a
  // stretching width plus a clamped height does not shrink its width back to
  // match the ratio once that clamp is the binding constraint. When that
  // happens the stage stays wider than the frame, `contain` correctly shrinks
  // the *image* into the extra space, and an overlay sized to 100% of the
  // *stage* then places every box somewhere in that leftover space instead of
  // on the frame — a track can end up rendered entirely off the visible
  // picture. Trusting CSS to make the container match the image is what broke
  // here, so the fix instead measures the image's real rendered rectangle
  // (via ResizeObserver, since letterboxing changes on any resize) and
  // positions the overlay directly against it, in pixels — correct
  // regardless of what shape the surrounding box ends up being.
  const [frameSize, setFrameSize] = useState<{ w: number; h: number } | null>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const [stageSize, setStageSize] = useState<{ w: number; h: number } | null>(null);
  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      const { width, height } = entry.contentRect;
      setStageSize({ w: width, h: height });
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  // The rectangle `object-fit: contain` actually draws the frame into, in
  // pixels relative to the stage — whichever axis has slack (the container is
  // relatively wider or taller than the frame) gets centered, the other fills
  // exactly. This is the same math `object-fit: contain` itself runs; doing
  // it again here is what lets the overlay agree with it even when the
  // stage's own shape does not match the frame's.
  const imageRect = useMemo(() => {
    if (!frameSize || !stageSize || stageSize.w <= 0 || stageSize.h <= 0) return null;
    const containerRatio = stageSize.w / stageSize.h;
    const imageRatio = frameSize.w / frameSize.h;
    const width = imageRatio > containerRatio ? stageSize.w : stageSize.h * imageRatio;
    const height = imageRatio > containerRatio ? stageSize.w / imageRatio : stageSize.h;
    return { left: (stageSize.w - width) / 2, top: (stageSize.h - height) / 2, width, height };
  }, [frameSize, stageSize]);
  useEffect(() => {
    setHoveredTrackId(null);
    setSelectedTrackId(null);
    setHiddenTrackIds(new Set());
    setFrameSize(null);
    // Not a reset to null: this effect fires while the DOM still shows the
    // *previous* event (useResource only swaps `data` once the new fetch
    // resolves), so stageRef is a real, already-mounted node whose current
    // size is accurate right now. Re-reading it synchronously means
    // stageSize is never null for a tick, and ResizeObserver still catches
    // any later change once the new image actually resizes the stage.
    const el = stageRef.current;
    setStageSize(el ? { w: el.clientWidth, h: el.clientHeight } : null);
  }, [eventId]);
  useEffect(() => {
    setBoxCorrections(detail.data?.box_corrections ?? []);
  }, [detail.data]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const event = detail.data?.event;
  const review = detail.data?.review ?? null;
  const boxes = useMemo(() => event?.meta?.boxes ?? [], [event]);
  // Versioned: an ongoing condition rewrites this file in place each time the
  // model sees the mess again, so the path is stable and the cache key is not.
  const snapshot = evidence.snapshot(event?.snapshot_path ?? null, event?.last_seen_ts);
  const clip = evidence.clip(event?.clip_path ?? null);

  const submit = async (verdict: 'confirmed' | 'false_positive') => {
    if (!event) return;
    setBusy(true);
    try {
      await api.review(event.id, verdict, 'console');
      detail.reload();
      onReviewed?.();
    } finally {
      setBusy(false);
    }
  };

  const toggleVisibility = (trackId: number) => {
    setHiddenTrackIds((current) => {
      const next = new Set(current);
      if (next.has(trackId)) next.delete(trackId);
      else next.add(trackId);
      return next;
    });
  };

  return (
    <div className="scrim" style={{ alignItems: 'center' }} onClick={onClose}>
      <div className="evidence" onClick={(bubble) => bubble.stopPropagation()} role="dialog">
        <div className="evidence__head">
          {event ? <SeverityGlyph severity={event.severity} size={11} /> : null}
          <span className="screen-title" style={{ fontSize: 'var(--fs-figure)' }}>
            {title}
          </span>
          {event ? <OriginBadge type={event.detector_type} /> : null}
          {event ? <StageBadge stage={event.stage} /> : null}
          <span className="mono" style={{ marginLeft: 'auto', fontSize: 'var(--fs-note)', color: 'var(--dim)' }}>
            {event ? `#${event.id} · ${event.camera_id} · ${dayAndTime(event.ts)}` : 'loading…'}
          </span>
          <button className="chip-button" onClick={onClose}>
            ESC · CLOSE
          </button>
        </div>

        <div
          ref={stageRef}
          className="evidence__stage"
          style={mode === 'frame' && frameSize ? { aspectRatio: `${frameSize.w} / ${frameSize.h}` } : undefined}
        >
          <div className="evidence__toggle">
            <button
              className={`media-toggle ${mode === 'frame' ? 'media-toggle--on' : ''}`}
              onClick={() => setMode('frame')}
              disabled={!snapshot}
            >
              FRAME
            </button>
            <button
              className={`media-toggle ${mode === 'clip' ? 'media-toggle--on' : ''}`}
              onClick={() => setMode('clip')}
              disabled={!clip}
              title={clip ? 'Evidence clip' : 'No clip on disk for this event'}
            >
              CLIP
            </button>
          </div>
          {mode === 'clip' && clip ? (
            <video src={clip} controls autoPlay muted playsInline />
          ) : snapshot || boxes.length > 0 ? (
            <div style={{ position: 'relative', width: '100%', height: '100%' }}>
              {snapshot ? (
                <img
                  src={snapshot}
                  alt="Frame at the moment of the violation"
                  onLoad={(loaded) =>
                    setFrameSize({
                      w: loaded.currentTarget.naturalWidth,
                      h: loaded.currentTarget.naturalHeight,
                    })
                  }
                />
              ) : (
                <div className="tile__signal">
                  <span className="mono" style={{ fontSize: 'var(--fs-note)' }}>
                    NO SNAPSHOT ON DISK
                  </span>
                </div>
              )}
              {mode === 'frame' && boxes.length > 0 && imageRect ? (
                <div
                  style={{
                    position: 'absolute',
                    left: imageRect.left,
                    top: imageRect.top,
                    width: imageRect.width,
                    height: imageRect.height,
                  }}
                >
                  <TrackBoxes
                    boxes={boxes}
                    interactive
                    hoveredTrackId={hoveredTrackId}
                    selectedTrackId={selectedTrackId}
                    hiddenTrackIds={hiddenTrackIds}
                    onHoverTrack={setHoveredTrackId}
                    onSelectTrack={setSelectedTrackId}
                  />
                </div>
              ) : null}
            </div>
          ) : (
            <div className="tile__signal">
              <span className="mono" style={{ fontSize: 'var(--fs-note)' }}>
                NO SNAPSHOT ON DISK
              </span>
            </div>
          )}
        </div>

        <div className="evidence__body">
          {event ? <EvidenceFacts event={event} zone={detail.data?.zone ?? null} camera={detail.data?.camera ?? null} /> : null}
          {event && boxes.length > 0 ? (
            <DetectionPanel
              eventId={event.id}
              boxes={boxes}
              inside={event.meta?.inside}
              corrections={boxCorrections}
              hoveredTrackId={hoveredTrackId}
              selectedTrackId={selectedTrackId}
              hiddenTrackIds={hiddenTrackIds}
              onHoverTrack={setHoveredTrackId}
              onSelectTrack={setSelectedTrackId}
              onToggleVisibility={toggleVisibility}
              onCorrected={setBoxCorrections}
              canCorrect={event.stage !== 'shadow'}
            />
          ) : null}
          {detail.data?.notifications?.length ? (
            <NotificationLog notifications={detail.data.notifications} />
          ) : null}
          {event ? (
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              {review ? (
                <span style={{ fontSize: 'var(--fs-meta)', color: 'var(--dim)' }}>
                  Ruled <strong>{review.verdict === 'confirmed' ? 'confirmed' : 'false positive'}</strong>
                  {review.reviewer ? ` by ${review.reviewer}` : ''} ·{' '}
                  {new Date(review.created_at).toLocaleString('en-GB')}
                </span>
              ) : event.stage === 'shadow' ? (
                <span style={{ fontSize: 'var(--fs-meta)', color: 'var(--faint)' }}>
                  {stageLook('shadow').title}
                </span>
              ) : (
                <>
                  <button className="button button--soft" disabled={busy} onClick={() => submit('confirmed')}>
                    Confirm
                  </button>
                  <button
                    className="button button--quiet"
                    disabled={busy}
                    onClick={() => submit('false_positive')}
                  >
                    False positive
                  </button>
                </>
              )}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

function EvidenceFacts({
  event,
  zone,
  camera,
}: {
  event: ArgusEvent;
  zone: Zone | null;
  camera: Camera | null;
}) {
  const camera_label = camera ? `${camera.name} (${camera.id})` : event.camera_id;
  const routed = event.routed_to ?? 'nobody (stage below auto-routed)';
  // A prompt-policy event has no zone and no headcount; printing the detector's
  // fact sheet for one produces a column of dashes instead of the grounds the
  // claim actually rests on.
  const rows: [string, string][] =
    event.detector_type === 'vlm'
      ? [
          ['Camera', camera_label],
          ['Raised by', originLook('vlm').label],
          ['Model', event.meta?.model ?? 'unknown'],
          ['Confidence', percent(event.confidence, 0)],
          [
            'Inference',
            event.meta?.latency_sec != null ? `${event.meta.latency_sec.toFixed(2)} s` : '—',
          ],
          ['Routed to', routed],
        ]
      : [
          ['Camera', camera_label],
          ['Zone', zone ? `${zone.name} · limit ${zone.sanctioned_limit ?? '—'}` : '—'],
          ['Detected', `${event.meta?.count ?? '—'} people at capture`],
          ['Confidence', percent(event.confidence, 0)],
          ['Detector', event.meta?.detector ?? event.detector_name],
          ['Routed to', routed],
        ];
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px 18px' }}>
      {rows.map(([label, value]) => (
        <div key={label}>
          <div className="eyebrow">{label}</div>
          <div style={{ fontSize: 'var(--fs-meta)', color: 'var(--text2)', marginTop: 3 }}>{value}</div>
        </div>
      ))}
      <div style={{ gridColumn: '1 / -1', fontSize: 'var(--fs-meta)', color: 'var(--dim)', lineHeight: 1.6 }}>
        {event.description}
      </div>
      {/* The span, when this event is an ongoing condition rather than a
          moment. `description` above is the *newest* sentence the model wrote;
          the opening one is kept in meta and shown beneath, because "what did
          it look like when this started" is the first question asked of
          something that has been true for three hours. */}
      {event.condition_key ? (
        <div style={{ gridColumn: '1 / -1' }}>
          <ConditionLine event={event} />
          {event.meta?.opened?.reason && event.meta.opened.reason !== event.description ? (
            <div style={{ fontSize: 'var(--fs-note)', color: 'var(--faint)', lineHeight: 1.6, marginTop: 6 }}>
              When it opened: “{event.meta.opened.reason}”
            </div>
          ) : null}
        </div>
      ) : null}
      {event.detector_type === 'vlm' && event.meta?.prompt ? (
        <div style={{ gridColumn: '1 / -1' }}>
          <div className="eyebrow" style={{ marginBottom: 5 }}>
            Prompt asked of this frame
          </div>
          <div className="studio__prompt" style={{ fontSize: 'var(--fs-note)' }}>
            {event.meta.prompt}
          </div>
        </div>
      ) : null}
      {event.suppressed ? (
        <div style={{ gridColumn: '1 / -1' }}>
          <div className="eyebrow" style={{ marginBottom: 5 }}>
            Held by the verifier — this alert was not sent
          </div>
          <div style={{ fontSize: 'var(--fs-meta)', color: 'var(--warn)', lineHeight: 1.6 }}>
            {event.suppressed_reason ?? 'no reason recorded'}
            <span className="mono" style={{ fontSize: 'var(--fs-note)', color: 'var(--faint)' }}>
              {' '}
              · {event.verified_by} · {percent(event.verify_confidence, 0)}
            </span>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function NotificationLog({ notifications }: { notifications: Notification[] }) {
  return (
    <div>
      <div className="eyebrow" style={{ marginBottom: 6 }}>
        Notifications
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
        {notifications.map((entry) => (
          <div key={entry.id} className="mono" style={{ fontSize: 'var(--fs-note)', color: 'var(--dim)' }}>
            <span style={{ color: entry.status === 'sent' ? 'var(--good)' : 'var(--crit)' }}>
              {entry.status.toUpperCase()}
            </span>{' '}
            {entry.channel} → {entry.target} · {new Date(entry.created_at).toLocaleString('en-GB')}
            {entry.detail ? ` · ${entry.detail}` : ''}
          </div>
        ))}
      </div>
    </div>
  );
}

export function clipLengthLabel(padSeconds: number | undefined): string {
  return duration(padSeconds ? padSeconds * 2 : null);
}

export type { Review };
