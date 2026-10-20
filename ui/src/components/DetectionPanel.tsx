/**
 * The list half of the evidence viewer's canvas-plus-list split.
 *
 * One row per detection in `event.meta.boxes`, synced bidirectionally with
 * the interactive `TrackBoxes` overlay drawn on the same frame: hovering a
 * row highlights its box (and raises it above the rest of a crowd), hovering
 * a box highlights its row. A row that has already been corrected shows the
 * verdict instead of the flag buttons — a correction is a fact about that
 * box now, not an open question.
 *
 * This is the piece that turns "an alert with boxes overlapping into an
 * unreadable smear" into "here are the N things it found, one at a time" —
 * and, because a false-positive flag here is a structured record scoped to
 * one detection rather than the whole event, it is also what makes a
 * reviewer's correction usable as labelled training data later.
 *
 * The rows are named and ordered by `frameOrdinals`, the same helper the
 * overlay draws its labels from, so "#4" in this list and "#4" on the frame
 * are the same person and the list reads down the frame in the order a person
 * scans it. The tracker's own id stays on the row, dimmed, because the
 * correction posted from here is stored against it.
 */

import { useState } from 'react';
import { api } from '../api/client';
import type { Box, BoxCorrection, BoxVerdict } from '../api/types';
import { percent } from '../lib/format';
import { inReadingOrder, trackTag } from './Overlay';

interface Props {
  eventId: number;
  boxes: Box[];
  inside?: boolean[];
  corrections: BoxCorrection[];
  hoveredTrackId: number | null;
  selectedTrackId: number | null;
  hiddenTrackIds: ReadonlySet<number>;
  onHoverTrack: (trackId: number | null) => void;
  onSelectTrack: (trackId: number | null) => void;
  onToggleVisibility: (trackId: number) => void;
  onCorrected: (corrections: BoxCorrection[]) => void;
  canCorrect: boolean;
}

export function DetectionPanel({
  eventId,
  boxes,
  inside,
  corrections,
  hoveredTrackId,
  selectedTrackId,
  hiddenTrackIds,
  onHoverTrack,
  onSelectTrack,
  onToggleVisibility,
  onCorrected,
  canCorrect,
}: Props) {
  const [busyTrackId, setBusyTrackId] = useState<number | null>(null);
  const byTrack = new Map(corrections.map((correction) => [correction.track_id, correction]));
  // Reading order, not detector order: a list that jumps around the frame is
  // a list you have to hunt in.
  const rows = inReadingOrder(boxes);
  // One row is ever highlighted, and it is the same one the overlay promotes.
  const activeTrackId = hoveredTrackId ?? selectedTrackId;

  const correct = async (trackId: number, verdict: BoxVerdict) => {
    setBusyTrackId(trackId);
    try {
      const result = await api.correctBox(eventId, trackId, verdict);
      onCorrected(result.box_corrections);
    } finally {
      setBusyTrackId(null);
    }
  };

  if (boxes.length === 0) {
    return (
      <div style={{ fontSize: 'var(--fs-meta)', color: 'var(--dim)' }}>
        Nothing was detected in this frame.
      </div>
    );
  }

  return (
    <div>
      <div className="eyebrow" style={{ marginBottom: 7 }}>
        Detections ({boxes.length})
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        {rows.map(({ box, index, ordinal }) => {
          const [, , , , trackId, confidence] = box;
          const correction = byTrack.get(trackId);
          const hidden = hiddenTrackIds.has(trackId);
          const active = trackId === activeTrackId;
          const isInside = inside?.[index];
          return (
            <div
              key={`${trackId}-${index}`}
              onMouseEnter={() => onHoverTrack(trackId)}
              onMouseLeave={() => onHoverTrack(null)}
              onClick={() => onSelectTrack(trackId === selectedTrackId ? null : trackId)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                padding: '5px 7px',
                borderRadius: 4,
                cursor: 'pointer',
                opacity: hidden ? 0.45 : 1,
                background: active ? 'rgba(227,161,60,.14)' : 'transparent',
                border: `1px solid ${active ? 'rgba(227,161,60,.5)' : 'transparent'}`,
              }}
            >
              <button
                title={hidden ? 'Show on frame' : 'Hide from frame'}
                onClick={(event) => {
                  event.stopPropagation();
                  onToggleVisibility(trackId);
                }}
                className="chip-button"
                style={{ padding: '2px 6px', fontSize: 'var(--fs-label)' }}
              >
                {hidden ? 'HIDDEN' : 'SHOWN'}
              </button>

              <span className="mono" style={{ fontSize: 'var(--fs-meta)', minWidth: 30 }}>
                #{ordinal}
              </span>
              <span className="mono" style={{ fontSize: 'var(--fs-note)', color: 'var(--dim)' }}>
                {percent(confidence, 0)}
              </span>
              <span
                className="mono"
                title="The tracker's own id — corrections are recorded against it"
                style={{ fontSize: 'var(--fs-label)', color: 'var(--faint)' }}
              >
                {trackId >= 0 ? trackTag(trackId) : 'untracked'}
              </span>
              {isInside !== undefined ? (
                <span
                  className="mono"
                  style={{
                    fontSize: 'var(--fs-label)',
                    color: isInside ? 'var(--warn)' : 'var(--faint)',
                  }}
                >
                  {isInside ? 'IN ZONE' : 'outside'}
                </span>
              ) : null}

              <div style={{ marginLeft: 'auto', display: 'flex', gap: 5 }}>
                {correction ? (
                  <span
                    className="mono"
                    style={{
                      fontSize: 'var(--fs-label)',
                      color: correction.verdict === 'false_positive' ? 'var(--crit)' : 'var(--good)',
                    }}
                  >
                    {correction.verdict === 'false_positive' ? 'MARKED WRONG' : 'CONFIRMED'}
                  </span>
                ) : canCorrect ? (
                  <button
                    className="button button--quiet"
                    disabled={busyTrackId === trackId}
                    onClick={(event) => {
                      event.stopPropagation();
                      correct(trackId, 'false_positive');
                    }}
                    style={{ padding: '2px 7px', fontSize: 'var(--fs-label)' }}
                  >
                    Flag as wrong
                  </button>
                ) : null}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
