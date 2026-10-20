/**
 * What the vision engine sees, drawn on top of what the camera sees.
 *
 * Zone polygons and track boxes both arrive normalized (0–1 of the frame), so
 * one SVG with a 0–100 viewBox and `preserveAspectRatio="none"` lands them
 * correctly on any tile size and on any camera resolution — which is why the
 * engine normalizes them in the first place.
 *
 * An over-limit zone switches from its own colour to the alert amber, because
 * the one thing an operator must be able to do from across a control room is
 * tell which zone is the problem.
 *
 * ---------------------------------------------------------------------------
 * Adaptive level of detail
 * ---------------------------------------------------------------------------
 *
 * The complaint that produced this file's second half was "so many bounding
 * boxes being drawn, most overlapping, I can't tell what I'm looking at". The
 * detector was not at fault — a measured sweep of the labelled set found zero
 * duplicate boxes in 162 detections. A packing line simply has up to 25 people
 * on it, and a worker at the far end of the hall is about 30 px tall. Drawing
 * a full rectangle plus a `Worker 2053 · 77%` tag around a 30 px person means
 * the annotation is several times larger than the thing annotated, so the
 * annotations collide with each other and bury the people underneath them.
 *
 * The fix is presentational and it is the oldest rule in the book: overview
 * first, details on demand.
 *
 *   - A crowded frame gets one quiet head marker per person and no labels.
 *     A ring at the top-centre of the box is where the eye already looks for a
 *     person, and it is the point least likely to land on top of a neighbour —
 *     two people standing shoulder to shoulder have overlapping bodies but
 *     separate heads.
 *   - A sparse frame gets corner brackets rather than a closed rectangle. The
 *     corners carry all the extent information a rectangle does and leave the
 *     person's face and torso unobscured.
 *   - Exactly one person — the one the pointer or the detection list is on —
 *     is ever promoted to a full bright box with a label.
 *
 * Track ids do not appear on the frame any more. `T-2053` is a fact about the
 * tracker's bookkeeping, not about the plant; an operator asked to compare the
 * overlay with a list needs `#3`, which is short enough to fit over a 30 px
 * person and is derived from where the person is standing. The id survives as
 * dimmed secondary detail in the detection list, because a reviewer's
 * correction is keyed by it.
 */

import { useEffect, useRef, useState } from 'react';
import type { Box, LiveZone } from '../api/types';
import { percent } from '../lib/format';

const OVER = 'rgba(227,161,60,'; // --hot, as an rgb triple for fills
const CALM = 'rgba(179,52,31,'; // the design's zone stroke on a quiet tile

function points(polygon: [number, number][]): string {
  return polygon.map(([x, y]) => `${(x * 100).toFixed(2)},${(y * 56).toFixed(2)}`).join(' ');
}

interface ZoneProps {
  zones: LiveZone[];
  /** Thicker strokes read better on a focus view or a TV across the room. */
  emphasis?: 'tile' | 'focus';
}

export function ZonePolygons({ zones, emphasis = 'tile' }: ZoneProps) {
  const width = emphasis === 'focus' ? 0.5 : 0.6;
  return (
    <svg
      viewBox="0 0 100 56"
      preserveAspectRatio="none"
      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
    >
      {zones.map((zone) => {
        const base = zone.over_limit ? OVER : CALM;
        return (
          <polygon
            key={zone.id}
            points={points(zone.polygon)}
            style={{
              fill: `${base}${zone.over_limit ? '0.10' : '0.09'})`,
              stroke: `${base}${zone.over_limit ? '0.9' : '0.5'})`,
              strokeWidth: width,
            }}
          />
        );
      })}
    </svg>
  );
}

/* -------------------------------------------------------------------------
 * How much detail a frame can carry
 * ---------------------------------------------------------------------- */

/** `markers` is the quiet head-ring treatment; `boxes` is corner brackets. */
export type OverlayDetail = 'boxes' | 'markers';

/**
 * Above this many people, rectangles start touching each other.
 *
 * Six is where the labelled packing-line frames stop being a handful of
 * separable people and start being a line: at seven or more, adjacent workers'
 * boxes overlap in more than half the sampled frames, and an overlap between
 * two rectangles reads as a third rectangle.
 */
export const CROWD_COUNT = 6;

/**
 * Below this box height — as a fraction of the frame — a rectangle is smaller
 * than its own decoration.
 *
 * A far worker on the measured footage is roughly 30 px tall in a 1080 px
 * frame, so about 3% of frame height; anything under 12% cannot hold a label
 * (11 px type plus padding is ~17 px, more than half such a box) and its
 * corner brackets would meet in the middle. The median is used rather than the
 * minimum so that one distant straggler does not downgrade a frame of six
 * people standing at the bench in front of the camera.
 */
export const SMALL_BOX_HEIGHT = 0.12;

/**
 * The one place the crowded/sparse rule lives, so it can be tuned once.
 *
 * A frame is crowded when it holds more than `CROWD_COUNT` people, or when the
 * median person is shorter than `SMALL_BOX_HEIGHT` of the frame. Either
 * condition alone is enough: four people at the far end of the hall are as
 * unreadable in rectangles as fifteen people at the bench.
 */
export function overlayDensity(boxes: Box[]): OverlayDetail {
  if (boxes.length === 0) return 'boxes';
  if (boxes.length > CROWD_COUNT) return 'markers';
  return medianBoxHeight(boxes) < SMALL_BOX_HEIGHT ? 'markers' : 'boxes';
}

function medianBoxHeight(boxes: Box[]): number {
  const heights = boxes.map(([, y1, , y2]) => Math.max(0, y2 - y1)).sort((a, b) => a - b);
  const middle = heights.length >> 1;
  const upper = heights[middle] ?? 0;
  return heights.length % 2 === 1 ? upper : ((heights[middle - 1] ?? 0) + upper) / 2;
}

/* -------------------------------------------------------------------------
 * The number an operator actually says out loud
 * ---------------------------------------------------------------------- */

/**
 * How far apart, as a fraction of frame height, two people's box centres can
 * be and still count as standing in the same row.
 *
 * Without a row, a strict sort by centre-y would number a line of workers at
 * one bench in whatever order their heads happen to fall by a pixel or two,
 * and the ordinals would zig-zag across the frame. A tenth of the frame is
 * about one person's head, which is roughly the tolerance a human uses for
 * "those two are standing together".
 *
 * Rows are found by *gap* rather than by a fixed grid, because a grid puts a
 * hard edge somewhere arbitrary and two people either side of it — 0.5% of the
 * frame apart — end up in different rows and far apart in the numbering. A gap
 * has no such edge: the band only has to be smaller than the space between two
 * genuine rows, not aligned with anything.
 */
export const ORDINAL_ROW_BAND = 0.1;

/**
 * Per-frame reading-order ordinals: the returned array is parallel to `boxes`,
 * and `ordinals[i]` is what `boxes[i]` is called on screen and in the list.
 *
 * Ordering is by box centre — rows from the back of the shot forwards, and
 * left to right within a row. It is derived purely from geometry, so the
 * overlay and any list built from the same `boxes` array agree without having
 * to pass numbers between them.
 *
 * Where people stand at continuously increasing depth rather than in ranks,
 * the gap rule chains them into one long row and the numbering becomes plain
 * left-to-right. That is the right answer for that picture: there are no rows
 * in it to read in order.
 */
export function frameOrdinals(boxes: Box[]): number[] {
  const marks = boxes.map((box, index) => ({
    index,
    y: (box[1] + box[3]) / 2,
    x: (box[0] + box[2]) / 2,
  }));

  const row = new Map<number, number>();
  let band = 0;
  let previous: number | null = null;
  for (const mark of [...marks].sort((a, b) => a.y - b.y || a.x - b.x || a.index - b.index)) {
    if (previous !== null && mark.y - previous > ORDINAL_ROW_BAND) band += 1;
    row.set(mark.index, band);
    previous = mark.y;
  }

  marks.sort(
    (a, b) =>
      (row.get(a.index) ?? 0) - (row.get(b.index) ?? 0) || a.x - b.x || a.index - b.index,
  );
  const ordinals = new Array<number>(boxes.length).fill(0);
  marks.forEach((mark, rank) => {
    ordinals[mark.index] = rank + 1;
  });
  return ordinals;
}

export interface OrderedDetection {
  box: Box;
  /** Where it sat in the caller's original `boxes` array — `meta.inside` and
   *  anything else parallel to that array is still indexed by this. */
  index: number;
  ordinal: number;
}

/**
 * The same detections, as rows in the order they are numbered.
 *
 * Every list that sits beside a frame — the evidence viewer's detection panel,
 * the live wall's focus sidebar — goes through here, so a list never disagrees
 * with the canvas about either a person's number or their place in the order.
 */
export function inReadingOrder(boxes: Box[]): OrderedDetection[] {
  const ordinals = frameOrdinals(boxes);
  return boxes
    .map((box, index) => ({ box, index, ordinal: ordinals[index] ?? index + 1 }))
    .sort((a, b) => a.ordinal - b.ordinal);
}

/* -------------------------------------------------------------------------
 * The overlay itself
 * ---------------------------------------------------------------------- */

/** Corner arm length: a share of the box's shorter side, within these bounds. */
const ARM_SHARE = 0.22;
const ARM_MIN = 6;
const ARM_MAX = 18;

interface BoxProps {
  boxes: Box[];
  /**
   * Label every person, not just the active one — right on the focus view,
   * where the sidebar lists the same people and there is room to read. It is
   * ignored in a crowded frame: a label per person is exactly the pile-up this
   * component exists to prevent, and a crowded frame labels only the box the
   * operator is pointing at.
   */
  labelled?: boolean;
  /**
   * Force a level of detail. `auto` (the default, and what every caller uses)
   * asks `overlayDensity` about this particular frame.
   */
  density?: 'auto' | OverlayDetail;
  /**
   * A quiet head-ring is sized for a small Live Wall tile, where it is already
   * most of the available space. The focus view renders the same frame many
   * times larger, and at that size a 7px outline ring reads as visual noise
   * rather than a person — it needs to be bigger and to read as a mark, not
   * just an edge, to survive a first glance across a busy packing line.
   */
  emphasis?: 'tile' | 'focus';
  /**
   * Turns hover/click on. With it off — the Live Wall tiles, the TV wall,
   * Triage — the overlay is a quiet read-only annotation. With it on (the
   * Evidence Viewer) every person keeps the quiet treatment and the one under
   * the pointer is promoted to a full bright box with a label.
   */
  interactive?: boolean;
  /** Which track the pointer or the detection list is currently on. */
  hoveredTrackId?: number | null;
  /** Which track is pinned open — stays highlighted after the pointer leaves. */
  selectedTrackId?: number | null;
  /** Boxes a reviewer has hidden via the detection list's visibility toggle —
   *  not deleted, not corrected, just decluttered off the canvas for now. */
  hiddenTrackIds?: ReadonlySet<number>;
  onHoverTrack?: (trackId: number | null) => void;
  onSelectTrack?: (trackId: number | null) => void;
}

export function TrackBoxes({
  boxes,
  labelled = false,
  density = 'auto',
  emphasis = 'tile',
  interactive = false,
  hoveredTrackId = null,
  selectedTrackId = null,
  hiddenTrackIds,
  onHoverTrack,
  onSelectTrack,
}: BoxProps) {
  // The corner arms are a share of the *person*, and a person's size on screen
  // is only knowable in pixels, so the layer measures itself. Until the first
  // measurement lands the arms sit at their minimum, which is the right way to
  // be wrong for one frame.
  const layer = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  useEffect(() => {
    const element = layer.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      if (!entry) return;
      const { width, height } = entry.contentRect;
      setSize({ width, height });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const detail = density === 'auto' ? overlayDensity(boxes) : density;
  const crowded = detail === 'markers';
  const ordinals = frameOrdinals(boxes);
  // Exactly one person is ever promoted. The pointer wins over the pinned
  // selection, so moving across the frame never leaves two bright boxes.
  const activeTrackId = interactive ? (hoveredTrackId ?? selectedTrackId) : null;

  // Painted largest first, so a far worker whose box sits wholly inside a near
  // worker's box is drawn — and hit-tested — on top of it. Without this the
  // small boxes in a receding line are unclickable.
  const painted = boxes
    .map((box, index) => ({ box, index }))
    .filter(({ box }) => !hiddenTrackIds?.has(box[4]))
    .sort((a, b) => boxArea(b.box) - boxArea(a.box));

  return (
    <div ref={layer} className="ov-layer">
      {painted.map(({ box, index }) => {
        const [rawX1, rawY1, rawX2, rawY2, trackId, confidence] = box;
        // Coordinates are supposed to be normalized 0-1, but nothing upstream
        // guarantees it — a track the tracker is coasting through an
        // occlusion, or a genuinely bad detection, can arrive with a
        // near-zero-area box sitting outside the frame entirely. Unclamped,
        // that renders as a stray dot with no relationship to anything on
        // screen, indistinguishable from a real detection at a glance.
        // Clamping to the frame and then requiring *some* visible area (not
        // just non-negative) is what tells the two apart: a person truly
        // half out of frame still clamps to a real sliver; a degenerate box
        // clamps to nothing and is skipped instead of drawn as a dot.
        const x1 = Math.min(1, Math.max(0, rawX1));
        const y1 = Math.min(1, Math.max(0, rawY1));
        const x2 = Math.min(1, Math.max(0, rawX2));
        const y2 = Math.min(1, Math.max(0, rawY2));
        const MIN_VISIBLE_FRACTION = 0.01; // 1% of the frame on that axis
        if (x2 - x1 < MIN_VISIBLE_FRACTION || y2 - y1 < MIN_VISIBLE_FRACTION) return null;
        const width = Math.max(0, x2 - x1);
        const height = Math.max(0, y2 - y1);
        const active = trackId === activeTrackId;
        const dimmed = activeTrackId !== null && !active;
        const shaped = active || !crowded;
        const labelHere = trackId >= 0 && (active || (labelled && !crowded));
        const arm = clamp(
          ARM_SHARE * Math.min(width * size.width, height * size.height),
          ARM_MIN,
          ARM_MAX,
        );
        return (
          <div
            key={`${trackId}-${index}`}
            className={`ov-hit${interactive ? ' ov-hit--live' : ''}`}
            style={{
              left: `${x1 * 100}%`,
              top: `${y1 * 100}%`,
              width: `${width * 100}%`,
              height: `${height * 100}%`,
              zIndex: active ? 2 : undefined,
            }}
            onMouseEnter={interactive ? () => onHoverTrack?.(trackId) : undefined}
            onMouseLeave={interactive ? () => onHoverTrack?.(null) : undefined}
            onClick={
              interactive
                ? () => onSelectTrack?.(trackId === selectedTrackId ? null : trackId)
                : undefined
            }
          >
            {shaped ? (
              <span
                className={`ov-shape${active ? ' ov-shape--active' : ''}${dimmed ? ' ov-dim' : ''}`}
                /* One size for all four mask layers — CSS repeats a short
                   layer list — and it is the arm length, which is this
                   person's size and so belongs with the data. */
                style={{ maskSize: `${arm}px ${arm}px`, WebkitMaskSize: `${arm}px ${arm}px` }}
              />
            ) : (
              <span
                className={`ov-head${emphasis === 'focus' ? ' ov-head--focus' : ''}${dimmed ? ' ov-dim' : ''}`}
              />
            )}
            {labelHere ? (
              <span
                className={`ov-label mono${y1 < 0.06 ? ' ov-label--below' : ''}`}
              >
                #{ordinals[index] ?? index + 1}
                {/* The score is a detail, and details are on demand: carrying
                    it on every label in a sparse frame is what makes two
                    labels wide enough to collide over two people standing
                    together. */}
                {active && Number.isFinite(confidence) ? ` · ${percent(confidence, 0)}` : ''}
              </span>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}

function boxArea(box: Box): number {
  return Math.max(0, box[2] - box[0]) * Math.max(0, box[3] - box[1]);
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}

/**
 * The tracker's own name for a person. Not shown on the frame any more — see
 * this file's header — but still printed as dimmed secondary detail in the
 * detection list and the focus sidebar, because a per-box correction is stored
 * against this id and a reviewer occasionally has to quote it.
 */
export function trackTag(trackId: number): string {
  return `T-${String(trackId).padStart(3, '0')}`;
}
