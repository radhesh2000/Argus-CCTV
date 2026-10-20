/**
 * Live Operations Wall.
 *
 * The grid, the NEEDS ATTENTION strip above it, the focus view with its zone
 * sidebar, and the ticker along the bottom.
 *
 * "Needs attention" is derived, not configured: a camera is hot if a zone on it
 * is over its sanctioned limit right now, or if it raised an alert in the last
 * fifteen minutes. That is the same judgement the engine made when it decided
 * to write an event, surfaced where somebody is looking.
 */

import { useEffect, useMemo, useState } from 'react';
import { api } from '../api/client';
import type { ArgusEvent, LiveCamera, LiveState } from '../api/types';
import { VideoFrame } from '../components/VideoFrame';
import { TrackBoxes, ZonePolygons, inReadingOrder, trackTag } from '../components/Overlay';
import { SeverityGlyph } from '../components/marks';
import { useResource } from '../lib/hooks';
import { clockTime, decodeLabel, originLook, relativeTime, shortTime } from '../lib/format';
import { useAppState } from '../state/AppState';

const HOT_WINDOW_SEC = 15 * 60;

interface Props {
  live: LiveState | null;
  loading: boolean;
  focusId: string | null;
  onFocus: (cameraId: string | null) => void;
  now: Date;
}

export function LiveWall({ live, loading, focusId, onFocus, now }: Props) {
  const { policyLabels, arrivals } = useAppState();
  // Recent events for the ticker and the hot-camera labels. Polled slowly and
  // merged with SSE arrivals, so a new alert lights a tile immediately.
  const recent = useResource(() => api.events({ limit: 40 }), []);

  const events = useMemo(() => {
    const seen = new Set<number>();
    return [...arrivals, ...(recent.data?.events ?? [])]
      .filter((event) => (seen.has(event.id) ? false : seen.add(event.id)))
      .sort((a, b) => b.ts - a.ts);
  }, [arrivals, recent.data]);

  const cameras = live?.cameras ?? [];
  const focus = focusId ? cameras.find((camera) => camera.id === focusId) ?? null : null;
  // Processing paused is a deliberate state, not a fleet of dead cameras: the
  // wall reads it calm — tiles say "paused", the count strip says why, and the
  // "needs attention" heuristic (which would otherwise light every tile that
  // alerted before the pause) stands down.
  const paused = live?.processing?.status === 'paused';

  const alertsByCamera = useMemo(() => {
    const cutoff = Date.now() / 1000 - HOT_WINDOW_SEC;
    const map = new Map<string, ArgusEvent>();
    for (const event of events) {
      if (event.ts >= cutoff && !map.has(event.camera_id)) map.set(event.camera_id, event);
    }
    return map;
  }, [events]);

  const hot = paused
    ? []
    : cameras.filter(
        (camera) => camera.zones.some((zone) => zone.over_limit) || alertsByCamera.has(camera.id),
      );

  const online = cameras.filter((camera) => camera.status === 'live').length;
  const degraded = cameras.filter((camera) => camera.status === 'connecting').length;
  // A camera nobody asked the engine to watch is not a camera that went dark.
  // Counting the two together is the same mistake §3.1 of
  // docs/PRODUCTION_READINESS.md was about: `offline` is what a cut cable reads,
  // and an operator's next move for those two facts is not the same.
  const disabled = cameras.filter((camera) => !camera.enabled).length;
  const offline = cameras.filter(
    (camera) => camera.enabled && (camera.status === 'offline' || camera.status === 'error'),
  ).length;

  const columns = cameras.length <= 4 ? 2 : cameras.length <= 12 ? 4 : 6;

  return (
    <div className="rise" style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
      <div className="wall__head">
        <div className="screen-title">Live Operations Wall</div>
        <div className="mono" style={{ fontSize: 'var(--fs-note)', color: 'var(--dim)' }}>
          {live?.shift ? `SHIFT ${live.shift.id} · ` : ''}
          {clockTime(now)}
        </div>
        {/* The four numbers a supervisor crosses the room to read, set as
            numbers rather than as a footnote in the same ink as the clock. A
            count that is not zero when it should be gets the warning ink. When
            processing is paused there is nothing to count — the cameras are not
            offline, they are simply not being watched — so the strip says that
            instead of flashing a fleet of false "offline"s. */}
        {paused ? (
          <div className="wall__counts">
            <span>
              <span className="live-dot live-dot--paused" />
              <span className="wall__count">{cameras.length}</span>{' '}
              {cameras.length === 1 ? 'camera' : 'cameras'} · processing paused
            </span>
          </div>
        ) : (
          <div className="wall__counts">
            <span>
              <span className="live-dot breathe" />
              <span className="wall__count">{online}</span> live
            </span>
            <span>
              <span style={{ color: 'var(--warn)' }}>◆</span>
              <span className={`wall__count ${degraded ? 'wall__count--warn' : ''}`}>{degraded}</span>{' '}
              connecting
            </span>
            <span>
              <span style={{ color: 'var(--dim)' }}>○</span>
              <span className={`wall__count ${offline ? 'wall__count--warn' : ''}`}>{offline}</span>{' '}
              offline
            </span>
            {disabled ? (
              <span>
                <span style={{ color: 'var(--dim)' }}>⊘</span>
                <span className="wall__count">{disabled}</span> disabled
              </span>
            ) : null}
          </div>
        )}
      </div>

      {focus ? (
        <FocusView
          camera={focus}
          events={events.filter((event) => event.camera_id === focus.id).slice(0, 8)}
          policyLabels={policyLabels}
          now={now}
          onClose={() => onFocus(null)}
        />
      ) : (
        <>
          <div className="wall__body">
            {cameras.length === 0 && !loading ? (
              <div className="notice" style={{ maxWidth: 620 }}>
                <div className="notice__title">No cameras yet</div>
                <div className="notice__body">
                  Nothing is enrolled on this appliance. Add an RTSP or ONVIF camera from
                  Cameras &amp; Zones, or run <code>argus seed</code> to import the demo fleet from
                  config/seed.yaml.
                </div>
              </div>
            ) : null}

            {paused && cameras.length > 0 ? (
              <div className="notice" style={{ margin: '4px 0 14px', maxWidth: 720 }}>
                <div className="notice__title">Processing is paused</div>
                <div className="notice__body">
                  No camera is being decoded and no detection is running, so the tiles below are
                  dark by design rather than by fault. Zone counts, tracks and alerts resume the
                  moment processing does — an administrator can resume it from the banner above.
                </div>
              </div>
            ) : null}

            {hot.length > 0 ? (
              <>
                <div className="wall__section" style={{ color: 'var(--warn)' }}>
                  Needs attention · {hot.length}
                </div>
                <div className="wall__hot">
                  {hot.map((camera) => (
                    <HotTile
                      key={camera.id}
                      camera={camera}
                      alert={alertsByCamera.get(camera.id) ?? null}
                      policyLabels={policyLabels}
                      onOpen={() => onFocus(camera.id)}
                    />
                  ))}
                </div>
                <div className="wall__section">All cameras · {cameras.length}</div>
              </>
            ) : null}

            <div className="wall__grid" style={{ gridTemplateColumns: `repeat(${columns}, 1fr)` }}>
              {cameras.map((camera) => (
                <WallTile
                  key={camera.id}
                  camera={camera}
                  now={now}
                  paused={paused}
                  onOpen={() => onFocus(camera.id)}
                />
              ))}
            </div>
          </div>
          <Ticker events={events.slice(0, 6)} policyLabels={policyLabels} />
        </>
      )}
    </div>
  );
}

/** Total people counted across a camera's zones, and whether that is knowable. */
function occupancy(camera: LiveCamera): { count: number | null; limit: number | null; over: boolean } {
  const known = camera.zones.filter((zone) => zone.count !== null);
  const limits = camera.zones.filter((zone) => zone.sanctioned_limit !== null);
  return {
    count: known.length ? known.reduce((sum, zone) => sum + (zone.count ?? 0), 0) : null,
    limit: limits.length ? limits.reduce((sum, zone) => sum + (zone.sanctioned_limit ?? 0), 0) : null,
    over: camera.zones.some((zone) => zone.over_limit),
  };
}

function healthMark(camera: LiveCamera): { glyph: string; color: string; breathing: boolean } {
  if (!camera.enabled) return { glyph: '⊘', color: 'var(--faint)', breathing: false };
  if (camera.status === 'live') return { glyph: '●', color: 'var(--good-dot)', breathing: true };
  if (camera.status === 'connecting') return { glyph: '◆', color: 'var(--warn)', breathing: false };
  return { glyph: '○', color: 'var(--faint)', breathing: false };
}

function WallTile({
  camera,
  now,
  paused = false,
  onOpen,
}: {
  camera: LiveCamera;
  now: Date;
  paused?: boolean;
  onOpen: () => void;
}) {
  const dark = paused || !camera.enabled || camera.status === 'offline' || camera.status === 'error';
  const { count, limit, over } = occupancy(camera);
  const mark = healthMark(camera);

  return (
    <div
      className={`tile tile--16x9 ${over ? 'tile--hot' : ''}`}
      onClick={onOpen}
      role="button"
      tabIndex={0}
      onKeyDown={(event) => event.key === 'Enter' && onOpen()}
    >
      {dark ? (
        <div className="tile__signal">
          <span style={{ fontSize: 'var(--fs-figure)' }}>{paused && camera.enabled ? '⏸' : '○'}</span>
          <span className="mono" style={{ fontSize: 'var(--fs-note)' }}>
            {paused && camera.enabled
              ? 'PAUSED · PROCESSING STOPPED'
              : !camera.enabled
                ? 'DISABLED · NOT BEING WATCHED'
                : 'NO SIGNAL'}
            {!paused && camera.enabled && camera.last_seen_at
              ? ` · ${relativeTime(new Date(camera.last_seen_at).getTime() / 1000)}`
              : ''}
          </span>
        </div>
      ) : (
        <>
          <VideoFrame stream={camera.stream}>
            <ZonePolygons zones={camera.zones} />
            {camera.tracks ? <TrackBoxes boxes={camera.tracks} /> : null}
          </VideoFrame>
          <div className="tile__scan" />
        </>
      )}
      <div className="tile__top">
        <span className="tile__id">{camera.id}</span>
        <span className="tile__name">{camera.name}</span>
        <span
          className={`tile__health ${mark.breathing ? 'breathe' : ''}`}
          style={{ color: mark.color }}
          title={camera.status_detail ?? camera.status}
        >
          {mark.glyph}
        </span>
      </div>
      {!dark ? (
        <div className="tile__bottom">
          <span className={`tile__occ ${over ? 'tile__occ--over' : ''}`}>
            {count === null ? '—' : count}
            {limit !== null ? `/${limit}` : ''}
          </span>
          <span className="tile__meta">{clockTime(now)}</span>
        </div>
      ) : null}
    </div>
  );
}

function HotTile({
  camera,
  alert,
  policyLabels,
  onOpen,
}: {
  camera: LiveCamera;
  alert: ArgusEvent | null;
  policyLabels: Record<string, string>;
  onOpen: () => void;
}) {
  const { count, limit } = occupancy(camera);
  const over = camera.zones.some((zone) => zone.over_limit);
  const label = alert
    ? `${policyLabels[alert.detector_name] ?? alert.detector_name} · ${shortTime(alert.ts)}`
    : 'Zone over sanctioned limit';

  return (
    <div className="tile tile--21x9 tile--hot" onClick={onOpen} role="button" tabIndex={0}>
      <VideoFrame stream={camera.stream}>
        <ZonePolygons zones={camera.zones} />
        {camera.tracks ? <TrackBoxes boxes={camera.tracks} /> : null}
      </VideoFrame>
      <div className="tile__top">
        <span className="tile__id">{camera.id}</span>
        <span className="tile__name">{camera.name}</span>
      </div>
      <div className="tile__alertbar">
        <span style={{ color: 'var(--hot)', fontSize: 'var(--fs-note)' }}>◆</span>
        <span style={{ fontSize: 'var(--fs-meta)', fontWeight: 600, color: 'var(--on-video)' }}>{label}</span>
        <span className={`tile__occ ${over ? 'tile__occ--over' : ''}`}>
          {count === null ? '—' : count}
          {limit !== null ? `/${limit}` : ''}
        </span>
      </div>
    </div>
  );
}

function FocusView({
  camera,
  events,
  policyLabels,
  now,
  onClose,
}: {
  camera: LiveCamera;
  events: ArgusEvent[];
  policyLabels: Record<string, string>;
  now: Date;
  onClose: () => void;
}) {
  const [zoneIndex, setZoneIndex] = useState(0);
  const zone = camera.zones[Math.min(zoneIndex, Math.max(0, camera.zones.length - 1))] ?? null;
  const dark = !camera.enabled || camera.status === 'offline' || camera.status === 'error';
  const tracks = camera.tracks ?? [];
  const decode = decodeLabel(camera.decode_mode);

  // A quiet head-ring tells an operator someone is standing there; it does
  // not say who, and on a packing line answering that takes a real box. This
  // is the same hover/select promotion the evidence viewer already uses
  // between its detection list and its overlay — one interaction, reused.
  const [hoveredTrackId, setHoveredTrackId] = useState<number | null>(null);
  const [selectedTrackId, setSelectedTrackId] = useState<number | null>(null);
  useEffect(() => {
    setHoveredTrackId(null);
    setSelectedTrackId(null);
  }, [camera.id]);

  return (
    <div className="focus">
      <div className="focus__stage">
        {dark ? (
          <div className="tile__signal">
            <span style={{ fontSize: 'var(--fs-vital)' }}>○</span>
            <span className="mono" style={{ fontSize: 'var(--fs-meta)' }}>
              NO SIGNAL · {camera.status_detail ?? camera.status}
            </span>
          </div>
        ) : (
          <>
            <VideoFrame stream={camera.stream}>
              <ZonePolygons zones={camera.zones} emphasis="focus" />
              <TrackBoxes
                boxes={tracks}
                labelled
                emphasis="focus"
                interactive
                hoveredTrackId={hoveredTrackId}
                selectedTrackId={selectedTrackId}
                onHoverTrack={setHoveredTrackId}
                onSelectTrack={setSelectedTrackId}
              />
            </VideoFrame>
            <div className="tile__scan" />
          </>
        )}
        <div className="focus__id mono">
          {camera.id} · {camera.name}
        </div>
        <div className="focus__meta mono">
          {clockTime(now)} · {camera.fps ? `${camera.fps.toFixed(2)} FPS detection` : 'no engine'}
          {decode ? ` · ${decode}` : ''}
        </div>
        <button className="focus__close mono" onClick={onClose}>
          ESC · CLOSE
        </button>
      </div>

      <div className="focus__side">
        <div>
          <div className="eyebrow" style={{ marginBottom: 10 }}>
            {camera.zones.length > 1 ? (
              <select
                className="focus__zonepick"
                value={zoneIndex}
                onChange={(event) => setZoneIndex(Number(event.target.value))}
              >
                {camera.zones.map((entry, index) => (
                  <option key={entry.id} value={index}>
                    ZONE · {entry.name}
                  </option>
                ))}
              </select>
            ) : (
              `Zone · ${zone?.name ?? 'none defined'}`
            )}
          </div>
          {zone ? (
            <>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
                <span className="mono" style={{ fontSize: 'var(--fs-title)', fontWeight: 700 }}>
                  {zone.count === null ? '—' : zone.count}
                </span>
                <span className="mono" style={{ color: 'var(--faint)', fontSize: 'var(--fs-body)' }}>
                  / {zone.sanctioned_limit ?? '—'} sanctioned
                </span>
              </div>
              <div className="focus__bar">
                <div
                  className="focus__barfill"
                  style={{
                    width: `${
                      zone.count !== null && zone.sanctioned_limit
                        ? Math.min(100, (zone.count / zone.sanctioned_limit) * 100)
                        : 0
                    }%`,
                    background: zone.over_limit ? 'var(--warn)' : 'var(--accent)',
                  }}
                />
              </div>
            </>
          ) : (
            <div style={{ fontSize: 'var(--fs-meta)', color: 'var(--faint)' }}>
              No zone drawn on this camera yet — add one in Cameras &amp; Zones and headcount
              policies will start applying to it.
            </div>
          )}
        </div>

        <div>
          <div className="eyebrow" style={{ marginBottom: 8 }}>
            Active tracks
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {tracks.length === 0 ? (
              <span style={{ fontSize: 'var(--fs-meta)', color: 'var(--faint)' }}>
                {camera.tracks === null ? 'Engine not running' : 'Nobody in frame'}
              </span>
            ) : (
              // Named and ordered by the same helper the overlay labels from,
              // so "#4" here is the fourth person read down the frame there.
              // Hover or click a row and that one person's quiet head-ring
              // promotes to a full bright box on the frame — the same
              // list-drives-overlay pairing the evidence viewer already uses.
              inReadingOrder(tracks).map(({ box, index, ordinal }) => {
                const trackId = box[4];
                const active = trackId === (hoveredTrackId ?? selectedTrackId);
                return (
                  <div
                    key={`${trackId}-${index}`}
                    className="focus__track mono"
                    onMouseEnter={() => setHoveredTrackId(trackId)}
                    onMouseLeave={() => setHoveredTrackId(null)}
                    onClick={() => setSelectedTrackId(trackId === selectedTrackId ? null : trackId)}
                    style={{
                      cursor: 'pointer',
                      padding: '3px 5px',
                      margin: '-3px -5px',
                      borderRadius: 4,
                      background: active ? 'rgba(227,161,60,.14)' : 'transparent',
                      border: `1px solid ${active ? 'rgba(227,161,60,.5)' : 'transparent'}`,
                    }}
                  >
                    <span style={{ color: 'var(--good)' }}>#{ordinal}</span>
                    <span>confidence {(box[5] * 100).toFixed(0)}%</span>
                    <span style={{ marginLeft: 'auto', color: 'var(--faint)' }}>
                      {trackId >= 0 ? trackTag(trackId) : 'unnamed'}
                    </span>
                  </div>
                );
              })
            )}
          </div>
        </div>

        <div>
          <div className="eyebrow" style={{ marginBottom: 8 }}>
            Recent events · this camera
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {events.length === 0 ? (
              <span style={{ fontSize: 'var(--fs-meta)', color: 'var(--faint)' }}>Nothing logged recently.</span>
            ) : (
              events.map((event) => (
                <div key={event.id} className="focus__event">
                  <SeverityGlyph severity={event.severity} size={9} />
                  <span
                    className="mono"
                    style={{ fontSize: 'var(--fs-label)', color: 'var(--faint)' }}
                    title={originLook(event.detector_type).title}
                  >
                    {originLook(event.detector_type).short}
                  </span>
                  <span style={{ flex: 1 }}>
                    {policyLabels[event.detector_name] ?? event.detector_name}
                  </span>
                  <span className="mono" style={{ fontSize: 'var(--fs-note)', color: 'var(--faint)' }}>
                    {shortTime(event.ts)}
                  </span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function Ticker({
  events,
  policyLabels,
}: {
  events: ArgusEvent[];
  policyLabels: Record<string, string>;
}) {
  return (
    <div className="ticker">
      <span className="mono ticker__label">TICKER</span>
      {events.length === 0 ? (
        <span style={{ fontSize: 'var(--fs-note)', color: 'var(--faint)' }}>No events logged yet.</span>
      ) : (
        events.map((event) => (
          <div key={event.id} className="ticker__item slidein">
            <SeverityGlyph severity={event.severity} size={9} />
            <span>{policyLabels[event.detector_name] ?? event.detector_name}</span>
            <span className="mono" style={{ fontSize: 'var(--fs-note)', color: 'var(--faint)' }}>
              {originLook(event.detector_type).short} · {event.camera_id} · {shortTime(event.ts)}
            </span>
          </div>
        ))
      )}
    </div>
  );
}
