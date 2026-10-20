/**
 * Wall TV — the control-room screen nobody sits at.
 *
 * Its own dark frame regardless of the app theme (a bright paper wall on a
 * 55-inch panel at night is not a kindness), no chrome, big numbers, and the
 * most recent critical alert across the top. Everything on it is legible from
 * the other side of a room, which is the only requirement it has.
 */

import { useMemo } from 'react';
import type { ArgusEvent, LiveCamera } from '../api/types';
import { VideoFrame } from './VideoFrame';
import { TrackBoxes, ZonePolygons } from './Overlay';
import { clockTime, shortTime } from '../lib/format';

interface Props {
  cameras: LiveCamera[];
  alerts: ArgusEvent[];
  policyLabels: Record<string, string>;
  now: Date;
  engineRunning: boolean;
  /** Processing is paused: the wall reads calm rather than a red fleet of dead
   *  cameras — this is the switch, not an outage. */
  paused?: boolean;
  onExit: () => void;
}

export function TvWall({ cameras, alerts, policyLabels, now, engineRunning, paused = false, onExit }: Props) {
  const online = cameras.filter((camera) => camera.status === 'live');
  const degraded = cameras.filter(
    (camera) => camera.status === 'connecting' || (camera.status === 'live' && camera.fps !== null && camera.fps < camera.target_fps * 0.6),
  );
  // A disabled camera is not a camera that went dark: nobody asked the engine
  // to watch it. Counting the two together sends an operator to look for a cut
  // cable — the same mistake docs/PRODUCTION_READINESS.md §3.1 was about.
  const disabled = cameras.filter((camera) => !camera.enabled);
  const offline = cameras.filter(
    (camera) => camera.enabled && (camera.status === 'offline' || camera.status === 'error'),
  );

  const headline = useMemo(() => alerts.find((event) => event.severity === 'high') ?? alerts[0], [alerts]);
  const columns = Math.max(1, Math.min(4, Math.ceil(Math.sqrt(Math.max(1, cameras.length)))));

  return (
    <div className="tv">
      <div className="tv__bar">
        <div className="tv__brand">ARGUS</div>
        {paused ? (
          <div className="tv__stat" style={{ color: 'var(--hot)' }}>
            <span className="live-dot live-dot--paused" />
            PROCESSING PAUSED
          </div>
        ) : (
          <>
            <div className="tv__stat" style={{ color: 'var(--on-video)' }}>
              <span className={`live-dot ${engineRunning ? 'breathe' : 'live-dot--down'}`} />
              {online.length} LIVE
            </div>
            <div className="tv__stat" style={{ color: 'var(--hot)' }}>
              <span>◆</span>
              {degraded.length} DEGRADED
            </div>
            <div className="tv__stat" style={{ color: 'var(--on-video-dim)' }}>
              <span>○</span>
              {offline.length} OFFLINE
            </div>
            {disabled.length ? (
              <div className="tv__stat" style={{ color: 'var(--on-video-dim)' }}>
                <span>⊘</span>
                {disabled.length} DISABLED
              </div>
            ) : null}
          </>
        )}
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 18 }}>
          <span className="tv__clock">{clockTime(now)}</span>
          <button className="tv__exit" onClick={onExit}>
            ESC · EXIT
          </button>
        </div>
      </div>
      {/*
        The breach, across the whole wall.

        It used to be a pill in the counter row, where the clock — bigger, and
        never news — outweighed it. A supervisor four metres away needs to be
        able to read what happened and where without walking closer, so it gets
        its own strip, in the one saturated colour this product owns.
      */}
      {headline ? (
        <div className="tv__alert">
          <span className="tv__alerttag">BREACH</span>
          <span className="tv__alertwhat">
            {policyLabels[headline.detector_name] ?? headline.detector_name}
          </span>
          <span className="tv__alertwhere">
            {headline.camera_id} · {shortTime(headline.ts)}
          </span>
        </div>
      ) : null}
      <div className="tv__grid" style={{ gridTemplateColumns: `repeat(${columns}, 1fr)` }}>
        {cameras.map((camera) => {
          const dark =
            paused || !camera.enabled || camera.status === 'offline' || camera.status === 'error';
          const zone = camera.zones[0];
          const over = camera.zones.some((entry) => entry.over_limit);
          const total = camera.zones.reduce(
            (sum, entry) => (entry.count === null ? sum : sum + entry.count),
            0,
          );
          const known = camera.zones.some((entry) => entry.count !== null);
          const limit = camera.zones.reduce(
            (sum, entry) => sum + (entry.sanctioned_limit ?? 0),
            0,
          );
          return (
            <div key={camera.id} className="tv__tile">
              {dark ? (
                <div className="tile__signal">
                  <span className="tv__signalglyph">
                    {paused && camera.enabled ? '⏸' : camera.enabled ? '○' : '⊘'}
                  </span>
                  <span className="tv__signalword">
                    {paused && camera.enabled ? 'PAUSED' : camera.enabled ? 'NO SIGNAL' : 'DISABLED'}
                  </span>
                </div>
              ) : (
                <>
                  <VideoFrame stream={camera.stream}>
                    <ZonePolygons zones={camera.zones} />
                    {camera.tracks ? <TrackBoxes boxes={camera.tracks} /> : null}
                  </VideoFrame>
                  <div className="tv__count">
                    <span className={`tv__countnow ${over ? 'tv__countnow--over' : ''}`}>
                      {known ? total : '—'}
                    </span>
                    {limit > 0 ? <span className="tv__countlimit">/{limit}</span> : null}
                  </div>
                </>
              )}
              <div className="tv__label">
                <span className="mono tv__id">{camera.id}</span>
                {zone ? <span className="tv__zone">{zone.name}</span> : null}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
