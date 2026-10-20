/**
 * One camera's live picture.
 *
 * WebRTC straight from go2rtc — no MJPEG, no re-encode, which is the whole
 * point of putting a gateway in front of the cameras. Three behaviours that are
 * not decoration:
 *
 * * **The connection only exists while the tile is on screen.** A 48-camera
 *   fleet scrolled past would otherwise hold 48 peer connections open for
 *   tiles nobody is looking at.
 * * **A still frame is the poster.** The wall shows the plant the moment it
 *   renders instead of a grid of black rectangles waiting on ICE.
 * * **Failure is stated.** A tile that cannot connect says so rather than
 *   sitting black and letting an operator assume the camera is dark.
 */

import { useEffect, useRef, useState } from 'react';
import { connectWebRtc, frameUrl } from '../api/gateway';
import type { PlayerState } from '../api/gateway';
import { useOnScreen } from '../lib/hooks';

interface Props {
  stream: string;
  /** Skip the connection entirely (offline camera, or the engine is stopped). */
  disabled?: boolean;
  /**
   * `cover` crops to fill; `fill` stretches to the given box and is what
   * ``VideoFrame`` wants, because it has already measured a box with the
   * video's own aspect ratio and nothing needs cropping or letterboxing.
   */
  fit?: 'cover' | 'contain' | 'fill';
  /** Intrinsic video dimensions, once the browser knows them. */
  onSize?: (width: number, height: number) => void;
  className?: string;
}

/** How often a tile with no live video re-fetches its still. */
const STILL_REFRESH_MS = 5000;

export function CameraVideo({ stream, disabled = false, fit = 'cover', onSize, className }: Props) {
  const holder = useRef<HTMLDivElement>(null);
  const video = useRef<HTMLVideoElement>(null);
  const onScreen = useOnScreen(holder);
  const [state, setState] = useState<PlayerState>('connecting');
  const [still, setStill] = useState<Still | null>(null);

  useEffect(() => {
    if (disabled || !onScreen) return;
    const element = video.current;
    if (!element) return;
    setState('connecting');
    // Since Phase 5 both the signalling socket and the poster go through
    // `/api/gateway/*` on this origin, so the tile no longer needs the
    // gateway's port out of `/api/config` — or a config load at all before it
    // can draw.
    return connectWebRtc(stream, element, setState);
  }, [stream, disabled, onScreen]);

  // Live video failed, and the appliance is still watching this camera: show
  // what it can see. See the `stills` note below.
  useEffect(() => {
    if (disabled || !onScreen || state !== 'failed') {
      setStill(null);
      return;
    }
    let alive = true;
    let current: string | null = null;
    const load = async () => {
      const next = await fetchStill(stream);
      if (!alive) {
        if (next) URL.revokeObjectURL(next.url);
        return;
      }
      if (current) URL.revokeObjectURL(current);
      current = next?.url ?? null;
      setStill(next);
    };
    void load();
    const timer = window.setInterval(() => void load(), STILL_REFRESH_MS);
    return () => {
      alive = false;
      window.clearInterval(timer);
      if (current) URL.revokeObjectURL(current);
    };
  }, [stream, disabled, onScreen, state]);

  const poster = disabled ? undefined : frameUrl(stream);

  return (
    <div ref={holder} className={className} style={{ position: 'absolute', inset: 0 }}>
      <video
        ref={video}
        poster={poster}
        autoPlay
        playsInline
        muted
        onLoadedMetadata={(event) => {
          const element = event.currentTarget;
          onSize?.(element.videoWidth, element.videoHeight);
        }}
        onResize={(event) => {
          const element = event.currentTarget;
          onSize?.(element.videoWidth, element.videoHeight);
        }}
        style={{ width: '100%', height: '100%', objectFit: fit, display: 'block' }}
      />
      {state === 'failed' && !disabled ? (
        <>
          {/*
            **Stills, not a black rectangle.** Until 2026-08-25 a tile that could
            not play live video covered itself with an opaque panel — hiding the
            still the appliance was already able to serve underneath it. That is
            the wrong trade for a plant: a five-second-old picture of the line,
            labelled as five seconds old, is worth more than a black square, and
            an operator who can see the scene can at least tell a stopped
            machine from a stopped stream.

            It has to be *labelled*, which is why the bar below carries the age
            and why it refreshes rather than freezing: a stale frame passed off
            as live is the one outcome worse than no frame at all.
          */}
          {still ? (
            <img
              src={still.url}
              alt=""
              onLoad={(event) => {
                const element = event.currentTarget;
                if (element.naturalWidth > 0) onSize?.(element.naturalWidth, element.naturalHeight);
              }}
              style={{
                position: 'absolute',
                inset: 0,
                width: '100%',
                height: '100%',
                objectFit: fit,
                display: 'block',
              }}
            />
          ) : (
            <div className="tile__signal" />
          )}
          {/*
            Signalling comes through the authenticated API on this origin, so if
            the console loaded at all, it reached the gateway. What failed is the
            *media*: either ICE could not connect the browser straight to the
            appliance's webrtc_port (a firewall, a different subnet), or the two
            ends share no video codec — which is what an H.265 camera and a
            browser without H.265 do (docs/PRODUCTION_READINESS.md §10.3.1).
            Both are "no live media", and neither is "the gateway is down".
          */}
          <div
            className="tile__degraded"
            title={
              still
                ? `Live video is not playing. The appliance is serving stills every ${STILL_REFRESH_MS / 1000} seconds.`
                : 'No live media and no still is available for this camera.'
            }
          >
            {still ? `STILLS · ${describeAge(still.ageSec)}` : 'NO PICTURE'}
          </div>
        </>
      ) : null}
    </div>
  );
}

interface Still {
  url: string;
  /** Seconds between capture and now, when the appliance told us. */
  ageSec: number | null;
}

/**
 * One still, as a blob URL, plus how old the appliance says it is.
 *
 * `fetch` rather than putting the URL straight on an `<img>`: the age lives in
 * a response header (`X-Argus-Frame-Age`), and a tile that cannot say how old
 * its picture is has no business showing one.
 */
async function fetchStill(stream: string): Promise<Still | null> {
  try {
    const response = await fetch(frameUrl(stream), { credentials: 'same-origin' });
    if (!response.ok) return null;
    const blob = await response.blob();
    if (blob.size === 0) return null;
    const header = response.headers.get('X-Argus-Frame-Age');
    const ageSec = header === null ? null : Number.parseFloat(header);
    return {
      url: URL.createObjectURL(blob),
      ageSec: ageSec === null || Number.isNaN(ageSec) ? null : ageSec,
    };
  } catch {
    return null;
  }
}

function describeAge(ageSec: number | null): string {
  if (ageSec === null) return 'JUST NOW';
  if (ageSec < 1) return 'JUST NOW';
  if (ageSec < 90) return `${Math.round(ageSec)}s OLD`;
  return `${Math.round(ageSec / 60)}m OLD`;
}
