/**
 * A camera's picture with its overlays pinned to the picture, not to the tile.
 *
 * This exists because of a bug you only see with real cameras. Zone polygons
 * and track boxes are normalized to the *frame*, so they only land correctly
 * when the box they are drawn in is exactly the box the video occupies. A tile
 * whose shape is decided by the layout — Wall TV's grid, or a 4:3 camera in a
 * 16:9 tile — is not that box, and the overlay drifts: the zone outline sits
 * beside the zone. On a safety wall that is worse than drawing nothing.
 *
 * So the video is letterboxed inside the tile, and the overlays are placed in a
 * box measured to match it. The arithmetic is done in JS rather than CSS
 * because "contain a box of a given aspect ratio inside another box" has no
 * reliable pure-CSS spelling that also gives the overlays a containing block.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { CameraVideo } from './CameraVideo';

interface Props {
  stream: string;
  disabled?: boolean;
  /** Used until the video reports its own dimensions. */
  fallbackAspect?: number;
  /** The measured box, for callers that need to map clicks onto the frame —
   *  the zone editor turns a mouse position into normalized coordinates. */
  boxRef?: React.Ref<HTMLDivElement>;
  children?: ReactNode;
}

interface Rect {
  width: number;
  height: number;
  left: number;
  top: number;
}

function contain(outerW: number, outerH: number, aspect: number): Rect {
  if (outerW <= 0 || outerH <= 0 || !Number.isFinite(aspect) || aspect <= 0) {
    return { width: outerW, height: outerH, left: 0, top: 0 };
  }
  let width = outerW;
  let height = outerW / aspect;
  if (height > outerH) {
    height = outerH;
    width = outerH * aspect;
  }
  return { width, height, left: (outerW - width) / 2, top: (outerH - height) / 2 };
}

export function VideoFrame({
  stream,
  disabled = false,
  fallbackAspect = 16 / 9,
  boxRef,
  children,
}: Props) {
  const holder = useRef<HTMLDivElement>(null);
  const [aspect, setAspect] = useState(fallbackAspect);
  const [outer, setOuter] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const element = holder.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      if (!entry) return;
      const { width, height } = entry.contentRect;
      setOuter({ width, height });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const onSize = useCallback((width: number, height: number) => {
    if (width > 0 && height > 0) setAspect(width / height);
  }, []);

  const rect = contain(outer.width, outer.height, aspect);

  return (
    <div ref={holder} style={{ position: 'absolute', inset: 0, overflow: 'hidden' }}>
      <div
        ref={boxRef}
        style={{
          position: 'absolute',
          left: rect.left,
          top: rect.top,
          width: rect.width,
          height: rect.height,
        }}
      >
        <CameraVideo stream={stream} disabled={disabled} fit="fill" onSize={onSize} />
        {children}
      </div>
    </div>
  );
}
