/**
 * The small set of data hooks every screen shares.
 *
 * No data-fetching library: the whole surface is a dozen GETs against one
 * origin, and a poll with an abort on unmount is the entire requirement. What
 * matters more than caching here is that a failed poll leaves the *last good*
 * data on screen — an operations wall that blanks itself because one request
 * timed out is worse than one showing a five-second-old count.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { ApiError } from '../api/client';

export interface Loadable<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  reload: () => void;
}

function describe(error: unknown): string {
  if (error instanceof ApiError) return `${error.url}: ${error.detail}`;
  if (error instanceof Error) return error.message;
  return String(error);
}

/** Fetch once, and again whenever `deps` change or `reload()` is called. */
export function useResource<T>(fetcher: () => Promise<T>, deps: unknown[] = []): Loadable<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [nonce, setNonce] = useState(0);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  useEffect(() => {
    setLoading(true);
    fetcher()
      .then((value) => {
        if (!alive.current) return;
        setData(value);
        setError(null);
      })
      .catch((cause: unknown) => {
        if (alive.current) setError(describe(cause));
      })
      .finally(() => {
        if (alive.current) setLoading(false);
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce]);

  const reload = useCallback(() => setNonce((value) => value + 1), []);
  return { data, error, loading, reload };
}

/**
 * Poll on an interval, keeping the previous value through a failure.
 *
 * Polling pauses while the tab is hidden: twelve WebRTC tiles and a two-second
 * poll behind a screensaver is a box working for nobody.
 *
 * An interval of `0` means "do not poll at all" — the caller's way of saying
 * this session may not read this endpoint. Without it, a SYSTEM monitoring
 * account would put two 403s a second into the appliance's log for a screen it
 * is not even being shown.
 */
export function usePoll<T>(
  fetcher: () => Promise<T>,
  intervalMs: number,
  deps: unknown[] = [],
): Loadable<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    let alive = true;
    let timer: number | undefined;

    if (intervalMs <= 0) {
      setLoading(false);
      return;
    }

    const tick = async () => {
      if (document.hidden) {
        timer = window.setTimeout(tick, intervalMs);
        return;
      }
      try {
        const value = await fetcher();
        if (!alive) return;
        setData(value);
        setError(null);
      } catch (cause) {
        if (alive) setError(describe(cause));
      } finally {
        if (alive) {
          setLoading(false);
          timer = window.setTimeout(tick, intervalMs);
        }
      }
    };

    void tick();
    const onVisible = () => {
      if (!document.hidden) {
        window.clearTimeout(timer);
        void tick();
      }
    };
    document.addEventListener('visibilitychange', onVisible);
    return () => {
      alive = false;
      window.clearTimeout(timer);
      document.removeEventListener('visibilitychange', onVisible);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, intervalMs, nonce]);

  const reload = useCallback(() => setNonce((value) => value + 1), []);
  return { data, error, loading, reload };
}

/** A ticking clock for the masthead. One interval for the whole app. */
export function useClock(): Date {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(timer);
  }, []);
  return now;
}

/** Window-level keydown, with the handler kept fresh without re-binding. */
export function useKeydown(handler: (event: KeyboardEvent) => void): void {
  const ref = useRef(handler);
  ref.current = handler;
  useEffect(() => {
    const listener = (event: KeyboardEvent) => ref.current(event);
    window.addEventListener('keydown', listener);
    return () => window.removeEventListener('keydown', listener);
  }, []);
}

/** True while the element is on screen. Tiles use this to hold their WebRTC
 *  connection open only while somebody can actually see them. */
export function useOnScreen<T extends Element>(ref: React.RefObject<T>, rootMargin = '200px'): boolean {
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new IntersectionObserver(
      (entries) => setVisible(entries.some((entry) => entry.isIntersecting)),
      { rootMargin },
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, [ref, rootMargin]);
  return visible;
}
