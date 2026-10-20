/**
 * The three things every screen needs, fetched once for the whole app.
 *
 * * **config** — the appliance's own description of itself: plant name, shift
 *   rota, which phases are enabled. Loaded once; a change to argus.yaml is a
 *   restart of the API anyway.
 * * **the event stream** — one SSE connection for the entire SPA. Nine screens
 *   opening their own would be nine database polls per interval, and the "N NEW
 *   ALERTS" pill only works if arrivals are held in one place rather than
 *   racing each screen's own list.
 * * **policy labels** — every event carries a `detector_name`; the human name
 *   lives on the policy. Resolved here so no screen has to join it twice.
 */

import { createContext, useContext, useEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { api } from '../api/client';
import type { AppConfig, ArgusEvent, Policy } from '../api/types';
import { useResource } from '../lib/hooks';
import { useSession } from './Session';

interface AppStateValue {
  config: AppConfig | null;
  configError: string | null;
  policies: Policy[];
  policyLabels: Record<string, string>;
  reloadPolicies: () => void;
  /** Events that arrived over SSE since the app started, newest first. */
  arrivals: ArgusEvent[];
  /**
   * Rows the appliance has *changed* since this browser last fetched them,
   * by id — an ongoing condition observed again or closed, or a clip landing.
   *
   * Kept apart from `arrivals` because the two are opposite interactions. An
   * arrival is new information that must not move the list under the reader,
   * so it waits behind the "N NEW ALERTS" pill. An update is the same card
   * saying something truer about itself, and holding *that* back would leave a
   * card reading "1 observation" while the appliance is on its fortieth. So
   * screens merge these straight over the row they already hold, in place,
   * without re-sorting: the card changes, the list does not move.
   */
  updates: Record<number, ArgusEvent>;
  /** True while the SSE connection is established. */
  streaming: boolean;
  /** Consume arrivals — the "show me the new ones" click. */
  drainArrivals: () => ArgusEvent[];
  /** Bump after a verdict so screens holding derived counts refresh. */
  reviewSerial: number;
  noteReview: () => void;
}

const AppStateContext = createContext<AppStateValue | null>(null);

const MAX_ARRIVALS = 200;

export function AppStateProvider({ children }: { children: ReactNode }) {
  // A SYSTEM monitoring account holds `system` and nothing else — it may read
  // the appliance's description of itself and its health, and no part of what
  // the cameras saw. Both of the calls below and the event stream are gated on
  // that, so its console loads clean instead of loading three 403s and an
  // EventSource that reconnects to one forever.
  const { can } = useSession();
  const watching = can('view');
  const config = useResource(() => api.config(), []);
  const policies = useResource(
    () => (watching ? api.policies() : Promise.resolve({ policies: [] })),
    [watching],
  );
  const [arrivals, setArrivals] = useState<ArgusEvent[]>([]);
  const [updates, setUpdates] = useState<Record<number, ArgusEvent>>({});
  const [streaming, setStreaming] = useState(false);
  const [reviewSerial, setReviewSerial] = useState(0);
  const arrivalsRef = useRef<ArgusEvent[]>([]);
  arrivalsRef.current = arrivals;

  useEffect(() => {
    if (!watching) return;
    // EventSource reconnects on its own and resumes from Last-Event-ID, which
    // the API honours — so a dropped connection loses nothing. It also carries
    // the session cookie automatically, being same-origin; when the session
    // ends this provider is unmounted (see main.tsx) and the cleanup below is
    // what stops it reconnecting into a 401 behind the login screen.
    const source = new EventSource('/api/events/stream');
    source.addEventListener('hello', () => setStreaming(true));
    source.addEventListener('event', (message) => {
      try {
        const event = JSON.parse((message as MessageEvent<string>).data) as ArgusEvent;
        setArrivals((current) => [event, ...current].slice(0, MAX_ARRIVALS));
      } catch {
        /* a malformed frame must not kill the stream */
      }
    });
    // A row the appliance changed after it was first sent. Two things send one:
    //
    //  * a clip landing. An alert is streamed the moment it is decided, ~8 s
    //    before its clip finishes being cut — the video covers seconds that
    //    have not happened yet. Without merging this, a live-arrived alert
    //    would show "no clip" forever, and the Live Wall (which prefers
    //    arrivals over its polled list) would keep serving that stale copy to
    //    every screen reading from it.
    //
    //  * an ongoing condition being observed again or closed. The row is the
    //    condition, so "still true, 12 observations, ongoing 3h" arrives as the
    //    same row with different numbers rather than as a new alert. That is
    //    the whole reason the feed no longer grows a card every cooldown.
    //
    // Merged into both places: over `arrivals`, so a card the reader has not
    // shown yet is still current when they do, and into `updates`, which is
    // what a screen showing a *fetched* page merges over its own rows.
    source.addEventListener('update', (message) => {
      try {
        const event = JSON.parse((message as MessageEvent<string>).data) as ArgusEvent;
        setArrivals((current) =>
          current.map((held) => (held.id === event.id ? event : held)),
        );
        setUpdates((current) => ({ ...current, [event.id]: event }));
      } catch {
        /* as above */
      }
    });
    source.addEventListener('error', () => setStreaming(false));
    source.addEventListener('open', () => setStreaming(true));
    return () => source.close();
  }, [watching]);

  const value = useMemo<AppStateValue>(() => {
    const list = policies.data?.policies ?? [];
    return {
      config: config.data,
      configError: config.error,
      policies: list,
      policyLabels: Object.fromEntries(list.map((policy) => [policy.id, policy.label])),
      reloadPolicies: policies.reload,
      arrivals,
      updates,
      streaming,
      drainArrivals: () => {
        const held = arrivalsRef.current;
        setArrivals([]);
        return held;
      },
      reviewSerial,
      noteReview: () => setReviewSerial((serial) => serial + 1),
    };
  }, [config.data, config.error, policies.data, policies.reload, arrivals, updates, streaming,
      reviewSerial]);

  return <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>;
}

export function useAppState(): AppStateValue {
  const value = useContext(AppStateContext);
  if (!value) throw new Error('useAppState used outside AppStateProvider');
  return value;
}
