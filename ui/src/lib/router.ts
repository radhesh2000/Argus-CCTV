/**
 * A router in forty lines, because that is the size of the problem.
 *
 * Nine screens, no nested routes, no loaders, no code splitting worth the
 * complexity on a LAN. Real URLs rather than hash routes so an operator can
 * bookmark the alert feed and a supervisor can be sent a link to one camera —
 * the API serves index.html for any path without a file suffix, which is what
 * makes that reload work (see SpaFiles in argus/api/app.py).
 */

import { useCallback, useEffect, useState } from 'react';
import type { Permission } from '../api/types';

export const SCREENS = [
  'live',
  'alerts',
  'triage',
  'search',
  'policies',
  'cameras',
  'analytics',
  'health',
  'settings',
] as const;

export type Screen = (typeof SCREENS)[number];

export const SCREEN_LABELS: Record<Screen, string> = {
  live: 'Live Wall',
  alerts: 'Alert Feed',
  triage: 'Review Triage',
  search: 'Search',
  policies: 'Policy Studio',
  cameras: 'Cameras & Zones',
  analytics: 'Analytics',
  health: 'Model Health',
  settings: 'Settings',
};

/**
 * The permission each screen's data needs — the client-side echo of the
 * matrix in `argus/api/auth.py`.
 *
 * It is *only* an echo. Nothing here is a security control: the API refuses
 * the request whatever the nav does. What this buys is a console that does not
 * offer a SYSTEM monitoring account eight screens of error bars — which is what
 * "read-only health and metrics" looks like if the nav ignores the role.
 */
export const SCREEN_PERMISSION: Record<Screen, Permission> = {
  live: 'view',
  alerts: 'view',
  triage: 'view',
  search: 'view',
  policies: 'view',
  cameras: 'view',
  analytics: 'view',
  health: 'system',
  settings: 'system',
};

export function screensFor(permissions: readonly Permission[]): Screen[] {
  return SCREENS.filter((screen) => permissions.includes(SCREEN_PERMISSION[screen]));
}

export interface Route {
  screen: Screen;
  params: URLSearchParams;
}

function parse(): Route {
  const path = window.location.pathname.replace(/^\/+|\/+$/g, '');
  const screen = (SCREENS as readonly string[]).includes(path) ? (path as Screen) : 'live';
  return { screen, params: new URLSearchParams(window.location.search) };
}

function href(screen: Screen, params?: Record<string, string | number | null | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params ?? {})) {
    if (value !== null && value !== undefined && value !== '') search.set(key, String(value));
  }
  const query = search.toString();
  return `/${screen === 'live' ? '' : screen}${query ? `?${query}` : ''}`;
}

export function useRouter() {
  const [route, setRoute] = useState<Route>(parse);

  useEffect(() => {
    const onPop = () => setRoute(parse());
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);

  const go = useCallback(
    (screen: Screen, params?: Record<string, string | number | null | undefined>) => {
      const url = href(screen, params);
      if (url !== window.location.pathname + window.location.search) {
        window.history.pushState({}, '', url);
      }
      setRoute(parse());
    },
    [],
  );

  /** Change one query parameter without touching the screen or adding history
   *  noise — focusing a camera should not cost four Back presses to undo. */
  const setParam = useCallback((key: string, value: string | null) => {
    const params = new URLSearchParams(window.location.search);
    if (value === null) params.delete(key);
    else params.set(key, value);
    const query = params.toString();
    window.history.replaceState({}, '', window.location.pathname + (query ? `?${query}` : ''));
    setRoute(parse());
  }, []);

  return { route, go, setParam };
}
