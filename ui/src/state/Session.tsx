/**
 * Who is signed in, and what happens when that stops being true.
 *
 * Sits *outside* `AppStateProvider` on purpose: config, the policy list and the
 * SSE connection all need a session, so they must not exist before there is
 * one. Signing out therefore unmounts them, which is also what closes the
 * event stream — an EventSource left open against a 401 reconnects forever.
 *
 * There is no token in this file. The session lives in an HttpOnly cookie the
 * browser attaches on its own; what the console holds is the *description* of
 * the session (`/api/auth/session`), which is public information about the
 * person already holding it.
 */

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import { ApiError, api, setUnauthenticatedHandler } from '../api/client';
import type { Permission, Session } from '../api/types';

interface SessionValue {
  user: Session | null;
  /** True until the first `/api/auth/session` answers — the moment before the
   *  console knows whether to draw a login form or the plant. */
  loading: boolean;
  /** Why the login screen is being shown, when it is not simply "you are not
   *  signed in": an expired session, or an appliance with no accounts yet. */
  notice: string | null;
  /** `no_accounts` gets its own screen: there is no password that would work,
   *  and the right thing to show is the command to run on the box. */
  code: string | null;
  signIn: (username: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
  can: (permission: Permission) => boolean;
}

const SessionContext = createContext<SessionValue | null>(null);

export function SessionProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);
  const [notice, setNotice] = useState<string | null>(null);
  const [code, setCode] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const { user: current } = await api.session();
      setUser(current);
      setNotice(null);
      setCode(null);
    } catch (error) {
      setUser(null);
      if (error instanceof ApiError && error.status === 401) {
        // The 401 body carries the reason. A freshly imaged appliance says
        // which command creates the first account; anything else is simply
        // "not signed in", which needs no message at all.
        const header = error.detail.includes('argus user add') ? 'no_accounts' : null;
        setCode(header);
        setNotice(header ? error.detail : null);
      } else {
        setNotice(error instanceof Error ? error.message : String(error));
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    // One listener for the whole console. A session can end between two polls
    // — it expired, an admin disabled the account, a laptop was revoked — and
    // when it does every screen starts failing at once. Without this the
    // operator gets nine error bars for one cause and no way back.
    setUnauthenticatedHandler((reason, detail) => {
      setUser((current) => {
        if (current === null) return null;
        setCode(reason);
        setNotice(
          reason === 'no_accounts'
            ? detail
            : 'Your session ended. Sign in to continue.',
        );
        return null;
      });
    });
    return () => setUnauthenticatedHandler(null);
  }, []);

  const signIn = useCallback(async (username: string, password: string) => {
    const { user: current } = await api.login(username, password);
    setUser(current);
    setNotice(null);
    setCode(null);
  }, []);

  const signOut = useCallback(async () => {
    try {
      await api.logout();
    } finally {
      // Even a failed logout clears the console: whatever the server did, this
      // browser is done showing the plant.
      setUser(null);
      setNotice(null);
      setCode(null);
    }
  }, []);

  const value = useMemo<SessionValue>(
    () => ({
      user,
      loading,
      notice,
      code,
      signIn,
      signOut,
      can: (permission: Permission) => user?.permissions.includes(permission) ?? false,
    }),
    [user, loading, notice, code, signIn, signOut],
  );

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSession(): SessionValue {
  const value = useContext(SessionContext);
  if (!value) throw new Error('useSession used outside SessionProvider');
  return value;
}
