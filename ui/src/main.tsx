/**
 * Entry point.
 *
 * Fonts are imported here as packages, not linked from a CDN: this appliance is
 * sold on "nothing leaves the plant network", and a stylesheet from
 * fonts.googleapis.com would make that false on the very first paint. Only the
 * weights the design uses are pulled in — Newsreader 400/500/600, Instrument
 * Serif 400, Space Mono 400/700 — and only the latin subset.
 */

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

import '@fontsource/newsreader/latin-400.css';
import '@fontsource/newsreader/latin-500.css';
import '@fontsource/newsreader/latin-600.css';
import '@fontsource/instrument-serif/latin-400.css';
import '@fontsource/space-mono/latin-400.css';
import '@fontsource/space-mono/latin-700.css';

import './styles/tokens.css';
import './styles/base.css';
import './styles/components.css';
import './styles/screens.css';

import { App } from './App';
import { Login, Splash } from './screens/Login';
import { AppStateProvider } from './state/AppState';
import { SessionProvider, useSession } from './state/Session';

/**
 * The gate.
 *
 * `AppStateProvider` is *inside* the signed-in branch, not around it, and that
 * placement is the whole session-expiry story: config, the policy list and the
 * one SSE connection all live in it, so losing the session unmounts them.
 * Otherwise an EventSource would sit there reconnecting to a 401 forever behind
 * a login form.
 */
function Root() {
  const { user, loading } = useSession();
  if (loading) return <Splash />;
  if (!user) return <Login />;
  return (
    <AppStateProvider>
      <App />
    </AppStateProvider>
  );
}

const container = document.getElementById('root');
if (!container) throw new Error('#root missing from index.html');

createRoot(container).render(
  <StrictMode>
    <SessionProvider>
      <Root />
    </SessionProvider>
  </StrictMode>,
);
