/**
 * The frame everything hangs in: masthead, numbered nav, the current screen,
 * and the two full-screen modes (Wall TV, ⌘K).
 *
 * Live state is polled here rather than in the wall, because the masthead's
 * engine light, the nav badges, the TV mode and the command palette all need
 * the same answer, and one poll for the app is one poll for the appliance.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { api } from './api/client';
import { Masthead, Nav } from './components/Masthead';
import { CommandPalette } from './components/CommandPalette';
import { TvWall } from './components/TvWall';
import { ProcessingBanner, PauseConfirm } from './components/Processing';
import { ErrorBar } from './components/marks';
import { useClock, useKeydown, usePoll } from './lib/hooks';
import { screensFor, useRouter } from './lib/router';
import { useTheme } from './lib/theme';
import { shiftForDate } from './lib/shifts';
import { useAppState } from './state/AppState';
import { useSession } from './state/Session';
import { LiveWall } from './screens/LiveWall';
import { AlertFeed } from './screens/AlertFeed';
import { Triage } from './screens/Triage';
import { SearchScreen } from './screens/SearchScreen';
import { PolicyStudio } from './screens/PolicyStudio';
import { CamerasAndZones } from './screens/CamerasAndZones';
import { Analytics } from './screens/Analytics';
import { ModelHealth } from './screens/ModelHealth';
import { Settings } from './screens/Settings';

/** The live wall's data. Two seconds matches the engine's stats cadence — any
 *  faster and the same numbers come back. */
const LIVE_POLL_MS = 2000;

export function App() {
  const { config, configError, policyLabels, arrivals } = useAppState();
  const { user, can } = useSession();
  const { route, go, setParam } = useRouter();
  const [theme, toggleTheme] = useTheme();
  const [tv, setTv] = useState(false);
  const [commands, setCommands] = useState(false);
  const now = useClock();

  // What this role can open. The API refuses the rest whatever the nav says —
  // this is so a SYSTEM monitoring account sees the two screens it can read
  // rather than nine, seven of which would be error bars.
  const screens = useMemo(() => screensFor(user?.permissions ?? []), [user?.permissions]);
  const watching = can('view');

  // Polling only the things this session may read. A SYSTEM account polling
  // /api/live/state every two seconds would be two 403s a second in the log
  // and an error bar that never clears.
  const live = usePoll(() => api.liveState(), watching ? LIVE_POLL_MS : 0, [watching]);
  // `limit: 1` on purpose: the badge needs a count, and the full queue is a
  // 241 KB payload this poll was fetching every 15 s to take its `.length` —
  // which was also the wrong number, being the page size and not the backlog.
  const triage = usePoll(() => api.triage({ limit: 1 }), watching ? 15000 : 0, [watching]);

  const cameras = live.data?.cameras ?? [];
  const pending = triage.data?.total ?? 0;
  const shift = useMemo(
    () => live.data?.shift ?? shiftForDate(config?.site.shifts ?? [], now),
    [live.data?.shift, config?.site.shifts, now],
  );

  // The appliance pause switch. State rides on the live poll above; actuation is
  // ADMIN-only (the API refuses it otherwise, this only hides the button). The
  // POST returns immediately — resuming does not wait for the VLM — so a reload
  // of the live poll is enough to pick up the transition to "starting".
  const processing = live.data?.processing ?? null;
  const canControlProcessing = can('admin');
  const [processingBusy, setProcessingBusy] = useState(false);
  const [processingError, setProcessingError] = useState<string | null>(null);
  const [confirmPause, setConfirmPause] = useState(false);

  const setProcessing = useCallback(
    async (enabled: boolean) => {
      setProcessingBusy(true);
      setProcessingError(null);
      try {
        await api.setProcessing(enabled);
        live.reload();
      } catch (error) {
        setProcessingError(error instanceof Error ? error.message : String(error));
      } finally {
        setProcessingBusy(false);
      }
    },
    [live],
  );

  useKeydown((event) => {
    const typing =
      event.target instanceof HTMLElement &&
      ['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target.tagName);
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
      event.preventDefault();
      setCommands((open) => !open);
      return;
    }
    if (event.key === 'Escape') {
      // One key closes whatever is on top, outermost last — the design's own
      // ordering: palette, then TV, then camera focus.
      if (commands) setCommands(false);
      else if (tv) setTv(false);
      else if (route.params.get('cam')) setParam('cam', null);
      return;
    }
    if (typing || event.metaKey || event.ctrlKey || event.altKey) return;
    // 1-9 jump between screens, matching the numbers printed in the nav — which
    // is why this indexes the *visible* list rather than a hard-coded nine.
    const digit = Number(event.key);
    if (Number.isInteger(digit) && digit >= 1 && digit <= 9) {
      const target = screens[digit - 1];
      if (target) go(target);
    }
  });

  // A role that cannot open the screen in the URL — a bookmarked Live Wall
  // opened by a SYSTEM account, or a demotion mid-session — lands on the first
  // screen it *can* open instead of on a wall of 403s.
  useEffect(() => {
    const fallback = screens[0];
    if (fallback && !screens.includes(route.screen)) go(fallback);
  }, [screens, route.screen, go]);

  const focusCamera = (cameraId: string) => {
    go('live', { cam: cameraId });
  };

  if (tv) {
    return (
      <TvWall
        cameras={cameras}
        alerts={arrivals}
        policyLabels={policyLabels}
        now={now}
        engineRunning={live.data?.engine.running ?? false}
        paused={processing?.status === 'paused'}
        onExit={() => setTv(false)}
      />
    );
  }

  return (
    <div className="app">
      <div className="app__column">
        <div className="masthead">
          <Masthead
            plant={config?.site.name ?? 'Argus Appliance'}
            now={now}
            shift={shift}
            engineRunning={live.data?.engine.running ?? false}
            engineReason={live.data?.engine.reason ?? null}
            processing={processing}
            canControlProcessing={canControlProcessing}
            processingBusy={processingBusy}
            onPauseProcessing={() => setConfirmPause(true)}
            onResumeProcessing={() => void setProcessing(true)}
            theme={theme}
            onToggleTheme={toggleTheme}
            onOpenCommands={() => setCommands(true)}
            onEnterTv={() => setTv(true)}
          />
          <Nav
            active={route.screen}
            screens={screens}
            pending={pending}
            unseen={arrivals.length}
            onNavigate={(screen) => go(screen)}
          />
        </div>

        <ProcessingBanner
          processing={processing}
          canControl={canControlProcessing}
          pending={processingBusy}
          error={processingError}
          onResume={() => void setProcessing(true)}
        />

        <ErrorBar error={configError ?? live.error} />

        <div className="app__screen">
          {route.screen === 'live' ? (
            <LiveWall
              live={live.data}
              loading={live.loading}
              focusId={route.params.get('cam')}
              onFocus={(cameraId) => setParam('cam', cameraId)}
              now={now}
            />
          ) : null}
          {route.screen === 'alerts' ? <AlertFeed /> : null}
          {route.screen === 'triage' ? <Triage onQueueChanged={triage.reload} /> : null}
          {route.screen === 'search' ? <SearchScreen /> : null}
          {route.screen === 'policies' ? <PolicyStudio /> : null}
          {route.screen === 'cameras' ? <CamerasAndZones live={live.data} onReload={live.reload} /> : null}
          {route.screen === 'analytics' ? <Analytics /> : null}
          {route.screen === 'health' ? <ModelHealth /> : null}
          {route.screen === 'settings' ? <Settings /> : null}
        </div>
      </div>

      {commands ? (
        <CommandPalette
          cameras={cameras}
          screens={screens}
          pending={pending}
          onNavigate={(screen) => go(screen)}
          onFocusCamera={focusCamera}
          onEnterTv={() => setTv(true)}
          onClose={() => setCommands(false)}
        />
      ) : null}

      {confirmPause ? (
        <PauseConfirm
          pending={processingBusy}
          onCancel={() => setConfirmPause(false)}
          onConfirm={() => {
            setConfirmPause(false);
            void setProcessing(false);
          }}
        />
      ) : null}
    </div>
  );
}
