/**
 * The nameplate and the numbered nav.
 *
 * Two details carry meaning rather than style. The breathing dot next to
 * ON-PREM is the engine's liveness, not an ornament: if `argus run` stops, it
 * turns red and stops breathing, because a wall of still pictures with a green
 * light is the worst thing this product could show. And the badge on Review
 * Triage is the real unreviewed count — the number the shift is judged on.
 */

import { useEffect, useRef, useState } from 'react';
import type { Screen } from '../lib/router';
import { SCREEN_LABELS } from '../lib/router';
import { clockTime, mastheadDate } from '../lib/format';
import type { Processing, Shift } from '../api/types';
import { shiftWindowLabel } from '../lib/shifts';
import { useSession } from '../state/Session';
import { ProcessingToggle } from './Processing';

interface Props {
  plant: string;
  now: Date;
  shift: Shift | null;
  engineRunning: boolean;
  engineReason: string | null;
  processing: Processing | null;
  /** ADMIN — mount the pause/resume toggle only then. */
  canControlProcessing: boolean;
  processingBusy: boolean;
  onPauseProcessing: () => void;
  onResumeProcessing: () => void;
  theme: 'light' | 'dark';
  onToggleTheme: () => void;
  onOpenCommands: () => void;
  onEnterTv: () => void;
}

/** The ON-PREM engine light. Paused is a deliberate state, not a fault, so it is
 *  a calm amber rather than the red of a crashed engine — the masthead must not
 *  cry "down" at a box that was switched off on purpose. */
function onpremLook(processing: Processing | null, engineRunning: boolean, engineReason: string | null) {
  if (processing?.status === 'paused') {
    return { dot: 'live-dot live-dot--paused', title: 'Processing paused — engine intentionally stopped, GPU idle' };
  }
  if (engineRunning) {
    return { dot: 'live-dot breathe', title: 'Vision engine running — all processing on this box' };
  }
  if (processing?.status === 'starting') {
    return { dot: 'live-dot live-dot--paused', title: 'Resuming — the vision engine is coming up' };
  }
  return { dot: 'live-dot live-dot--down', title: engineReason ?? 'Vision engine not running' };
}

export function Masthead({
  plant,
  now,
  shift,
  engineRunning,
  engineReason,
  processing,
  canControlProcessing,
  processingBusy,
  onPauseProcessing,
  onResumeProcessing,
  theme,
  onToggleTheme,
  onOpenCommands,
  onEnterTv,
}: Props) {
  const onprem = onpremLook(processing, engineRunning, engineReason);
  return (
    <div className="masthead__top">
      <div className="masthead__title">ARGUS</div>
      <div className="masthead__strap">The Plant Observer · {plant}</div>
      <div className="masthead__tools">
        <span className="masthead__clock">
          {mastheadDate(now)}
          {shift ? ` · SHIFT ${shift.id} ${shiftWindowLabel(shift)}` : ''} · {clockTime(now)}
        </span>
        <span className="masthead__onprem" title={onprem.title}>
          <span className={onprem.dot} />
          ON-PREM
        </span>
        {canControlProcessing ? (
          <ProcessingToggle
            processing={processing}
            pending={processingBusy}
            onPause={onPauseProcessing}
            onResume={onResumeProcessing}
          />
        ) : null}
        <button className="chip-button" onClick={onOpenCommands} title="Command palette (⌘K)">
          ⌘K
        </button>
        <button className="chip-button" onClick={onToggleTheme}>
          ◑ {theme === 'dark' ? 'Dark' : 'Light'}
        </button>
        <button className="chip-button" onClick={onEnterTv} title="Fullscreen wall for a control-room screen">
          ⛶ WALL TV
        </button>
        <WhoAmI />
      </div>
    </div>
  );
}

/** Who this browser is, and the way out.
 *
 *  In the masthead rather than buried in Settings because it answers a question
 *  a shared control-room terminal raises constantly — "whose verdicts am I
 *  about to record?" — and because the role beside the name is now load-bearing:
 *  it is why some screens are missing from the nav. */
function WhoAmI() {
  const { user, signOut } = useSession();
  const [open, setOpen] = useState(false);
  const holder = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const dismiss = (event: MouseEvent) => {
      if (!holder.current?.contains(event.target as Node)) setOpen(false);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', dismiss);
    document.addEventListener('keydown', escape);
    return () => {
      document.removeEventListener('mousedown', dismiss);
      document.removeEventListener('keydown', escape);
    };
  }, [open]);

  if (!user) return null;

  return (
    <div className="whoami" ref={holder}>
      <button className="whoami__button" onClick={() => setOpen((shown) => !shown)}>
        {initials(user.display_name || user.username)}
        <span className="whoami__role">{user.role}</span>
      </button>
      {open ? (
        <div className="whoami__menu">
          <div>
            <div className="whoami__who">{user.display_name}</div>
            <div className="whoami__detail">
              {user.username} · {user.role}
              <br />
              Session ends {new Date(user.expires_at).toLocaleString()}
            </div>
          </div>
          <button className="whoami__signout" onClick={() => void signOut()}>
            Sign out
          </button>
        </div>
      ) : null}
    </div>
  );
}

/** "J. Whitfield" -> "JW"; "ops" -> "OP". The design's initials, from whatever
 *  the account actually has. */
export function initials(name: string): string {
  const parts = name.split(/[\s._-]+/).filter(Boolean);
  const first = parts[0] ?? '?';
  const last = parts[parts.length - 1] ?? first;
  if (parts.length >= 2) return (first.charAt(0) + last.charAt(0)).toUpperCase();
  return first.slice(0, 2).toUpperCase();
}

interface NavProps {
  active: Screen;
  /** The screens this role's session can actually load. A SYSTEM monitoring
   *  account gets two, not nine and seven error bars. */
  screens: readonly Screen[];
  pending: number;
  unseen: number;
  onNavigate: (screen: Screen) => void;
}

export function Nav({ active, screens, pending, unseen, onNavigate }: NavProps) {
  return (
    <div className="nav">
      {screens.map((screen, index) => {
        const badge = screen === 'triage' ? pending : screen === 'alerts' ? unseen : 0;
        return (
          <button
            key={screen}
            className={`nav__tab ${screen === active ? 'nav__tab--active' : ''}`}
            onClick={() => onNavigate(screen)}
          >
            {/* Numbered by position in *this* nav, so the printed number and
                the digit key that jumps there never disagree — which they would
                if the numbering came from the full nine-screen list. */}
            <span className="nav__num">{String(index + 1).padStart(2, '0')}</span>
            <span>{SCREEN_LABELS[screen]}</span>
            {/* A backlog is a *signal* in the nav, not a readout: past a few
                hundred the exact figure changes nothing an operator would do,
                and five digits in a pill turn the tab label into a number.
                The real count is on the screen the tab opens. */}
            {badge > 0 ? (
              <span className="nav__badge" title={`${badge.toLocaleString()} pending`}>
                {badge > 999 ? '999+' : badge}
              </span>
            ) : null}
          </button>
        );
      })}
    </div>
  );
}
