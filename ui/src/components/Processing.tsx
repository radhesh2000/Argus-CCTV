/**
 * The appliance pause switch, as the operator meets it.
 *
 * Three surfaces, one truth (`live.processing`, polled every two seconds):
 *
 *  - the **banner**, across the whole console, shown only when the box is not
 *    simply running. Paused, it says so plainly and — for an admin — offers the
 *    way back; resuming, it says the models are warming up and gets out of the
 *    way. It is deliberately not shown while running: a banner that is always
 *    there is a banner nobody reads.
 *  - the **masthead toggle**, ADMIN only, which is the *pause* affordance (the
 *    banner is the resume one) and carries the confirm step, because pausing
 *    stops a whole plant's monitoring.
 *  - the confirm **dialog**, so that stopping detection across every camera is
 *    never one stray click.
 *
 * Non-admins see the banner (so a dark wall is explained) but no controls; the
 * API refuses the write regardless, this only spares them a button that 403s.
 */

import { useEffect } from 'react';
import type { Processing } from '../api/types';

// --------------------------------------------------------------------------
// the banner
// --------------------------------------------------------------------------

interface BannerProps {
  processing: Processing | null;
  /** ADMIN — the only role the API lets actuate the switch. */
  canControl: boolean;
  pending: boolean;
  error: string | null;
  onResume: () => void;
}

export function ProcessingBanner({ processing, canControl, pending, error, onResume }: BannerProps) {
  // Nothing to say while the box is running normally — and nothing to say
  // before the first poll answers, rather than a flash of "paused".
  if (!processing || processing.status === 'running') return null;

  if (processing.status === 'starting') {
    return (
      <div className="procbanner procbanner--resuming" role="status">
        <span className="procbanner__dot procbanner__dot--pulse" />
        <div className="procbanner__text">
          <span className="procbanner__title">Resuming — models warming up…</span>
          <span className="procbanner__sub">
            The vision engine is coming back up
            {processing.vlm_required ? '; the VLM takes a few minutes to load' : ''}. Live tiles
            return as the cameras reconnect.
          </span>
        </div>
      </div>
    );
  }

  // paused
  return (
    <div className="procbanner procbanner--paused" role="alert">
      <span className="procbanner__dot" />
      <div className="procbanner__text">
        <span className="procbanner__title">
          Processing paused — cameras, detection and alerts are stopped.
        </span>
        <span className="procbanner__sub">
          The console stays up and the GPU is idle.
          {processing.updated_by ? ` Paused by ${processing.updated_by}.` : ''}{' '}
          {canControl ? 'Resume whenever you are ready.' : 'An administrator can resume it.'}
        </span>
      </div>
      {error ? <span className="procbanner__error mono">{error}</span> : null}
      {canControl ? (
        <button className="button procbanner__action" disabled={pending} onClick={onResume}>
          {pending ? 'Resuming…' : 'Resume processing'}
        </button>
      ) : null}
    </div>
  );
}

// --------------------------------------------------------------------------
// the masthead toggle
// --------------------------------------------------------------------------

interface ToggleProps {
  processing: Processing | null;
  pending: boolean;
  onPause: () => void;
  onResume: () => void;
}

/** ADMIN-only. Render decides nothing about permission — App only mounts this
 *  for an admin — so this is purely the control and its confirm step. */
export function ProcessingToggle({ processing, pending, onPause, onResume }: ToggleProps) {
  if (!processing) return null;
  const { status } = processing;
  const running = status === 'running';
  const label = status === 'paused' ? 'Paused' : status === 'starting' ? 'Resuming' : 'Running';
  const dotClass =
    status === 'running'
      ? 'proc-dot proc-dot--live breathe'
      : status === 'paused'
        ? 'proc-dot proc-dot--paused'
        : 'proc-dot proc-dot--warming';

  const busy = pending || status === 'starting';
  const onClick = () => {
    if (busy) return;
    if (running) onPause();
    else onResume();
  };

  return (
    <button
      className="chip-button chip-button--proc"
      onClick={onClick}
      disabled={busy}
      title={
        running
          ? 'Pause detection and the VLM across all cameras'
          : status === 'paused'
            ? 'Resume processing'
            : 'Models are warming up'
      }
    >
      <span className={dotClass} />
      {running ? '⏸ ' : status === 'paused' ? '▶ ' : ''}PROC · {label.toUpperCase()}
    </button>
  );
}

// --------------------------------------------------------------------------
// the confirm dialog (pause only — resume needs no ceremony)
// --------------------------------------------------------------------------

interface ConfirmProps {
  pending: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

export function PauseConfirm({ pending, onConfirm, onCancel }: ConfirmProps) {
  useEffect(() => {
    const escape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onCancel();
    };
    document.addEventListener('keydown', escape);
    return () => document.removeEventListener('keydown', escape);
  }, [onCancel]);

  return (
    <div className="scrim" style={{ alignItems: 'center' }} onClick={onCancel}>
      <div className="confirm" onClick={(bubble) => bubble.stopPropagation()} role="dialog" aria-modal="true">
        <div className="confirm__title">Pause processing?</div>
        <div className="confirm__body">
          This stops detection and the VLM across <strong>all cameras</strong>. No footage is
          decoded, no alerts are raised, and the GPU goes idle. The console stays up and you can
          resume anytime.
        </div>
        <div className="confirm__actions">
          <button className="button--ghost" onClick={onCancel} disabled={pending}>
            Cancel
          </button>
          <button className="button" onClick={onConfirm} disabled={pending}>
            {pending ? 'Pausing…' : 'Pause processing'}
          </button>
        </div>
      </div>
    </div>
  );
}
