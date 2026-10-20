/**
 * ⌘K — go anywhere, focus any camera.
 *
 * Built from what the appliance actually has, and from what this session may
 * open: the screens in the nav plus every camera in the fleet, filtered as you
 * type. The design shows a static list; a palette that cannot find CAM-07 on a
 * 48-camera site would be furniture — and one that offers a SYSTEM account
 * seven screens it will be refused would be worse than furniture.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import type { Screen } from '../lib/router';
import { SCREEN_LABELS } from '../lib/router';
import type { LiveCamera } from '../api/types';

export interface Command {
  key: string;
  glyph: string;
  label: string;
  run: () => void;
}

interface Props {
  cameras: LiveCamera[];
  /** The same list the nav shows — see App.tsx. */
  screens: readonly Screen[];
  pending: number;
  onNavigate: (screen: Screen) => void;
  onFocusCamera: (cameraId: string) => void;
  onEnterTv: () => void;
  onClose: () => void;
}

export function CommandPalette({
  cameras,
  screens: reachable,
  pending,
  onNavigate,
  onFocusCamera,
  onEnterTv,
  onClose,
}: Props) {
  const [text, setText] = useState('');
  const [cursor, setCursor] = useState(0);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    input.current?.focus();
  }, []);

  const commands = useMemo<Command[]>(() => {
    const screens: Command[] = reachable.map((screen) => ({
      key: screen,
      glyph: '→',
      label:
        screen === 'triage' && pending > 0
          ? `Go to Review Triage (${pending} pending)`
          : `Go to ${SCREEN_LABELS[screen]}`,
      run: () => onNavigate(screen),
    }));
    const tiles: Command[] = cameras.map((camera) => ({
      key: `cam:${camera.id}`,
      glyph: '⌕',
      label: `Focus ${camera.id} · ${camera.name}`,
      run: () => onFocusCamera(camera.id),
    }));
    return [
      ...screens,
      ...tiles,
      { key: 'tv', glyph: '⛶', label: 'Enter Wall TV mode', run: onEnterTv },
    ];
  }, [cameras, reachable, pending, onNavigate, onFocusCamera, onEnterTv]);

  const matches = useMemo(() => {
    const needle = text.trim().toLowerCase();
    const list = needle
      ? commands.filter((command) => command.label.toLowerCase().includes(needle))
      : commands;
    return list.slice(0, 8);
  }, [commands, text]);

  const choose = (command: Command | undefined) => {
    if (!command) return;
    command.run();
    onClose();
  };

  return (
    <div className="scrim" style={{ paddingTop: '14vh' }} onClick={onClose}>
      <div
        className="palette slidein"
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-label="Command palette"
      >
        <div className="palette__field">
          <span style={{ color: 'var(--dim)', fontSize: 'var(--fs-lede)' }}>⌕</span>
          <input
            ref={input}
            value={text}
            placeholder="Jump to camera, screen, policy…"
            onChange={(event) => {
              setText(event.target.value);
              setCursor(0);
            }}
            onKeyDown={(event) => {
              if (event.key === 'ArrowDown') {
                event.preventDefault();
                setCursor((index) => Math.min(index + 1, matches.length - 1));
              } else if (event.key === 'ArrowUp') {
                event.preventDefault();
                setCursor((index) => Math.max(index - 1, 0));
              } else if (event.key === 'Enter') {
                event.preventDefault();
                choose(matches[cursor]);
              }
            }}
          />
        </div>
        <div style={{ padding: 7 }}>
          {matches.length === 0 ? (
            <div style={{ padding: '9px 11px', color: 'var(--faint)', fontSize: 'var(--fs-meta)' }}>
              Nothing matches “{text}”.
            </div>
          ) : null}
          {matches.map((command, index) => (
            <button
              key={command.key}
              className={`palette__row ${index === cursor ? 'palette__row--active' : ''}`}
              onMouseEnter={() => setCursor(index)}
              onClick={() => choose(command)}
            >
              <span className="mono palette__glyph">{command.glyph}</span>
              {command.label}
            </button>
          ))}
        </div>
        {/* The keys, written down. A palette that only rewards people who
            already know how to drive it is a palette for the person who built
            it. */}
        <div className="palette__foot">
          <span>↑↓ NAVIGATE</span>
          <span>⏎ OPEN</span>
          <span>ESC CLOSE</span>
          <span style={{ marginLeft: 'auto' }}>
            {matches.length} {matches.length === 1 ? 'MATCH' : 'MATCHES'}
          </span>
        </div>
      </div>
    </div>
  );
}
