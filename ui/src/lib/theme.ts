/**
 * Ink or paper.
 *
 * The choice is written to `data-theme` on the document element (the tokens
 * file keys off it) and remembered in localStorage — a control room that runs
 * dark should not have to say so again after every reload.
 *
 * **With nothing remembered, this opens dark**, and that is a deliberate change
 * from following `prefers-color-scheme`. The OS preference is a statement about
 * the machine somebody reads email on; this console is a wall in a room whose
 * lights are usually down, sitting beside camera tiles that are dark whatever
 * the theme says. A paper-white sheet between two night-shift feeds is a glare
 * source, and on a shared appliance the first operator of the day would have to
 * fix it on every fresh browser profile.
 *
 * The paper theme is one click away in the masthead and is maintained to the
 * same standard — every token pair in it is checked by tests/test_ui_design.py,
 * the same as the dark one. It is genuinely the better theme for the two places
 * this product gets printed and read on paper: the analytics screen and an
 * exported incident.
 */

import { useCallback, useEffect, useState } from 'react';

export type Theme = 'light' | 'dark';

const STORAGE_KEY = 'argus.theme';

function initial(): Theme {
  const stored = window.localStorage.getItem(STORAGE_KEY);
  if (stored === 'light' || stored === 'dark') return stored;
  return 'dark';
}

export function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(initial);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem(STORAGE_KEY, theme);
  }, [theme]);

  const toggle = useCallback(() => setTheme((current) => (current === 'dark' ? 'light' : 'dark')), []);
  return [theme, toggle];
}
