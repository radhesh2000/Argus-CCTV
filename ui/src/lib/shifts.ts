/**
 * Which shift a moment belongs to.
 *
 * The rota comes from the appliance (`site.shifts` in argus.yaml), so a plant
 * running two twelve-hour shifts gets two columns in the heatmap and the right
 * word in the masthead without anybody editing this file. The same wrapping
 * rule as the server's `SiteConfig.shift_for_hour`: a shift owns every hour
 * from its start until the next one begins, and the last one wraps midnight.
 */

import type { Shift } from '../api/types';

export function shiftForHour(shifts: Shift[], hour: number): Shift | null {
  if (shifts.length === 0) return null;
  const clamped = ((hour % 24) + 24) % 24;
  let current = shifts[shifts.length - 1] ?? null;
  for (const shift of shifts) {
    if (clamped >= shift.start_hour) current = shift;
  }
  return current;
}

export function shiftForDate(shifts: Shift[], date: Date): Shift | null {
  return shiftForHour(shifts, date.getHours());
}

/** `SHIFT B · 14–22`, the way the filter sidebar writes it. */
export function shiftWindowLabel(shift: Shift): string {
  const pad = (hour: number) => String(hour).padStart(2, '0');
  return `${pad(shift.start_hour)}–${pad(shift.end_hour)}`;
}
