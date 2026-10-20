/**
 * One card per policy — the report-cards section of Compliance Analytics.
 *
 * Built entirely from `AnalyticsSummary.by_type`, the same automatic,
 * zero-hardcoded-policy-names source the screen's existing by-type list
 * already uses: a new policy in Policy Studio gets a card here the moment it
 * raises its first event in the window, with no code change. There is
 * deliberately no fixed list of policy names anywhere in this file.
 *
 * Each card gets three breakdowns the top-level screen doesn't carry per
 * policy: a severity mix (the same high/medium/low field triage sorts by —
 * not an invented per-policy scale), a shift split (folded from local hour
 * the same way the zone heatmap is), and a daily trend. All three come
 * straight off the API response; nothing here is estimated client-side.
 *
 * Colour is identity, not state: each card gets one of a small validated
 * categorical set, cycled by position, so "which policy is this" never
 * borrows --good/--warn/--crit, which stay reserved for whether a value
 * is good or bad.
 */

import type { AnalyticsSummary, Severity } from '../api/types';
import { severityLook, originLook, count } from '../lib/format';

const REP_COLORS = ['var(--rep-1)', 'var(--rep-2)', 'var(--rep-3)', 'var(--rep-4)'];
const SEVERITY_ORDER: Severity[] = ['high', 'medium', 'low'];

export function PolicyReportCards({ data }: { data: AnalyticsSummary }) {
  if (data.by_type.length === 0) return null;

  return (
    <div className="panel">
      <div style={{ fontWeight: 600, fontSize: 'var(--fs-body)', marginBottom: 4 }}>
        Policy reports
      </div>
      <div style={{ fontSize: 'var(--fs-note)', color: 'var(--faint)', marginBottom: 14 }}>
        One card per policy that raised an incident in this window — shift split, severity mix and
        trend, each real and automatic. Add a policy in Policy Studio and it appears here the first
        time it fires, not before.
      </div>
      <div className="policyreports">
        {data.by_type.map((row, index) => (
          <PolicyCard key={row.key} row={row} shifts={data.shifts} color={REP_COLORS[index % REP_COLORS.length]!} />
        ))}
      </div>
    </div>
  );
}

function PolicyCard({
  row,
  shifts,
  color,
}: {
  row: AnalyticsSummary['by_type'][number];
  shifts: AnalyticsSummary['shifts'];
  color: string;
}) {
  const severityEntries = SEVERITY_ORDER.filter((key) => (row.by_severity[key] ?? 0) > 0);
  const severityTotal = severityEntries.reduce((sum, key) => sum + (row.by_severity[key] ?? 0), 0);

  const shiftMax = Math.max(1, ...shifts.map((shift) => row.by_shift[shift.id]?.raw ?? 0));

  const dayPeak = Math.max(1, ...row.by_day.map((day) => day.raw));

  return (
    <div className="policycard">
      <div className="policycard__head">
        <span className="policycard__swatch" style={{ background: color }} />
        <span className="policycard__name" title={row.key}>
          {row.label}
        </span>
        <span className="mono" style={{ fontSize: 'var(--fs-label)', color: 'var(--faint)' }} title={originLook(row.detector_type).title}>
          {originLook(row.detector_type).short}
        </span>
      </div>

      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
        <span className="mono" style={{ fontSize: 'var(--fs-vital)', fontWeight: 700 }}>
          {count(row.confirmed)}
        </span>
        <span style={{ fontSize: 'var(--fs-note)', color: 'var(--faint)' }}>
          verified of {count(row.raw)} incidents · {count(row.observations)} observations
        </span>
      </div>

      {severityTotal > 0 ? (
        <div style={{ marginTop: 10 }}>
          <div className="policycard__stack">
            {severityEntries.map((key) => (
              <span
                key={key}
                style={{
                  width: `${((row.by_severity[key] ?? 0) / severityTotal) * 100}%`,
                  background: severityLook(key).color,
                }}
                title={`${severityLook(key).label}: ${row.by_severity[key]}`}
              />
            ))}
          </div>
          <div style={{ display: 'flex', gap: 12, marginTop: 5, flexWrap: 'wrap' }}>
            {severityEntries.map((key) => (
              <span key={key} className="mono" style={{ fontSize: 'var(--fs-label)', color: 'var(--faint)' }}>
                <span className="legend-swatch" style={{ background: severityLook(key).color }} />
                {severityLook(key).label.toLowerCase()} · {row.by_severity[key]}
              </span>
            ))}
          </div>
        </div>
      ) : null}

      <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 4 }}>
        {shifts.map((shift) => {
          const value = row.by_shift[shift.id]?.raw ?? 0;
          return (
            <div key={shift.id} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span className="mono" style={{ fontSize: 'var(--fs-label)', color: 'var(--faint)', width: 48, flex: 'none' }}>
                SHIFT {shift.id}
              </span>
              <span style={{ flex: 1, height: 10, background: 'var(--track)', borderRadius: 2, position: 'relative', overflow: 'hidden' }}>
                <span
                  style={{
                    position: 'absolute', left: 0, top: 0, bottom: 0,
                    width: `${Math.max(value > 0 ? 4 : 0, (value / shiftMax) * 100)}%`,
                    background: color, borderRadius: 2,
                  }}
                />
              </span>
              <span className="mono" style={{ fontSize: 'var(--fs-label)', color: 'var(--faint)', width: 22, textAlign: 'right' }}>
                {value}
              </span>
            </div>
          );
        })}
      </div>

      {row.by_day.length > 1 ? (
        <div className="policycard__spark" title="Daily incidents across the selected window">
          {row.by_day.map((day) => (
            <span
              key={day.day}
              style={{ height: `${Math.max(day.raw > 0 ? 6 : 0, (day.raw / dayPeak) * 100)}%`, background: color }}
            />
          ))}
        </div>
      ) : null}
    </div>
  );
}
