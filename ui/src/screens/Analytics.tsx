/**
 * Compliance Analytics.
 *
 * Every number comes from /api/analytics/summary, which counts real events and
 * real human verdicts. The distinction the design draws — raw detections versus
 * *verified* violations — is the honest core of this screen: a detection is what
 * a model thought, a verified violation is what a person confirmed, and an
 * auditor is only ever shown the second.
 *
 * The heatmap's columns are the plant's own shifts, read from config, so a site
 * running two twelve-hour shifts gets two columns without a code change.
 *
 * **Incidents, not sightings.** Every count on this screen is a *condition* —
 * one per thing that happened, however long it went on — and the sampled
 * verdicts behind it are named separately wherever they would otherwise be
 * mistaken for volume. That is not a presentation choice: before schema
 * migration 7 a prompt policy raised a row every cooldown, so one laminate
 * offcut on one apron overnight arrived here as forty violations, and the trend
 * chart was drawing the policy's sampling interval rather than the plant's
 * behaviour. Both numbers are reported and neither is hidden, because counting
 * only incidents would hide that the mess sat there for nine hours.
 */

import { useState } from 'react';
import { api } from '../api/client';
import type { AnalyticsSummary, AnalyticsTotals } from '../api/types';
import { ErrorBar } from '../components/marks';
import { useResource } from '../lib/hooks';
import { count, dayLabel, duration, originLook, percent } from '../lib/format';
import { PolicyReportCards } from './PolicyReportCards';

const WINDOWS = [7, 30, 90];

export function Analytics() {
  const [days, setDays] = useState(30);
  const summary = useResource(() => api.analytics(days), [days]);
  const data = summary.data;

  return (
    <div className="rise analytics">
      <div className="analytics__inner">
        <div className="analytics__head">
          <div className="screen-title">Compliance Analytics</div>
          <div className="analytics__window">
            {WINDOWS.map((option) => (
              <button
                key={option}
                className={`analytics__windowbtn ${option === days ? 'analytics__windowbtn--on' : ''}`}
                onClick={() => setDays(option)}
              >
                {option} DAYS
              </button>
            ))}
          </div>
          {data ? (
            <span className="mono" style={{ fontSize: 'var(--fs-note)', color: 'var(--faint)' }}>
              {dayLabel(data.window.since_day)} – {dayLabel(data.window.until_day)}
            </span>
          ) : null}
        </div>

        <ErrorBar error={summary.error} />

        {data ? (
          <>
            <Scorecards totals={data.totals} previous={data.previous} />
            <TrendChart data={data} />
            <div className="analytics__split">
              <Heatmap data={data} />
              <ByType data={data} />
            </div>
            <PolicyReportCards data={data} />
            <div style={{ fontSize: 'var(--fs-note)', color: 'var(--faint)', lineHeight: 1.7 }}>
              Counts are incidents raised by this appliance between{' '}
              {dayLabel(data.window.since_day)} and {dayLabel(data.window.until_day)}, bucketed on
              the appliance's local clock by the moment each one began. A prompt policy's incident
              is a condition that lasts, counted once however many times the model saw it —{' '}
              {count(data.totals.events)} incidents over {count(data.totals.observations)}{' '}
              observations in this window. “Verified” means a human recorded a confirmed verdict;
              every verified violation has its annotated frame and clip on this box for{' '}
              {data.totals.events > 0 ? 'the configured retention window' : 'as long as retention allows'}.
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
}

function delta(now: number, before: number): { text: string; color: string } {
  const change = now - before;
  if (before === 0 && now === 0) return { text: '±0', color: 'var(--dim)' };
  if (change === 0) return { text: '±0', color: 'var(--dim)' };
  return {
    text: `${change > 0 ? '+' : '−'}${Math.abs(change)}`,
    // More violations is worse; the colour follows meaning, not direction.
    color: change > 0 ? 'var(--warn)' : 'var(--good)',
  };
}

function Scorecards({ totals, previous }: { totals: AnalyticsTotals; previous: AnalyticsTotals }) {
  const precisionDelta =
    totals.precision !== null && previous.precision !== null
      ? {
          text: `${totals.precision >= previous.precision ? '+' : '−'}${Math.abs(
            (totals.precision - previous.precision) * 100,
          ).toFixed(1)} pt`,
          color: totals.precision >= previous.precision ? 'var(--good)' : 'var(--warn)',
        }
      : { text: 'no prior', color: 'var(--faint)' };

  const responseDelta =
    totals.median_response_sec !== null && previous.median_response_sec !== null
      ? {
          text: `${totals.median_response_sec <= previous.median_response_sec ? '−' : '+'}${duration(
            Math.abs(totals.median_response_sec - previous.median_response_sec),
          )}`,
          color:
            totals.median_response_sec <= previous.median_response_sec
              ? 'var(--good)'
              : 'var(--warn)',
        }
      : { text: 'no prior', color: 'var(--faint)' };

  const cards: { label: string; value: string; delta: { text: string; color: string }; hint: string }[] = [
    {
      label: 'Review precision',
      value: percent(totals.precision),
      delta: precisionDelta,
      hint: `${count(totals.confirmed)} confirmed of ${count(totals.reviewed)} reviewed`,
    },
    {
      label: 'Verified violations',
      value: count(totals.confirmed),
      delta: delta(totals.confirmed, previous.confirmed),
      // `events` counts incidents; `observations` counts how many times they
      // were seen. Saying only the first would hide a mess that lasted a
      // shift; saying only the second would call it forty violations.
      hint: `${count(totals.events)} incidents · ${count(totals.observations)} observations`,
    },
    {
      label: 'Awaiting review',
      value: count(totals.pending),
      delta: delta(totals.pending, previous.pending),
      hint: `${count(totals.open_high)} of them high severity${
        totals.ongoing > 0 ? ` · ${count(totals.ongoing)} still ongoing` : ''
      }`,
    },
    {
      label: 'Median response',
      value: totals.median_response_sec === null ? '—' : duration(totals.median_response_sec),
      delta: responseDelta,
      hint: 'violation → human verdict',
    },
  ];

  return (
    <div className="analytics__cards">
      {cards.map((card) => (
        <div key={card.label} className="panel scorecard">
          <div style={{ fontSize: 'var(--fs-note)', color: 'var(--dim)' }}>{card.label}</div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginTop: 8 }}>
            <span className="mono" style={{ fontSize: 'var(--fs-vital)', fontWeight: 700 }}>
              {card.value}
            </span>
            <span className="mono" style={{ fontSize: 'var(--fs-note)', color: card.delta.color }}>
              {card.delta.text}
            </span>
          </div>
          <div style={{ fontSize: 'var(--fs-note)', color: 'var(--faint)', marginTop: 4 }}>{card.hint}</div>
        </div>
      ))}
    </div>
  );
}

function TrendChart({ data }: { data: AnalyticsSummary }) {
  const peak = Math.max(1, ...data.by_day.map((day) => day.raw));
  const ticks = [0, Math.floor(data.by_day.length / 3), Math.floor((2 * data.by_day.length) / 3), data.by_day.length - 1];

  return (
    <div className="panel">
      <div style={{ display: 'flex', alignItems: 'baseline', marginBottom: 14 }}>
        <span style={{ fontWeight: 600, fontSize: 'var(--fs-body)' }}>Incidents per day</span>
        <span style={{ marginLeft: 'auto', fontSize: 'var(--fs-note)', color: 'var(--dim)' }}>
          <span className="legend-swatch" style={{ background: 'var(--accent)' }} />
          verified&nbsp;&nbsp;
          <span className="legend-swatch" style={{ background: 'var(--track)' }} />
          incidents raised
        </span>
      </div>
      <div className="trend">
        {data.by_day.map((day) => {
          const verified = (day.confirmed / peak) * 100;
          const rest = ((day.raw - day.confirmed) / peak) * 100;
          return (
            <div
              key={day.day}
              className="trend__col"
              title={
                `${dayLabel(day.day)} · ${day.raw} incident${day.raw === 1 ? '' : 's'}` +
                ` (${day.observations} observation${day.observations === 1 ? '' : 's'})` +
                `, ${day.confirmed} verified, ${day.false_positive} false positive`
              }
            >
              <div style={{ height: `${Math.max(0, rest)}%`, background: 'var(--track)' }} />
              <div style={{ height: `${Math.max(day.confirmed > 0 ? 2 : 0, verified)}%`, background: 'var(--accent)' }} />
            </div>
          );
        })}
      </div>
      <div className="trend__axis mono">
        {ticks
          .filter((tick, index, list) => list.indexOf(tick) === index)
          .map((tick) => (
            <span key={tick}>{data.by_day[tick] ? dayLabel(data.by_day[tick]!.day) : ''}</span>
          ))}
      </div>
    </div>
  );
}

function Heatmap({ data }: { data: AnalyticsSummary }) {
  const peak = Math.max(
    1,
    ...data.by_zone_shift.flatMap((row) => data.shifts.map((shift) => row.raw[shift.id] ?? 0)),
  );

  return (
    <div className="panel">
      <div style={{ fontWeight: 600, fontSize: 'var(--fs-body)', marginBottom: 12 }}>
        Violation clusters — zone × shift
      </div>
      {data.by_zone_shift.length === 0 ? (
        <div style={{ fontSize: 'var(--fs-meta)', color: 'var(--faint)' }}>
          No events in this window, so there is nothing to cluster.
        </div>
      ) : (
        <div
          className="heat"
          style={{ gridTemplateColumns: `110px repeat(${data.shifts.length}, 1fr)` }}
        >
          <span />
          {data.shifts.map((shift) => (
            <span key={shift.id} className="mono heat__head">
              SHIFT {shift.id}
            </span>
          ))}
          {data.by_zone_shift.map((row) => (
            <Row key={row.zone_id ?? row.zone_name} row={row} shifts={data.shifts} peak={peak} />
          ))}
        </div>
      )}
    </div>
  );
}

function Row({
  row,
  shifts,
  peak,
}: {
  row: AnalyticsSummary['by_zone_shift'][number];
  shifts: AnalyticsSummary['shifts'];
  peak: number;
}) {
  return (
    <>
      <span style={{ fontSize: 'var(--fs-note)', color: 'var(--text2)' }}>{row.zone_name}</span>
      {shifts.map((shift) => {
        const value = row.raw[shift.id] ?? 0;
        const intensity = (value / peak) * 0.85;
        return (
          <div
            key={shift.id}
            className="heat__cell mono"
            style={{
              background: `rgba(179,52,31,${intensity.toFixed(2)})`,
              color: intensity > 0.5 ? '#F6F1E2' : 'var(--text2)',
            }}
            title={
              `${row.zone_name} · shift ${shift.id} · ${value} incident${value === 1 ? '' : 's'}` +
              ` (${row.observations[shift.id] ?? 0} observations),` +
              ` ${row.confirmed[shift.id] ?? 0} verified`
            }
          >
            {value}
          </div>
        );
      })}
    </>
  );
}

function ByType({ data }: { data: AnalyticsSummary }) {
  const peak = Math.max(1, ...data.by_type.map((row) => row.raw));
  return (
    <div className="panel">
      <div style={{ fontWeight: 600, fontSize: 'var(--fs-body)', marginBottom: 12 }}>By violation type</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {data.by_type.length === 0 ? (
          <div style={{ fontSize: 'var(--fs-meta)', color: 'var(--faint)' }}>Nothing raised in this window.</div>
        ) : null}
        {data.by_type.map((row) => (
          <div key={row.key} className="bytype">
            <span className="bytype__label" title={row.key}>
              {/* Which engine produced this row. Prompt-policy counts and
                  detector counts sit in the same chart and are not the same
                  kind of number — one is a model's opinion sampled every few
                  seconds, the other a count on every analysed frame. */}
              <span
                className="mono bytype__origin"
                title={originLook(row.detector_type).title}
              >
                {originLook(row.detector_type).short}
              </span>
              {row.label}
            </span>
            <div className="bytype__track">
              <div
                className="bytype__raw"
                style={{ width: `${(row.raw / peak) * 100}%` }}
              />
              <div
                className="bytype__verified"
                style={{ width: `${(row.confirmed / peak) * 100}%` }}
              />
            </div>
            <span
              className="mono bytype__value"
              title={
                `${row.confirmed} verified of ${row.raw} incident${row.raw === 1 ? '' : 's'}, ` +
                `seen ${row.observations} time${row.observations === 1 ? '' : 's'}`
              }
            >
              {row.confirmed}
              <span style={{ color: 'var(--faint)' }}> /{row.raw}</span>
            </span>
          </div>
        ))}
      </div>
      <div className="bytype__foot">
        Verified by human review / incidents raised. A prompt policy's incident is a
        <em> condition</em>: it stays one row while it goes on, so a mess that lasted a shift
        counts once here and its sightings are on the tooltip. Every verified violation carries
        its evidence clip and the timestamp of the verdict. <span className="mono">VLM</span>{' '}
        marks a prompt policy judged by the on-box model; <span className="mono">DET</span> the
        fast detector.
      </div>
    </div>
  );
}
