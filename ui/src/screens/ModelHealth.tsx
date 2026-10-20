/**
 * Model Health & Accuracy.
 *
 * Vitals from /api/health (GPU, unified memory, disk, engine, VLM queue),
 * detector accuracy from real verdicts, per-camera achieved fps from the
 * engine's own measurements.
 *
 * Two things this screen has to get right about the VLM. The **queue** is a
 * number only `argus run` can produce — it is the scheduler's own counter — so
 * when the engine is stopped the API answers with nulls and this screen prints
 * "—" rather than a zero, because a zero here reads as "idle and healthy",
 * which is the opposite of true. And the **degraded state is a first-class
 * render**, not an error: the model server is optional equipment on an
 * appliance, it takes minutes to load, and the rest of the box (decode,
 * detection, zones, alerts) is untouched while it is down. The panel therefore
 * says what is missing and which command supplies it.
 *
 * GB10 reports no discrete VRAM (the 128 GB is unified with the CPU), so the
 * memory vital is labelled for what it is rather than left blank or invented.
 */

import { useMemo } from 'react';
import { api } from '../api/client';
import type { AccuracyReport, Health, VlmHealth, VlmSchedule } from '../api/types';
import { NotEnabled, OriginBadge, StageBadge, ErrorBar } from '../components/marks';
import { usePoll, useResource } from '../lib/hooks';
import { bytesFromMb, count, duration, percent } from '../lib/format';
import { useAppState } from '../state/AppState';

export function ModelHealth() {
  const { policies, policyLabels, config } = useAppState();
  const health = usePoll(() => api.health(), 5000);
  const accuracy = useResource(() => api.accuracy(14), []);

  // How many prompt policies this appliance is configured to run, which is the
  // number that makes a degraded VLM worth acting on: nothing is broken if
  // nothing was going to use it.
  const promptPolicies = policies.filter((policy) => policy.kind === 'vlm').length;

  return (
    <div className="rise health">
      <div className="health__inner">
        <div className="screen-title">Model Health &amp; Accuracy</div>
        <ErrorBar error={health.error ?? accuracy.error} />

        {health.data ? (
          <Vitals
            health={health.data}
            pending={accuracy.data?.total_pending ?? 0}
            paused={health.data.processing?.status === 'paused'}
          />
        ) : null}

        {health.data ? (
          <div className="health__vlm">
            <VlmPanel
              vlm={health.data.vlm}
              promptPolicies={promptPolicies}
              paused={health.data.processing?.status === 'paused'}
            />
          </div>
        ) : null}

        <DetectorTable
          accuracy={accuracy.data}
          stages={Object.fromEntries(policies.map((policy) => [policy.id, policy.stage]))}
          labels={policyLabels}
        />

        <div className="health__split">
          <NotEnabled title="Retraining exports not enabled" phase="Phase 6">
            False positives ruled in triage are recorded with their evidence and are ready to be
            exported as a labelled set, but the on-box retraining loop that consumes them is not
            installed. The export history the design shows here would be empty, so it is not
            drawn with invented rows.
            {accuracy.data ? (
              <>
                <br />
                <br />
                <span className="mono" style={{ fontSize: 'var(--fs-note)' }}>
                  {count(
                    accuracy.data.detectors.reduce((sum, row) => sum + row.false_positive, 0),
                  )}{' '}
                  false positives are already labelled and waiting.
                </span>
              </>
            ) : null}
          </NotEnabled>

          <div className="panel">
            <div style={{ fontSize: 'var(--fs-meta)', fontWeight: 600, marginBottom: 12 }}>
              Per-camera inference FPS
            </div>
            <CameraFps health={health.data} target={config?.detection.fps ?? 5} />
          </div>
        </div>
      </div>
    </div>
  );
}

/** The VLM queue vital, in the design's own three lines. */
function vlmVital(vlm: VlmHealth, paused: boolean): { value: string; sub: string; color?: string } {
  if (paused) {
    return { value: 'PAUSED', sub: 'resumes with processing', color: 'var(--warn)' };
  }
  if (!vlm.enabled) {
    return { value: 'OFF', sub: 'prompt policies not evaluated', color: 'var(--faint)' };
  }
  if (!vlm.available) {
    return { value: 'DOWN', sub: 'model not answering', color: 'var(--crit)' };
  }
  if (!vlm.inference) {
    // The endpoint answered, but nobody is feeding it: the queue belongs to the
    // scheduler and the scheduler lives inside `argus run`.
    return { value: '—', sub: 'model up · engine not sampling', color: 'var(--warn)' };
  }
  const { queue_depth, in_flight, avg_latency_sec, last_inference_age_sec } = vlm.inference;
  return {
    value: String(queue_depth),
    sub:
      `${in_flight} in flight · ${avg_latency_sec.toFixed(1)} s avg` +
      (last_inference_age_sec === null ? '' : ` · last ${duration(last_inference_age_sec)} ago`),
    color: queue_depth > 0 ? 'var(--warn)' : 'var(--good)',
  };
}

/**
 * The DISK vital's second line.
 *
 * The headline percentage is the *filesystem*, which on a 3.7 TB appliance is
 * the slowest-moving number on this screen — it read 36% full while
 * `data/clips` was growing 250 GB a day and two and a half days from being a
 * problem. Evidence against its configured budget is the number that actually
 * moves, so that is what this line says wherever a budget exists.
 *
 * Where none is configured there is nothing to measure against, and it says
 * what it knows instead of inventing a denominator.
 */
function diskSub(disk: Health['disk']): string {
  const held = `${disk.free_gb} GB free`;
  const { used_gb, budget_gb } = disk.evidence;
  if (budget_gb === null) {
    return `${held} · Argus holds ${bytesFromMb(disk.argus_mb)}`;
  }
  // Whole GB above 10, one decimal below it. A bench appliance with a 2 GB
  // budget should not read "0 / 2 GB" while it is holding evidence.
  const gb = (value: number) => (value >= 10 ? Math.round(value) : Math.round(value * 10) / 10);
  const share = Math.round((used_gb / budget_gb) * 100);
  return `${held} · evidence ${gb(used_gb)} / ${gb(budget_gb)} GB (${share}%)`;
}

function Vitals({ health, pending, paused }: { health: Health; pending: number; paused: boolean }) {
  const gpu = health.gpu.gpus[0] ?? null;
  const unified = health.gpu.memory_source === 'unified';
  const engine = health.engine;
  const vlm = vlmVital(health.vlm, paused);

  const vitals: { label: string; value: string; sub: string; color?: string }[] = [
    {
      label: 'GPU UTILIZATION',
      value: gpu?.utilization_percent != null ? `${gpu.utilization_percent}%` : '—',
      sub: gpu?.name ?? health.gpu.reason ?? 'no GPU probe',
    },
    {
      label: unified ? 'UNIFIED MEMORY' : 'VRAM',
      // "53 / 131 GB" rather than "53.0 GB / 130.7 GB": the vital card is
      // narrow and the units are the same on both sides of the slash.
      value:
        gpu?.memory_used_mb != null && gpu?.memory_total_mb != null
          ? `${Math.round(gpu.memory_used_mb / 1000)} / ${Math.round(gpu.memory_total_mb / 1000)} GB`
          : '—',
      sub: unified ? 'GB10 shares one pool with the CPU' : 'device memory',
    },
    {
      label: 'DISK',
      value: `${health.disk.percent}%`,
      sub: diskSub(health.disk),
      // Over the evidence budget is worth the same amber as a full filesystem:
      // it is the earlier of the two warnings and the one an operator can still
      // act on before anything is deleted.
      color:
        health.disk.over_threshold || health.disk.evidence.over_budget
          ? 'var(--warn)'
          : undefined,
    },
    {
      label: 'ENGINE',
      value: engine.running ? 'RUNNING' : paused ? 'PAUSED' : 'STOPPED',
      sub: engine.running
        ? `up ${duration(engine.uptime_sec ?? 0)}`
        : paused
          ? 'processing switched off — GPU idle'
          : (engine.reason ?? 'argus run is not going'),
      // Paused is a deliberate switch, so it is amber, not the crit red of a
      // crashed engine — the vital must not read "broken" at a box turned off.
      color: engine.running ? 'var(--good)' : paused ? 'var(--warn)' : 'var(--crit)',
    },
    { label: 'VLM QUEUE', ...vlm },
    {
      label: 'REVIEW BACKLOG',
      value: count(pending),
      sub: pending === 0 ? 'queue clear' : 'awaiting a human verdict',
      color: pending > 0 ? 'var(--warn)' : undefined,
    },
  ];

  return (
    <div className="health__vitals">
      {vitals.map((vital) => (
        <div
          key={vital.label}
          className={`panel vital ${vital.color === 'var(--crit)' ? 'vital--bad' : ''} ${
            vital.color === 'var(--warn)' ? 'vital--warn' : ''
          }`}
        >
          <div className="vital__label mono">{vital.label}</div>
          <div className="vital__value" style={{ color: vital.color ?? 'var(--text)' }}>
            {vital.value}
          </div>
          <div className="vital__sub">{vital.sub}</div>
        </div>
      ))}
    </div>
  );
}

/**
 * The VLM policy engine, in full.
 *
 * Renders in every state the appliance can actually be in: off by
 * configuration, installed but not started, up but unfed, and running. The
 * degraded copy names the command that fixes it, because "DEGRADED" alone
 * sends an operator to a log file.
 */
function VlmPanel({
  vlm,
  promptPolicies,
  paused,
}: {
  vlm: VlmHealth;
  promptPolicies: number;
  paused: boolean;
}) {
  const state = paused
    ? { label: 'PAUSED', className: 'badge badge--pilot' }
    : !vlm.enabled
      ? { label: 'DISABLED', className: 'badge badge--shadow' }
      : vlm.available
        ? { label: 'READY', className: 'badge badge--live' }
        : { label: 'DEGRADED', className: 'badge badge--pilot' };

  const schedules = vlm.schedules ?? [];

  return (
    <div className="panel vlm">
      <div className="vlm__head">
        <span style={{ fontSize: 'var(--fs-meta)', fontWeight: 600 }}>VLM policy engine</span>
        <span className={state.className}>{state.label}</span>
        <span className="mono vlm__model" title="served model id">
          {vlm.model ?? vlm.configured_model}
        </span>
        <span className="mono vlm__endpoint">{vlm.endpoint}</span>
      </div>

      {vlm.model && vlm.model !== vlm.configured_model ? (
        <div className="vlm__note" style={{ color: 'var(--warn)' }}>
          Serving <span className="mono">{vlm.model}</span>, but this appliance is configured for{' '}
          <span className="mono">{vlm.configured_model}</span>.
        </div>
      ) : null}

      <div className="vlm__facts">
        <Fact label="EVALUATIONS" value={count(vlm.evaluations)} unknown={vlm.evaluations === null} />
        <Fact label="EVENTS RAISED" value={count(vlm.events)} unknown={vlm.events === null} />
        <Fact
          label="AVG LATENCY"
          value={vlm.inference ? `${vlm.inference.avg_latency_sec.toFixed(2)} s` : '—'}
          unknown={!vlm.inference}
        />
        <Fact
          label="LAST INFERENCE"
          value={
            vlm.inference?.last_inference_age_sec != null
              ? `${duration(vlm.inference.last_inference_age_sec)} ago`
              : '—'
          }
          unknown={!vlm.inference || vlm.inference.last_inference_age_sec === null}
        />
        <Fact
          label="VERIFICATION"
          value={vlm.verification.enabled ? 'ON' : 'OFF'}
          sub={
            vlm.verification.enabled
              ? `${count(vlm.verification.checked)} checked · ${count(vlm.verification.suppressed)} held`
              : `would check ${vlm.verification.stages.join(', ')}`
          }
        />
      </div>

      {paused ? (
        <div className="vlm__fix">
          Processing is paused, so the VLM server is stopped along with the vision engine — this
          is the pause switch, not a fault. It comes back up when processing resumes; the model
          then takes a few minutes to load before it answers.
        </div>
      ) : null}

      {!paused && vlm.enabled && !vlm.available ? (
        <div className="vlm__degraded">
          <div style={{ fontWeight: 600, fontSize: 'var(--fs-meta)', marginBottom: 6 }}>
            {promptPolicies > 0
              ? `${promptPolicies} prompt polic${promptPolicies === 1 ? 'y is' : 'ies are'} configured and not running`
              : 'The model is not answering'}
          </div>
          <div className="vlm__reason mono">{vlm.reason ?? 'no reason reported'}</div>
          <div className="vlm__fix">
            {!vlm.install.venv_ready ? (
              <>
                No serving environment at <span className="mono">{vlm.install.venv}</span> — run{' '}
                <code>scripts/setup_vlm_server.sh</code>.
              </>
            ) : !vlm.install.model_ready ? (
              <>
                No weights at <span className="mono">{vlm.install.model_path}</span> — run{' '}
                <code>scripts/fetch_vlm.sh</code>.
              </>
            ) : vlm.install.pid ? (
              <>
                The server was started (pid <span className="mono">{vlm.install.pid}</span>) and is
                not answering yet — an 8B checkpoint takes about two minutes to load.{' '}
                <span className="mono">{vlm.install.log}</span> has the detail.
              </>
            ) : (
              <>
                Weights and environment are installed but nothing is serving them — run{' '}
                <code>argus vlm start</code>.
              </>
            )}{' '}
            Detection, tracking, zone counts and alerts are unaffected: only prompt policies stop.
          </div>
        </div>
      ) : null}

      {!vlm.enabled ? (
        <div className="vlm__fix">
          <code>vlm.enabled</code> is off in the appliance config, so <code>kind: vlm</code>{' '}
          policies are stored and never evaluated
          {promptPolicies > 0 ? ` (${promptPolicies} defined)` : ''}. Nothing else changes — this is
          the Phase-1 appliance.
        </div>
      ) : null}

      {schedules.length > 0 ? (
        <ScheduleTable schedules={schedules} />
      ) : vlm.enabled && vlm.available && vlm.source === 'engine' ? (
        <div className="vlm__fix">
          The model is ready and no prompt policy is scheduled against it. A{' '}
          <code>kind: vlm</code> policy has to be enabled and past draft before the scheduler picks
          it up.
        </div>
      ) : null}

      {!paused && vlm.source === 'probe' && vlm.enabled ? (
        <div className="vlm__note">
          Read by probing the endpoint directly — the engine is not running, so per-policy sampling
          figures are not being produced by anything.
        </div>
      ) : null}
    </div>
  );
}

function Fact({
  label,
  value,
  sub,
  unknown = false,
}: {
  label: string;
  value: string;
  sub?: string;
  unknown?: boolean;
}) {
  return (
    <div className="vlm__fact">
      <div className="vital__label mono">{label}</div>
      <div className="mono vlm__factvalue" style={{ color: unknown ? 'var(--faint)' : 'var(--text)' }}>
        {unknown ? '—' : value}
      </div>
      {sub ? <div className="vital__sub">{sub}</div> : null}
    </div>
  );
}

/** Every (policy × camera) the scheduler is running, with what it has produced. */
function ScheduleTable({ schedules }: { schedules: VlmSchedule[] }) {
  return (
    <div className="table table--schedules">
      <div className="table__head">
        <span>PROMPT POLICY</span>
        <span>CAMERA</span>
        <span>SAMPLING</span>
        <span>EVALS / INCIDENTS</span>
        <span>DROPPED</span>
        <span>LAST VERDICT</span>
      </div>
      {schedules.map((schedule) => (
        <div key={`${schedule.policy_id}:${schedule.camera_id}`} className="table__row table__row--static">
          <span style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
            <OriginBadge type="vlm" short />
            <span className="ellipsis" style={{ fontSize: 'var(--fs-meta)' }}>
              {schedule.label}
            </span>
          </span>
          <span
            className="mono ellipsis"
            style={{ fontSize: 'var(--fs-note)', color: 'var(--dim)', paddingRight: 10 }}
            title={schedule.camera_id}
          >
            {schedule.camera_id}
          </span>
          <span className="mono" style={{ fontSize: 'var(--fs-note)', color: 'var(--dim)' }}>
            1 / {schedule.interval_sec}s
          </span>
          {/* Incidents opened, and — since the appliance folds re-sightings
              into the row that opened rather than raising a new alert every
              cooldown — how many of those re-sightings this schedule has
              filed. A policy watching something that does not move shows a
              small first number and a large second one, which is the shape
              this feature exists to produce. */}
          <span
            className="mono"
            style={{ fontSize: 'var(--fs-note)', color: 'var(--dim)' }}
            title={
              `${schedule.evaluations} evaluations, ${schedule.events} incident(s) opened, ` +
              `${schedule.observations} re-sighting(s) folded in, ${schedule.closed} closed` +
              (schedule.last_close_reason ? ` (last: ${schedule.last_close_reason})` : '')
            }
          >
            {count(schedule.evaluations)} / {count(schedule.events)}
            {schedule.observations > 0 ? (
              <span style={{ color: 'var(--faint)' }}> +{count(schedule.observations)}</span>
            ) : null}
          </span>
          <span
            className="mono"
            style={{
              fontSize: 'var(--fs-note)',
              // Dropped samples are the honest symptom of a model slower than
              // the policy's interval — the scheduler skips rather than queues.
              color: schedule.skipped_in_flight > 0 ? 'var(--warn)' : 'var(--faint)',
            }}
            title="samples skipped because the previous one had not come back yet"
          >
            {count(schedule.skipped_in_flight)}
          </span>
          <span className="ellipsis" style={{ fontSize: 'var(--fs-note)', color: 'var(--dim)' }}>
            {schedule.last_error ? (
              <span style={{ color: 'var(--crit)' }}>{schedule.last_error}</span>
            ) : schedule.last_verdict ? (
              <>
                {schedule.last_verdict.triggered ? '▲ ' : '○ '}
                {schedule.last_verdict.reason ?? '—'}
              </>
            ) : (
              <span style={{ color: 'var(--faint)' }}>not sampled yet</span>
            )}
          </span>
        </div>
      ))}
    </div>
  );
}

function DetectorTable({
  accuracy,
  stages,
  labels,
}: {
  accuracy: AccuracyReport | null;
  stages: Record<string, string>;
  labels: Record<string, string>;
}) {
  const rows = accuracy?.detectors ?? [];
  return (
    <div className="table table--detectors">
      <div className="table__head">
        <span>DETECTOR</span>
        <span>STAGE</span>
        <span>PRECISION</span>
        <span>CONFIRMED / FP</span>
        <span>REVIEWED</span>
        <span>BACKLOG</span>
      </div>
      {rows.length === 0 ? (
        <div style={{ padding: 18, fontSize: 'var(--fs-meta)', color: 'var(--faint)' }}>
          No detector has produced an event yet.
        </div>
      ) : null}
      {rows.map((row) => (
        <div key={row.detector_name} className="table__row table__row--static">
          <span style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
            {/* Which engine produced these numbers. A prompt policy's precision
                and a detector's precision are both real and not comparable. */}
            <OriginBadge type={row.detector_type} short />
            <span className="ellipsis" style={{ fontSize: 'var(--fs-meta)' }}>
              {labels[row.detector_name] ?? row.detector_name}
            </span>
          </span>
          <span>
            <StageBadge stage={(stages[row.detector_name] ?? 'shadow') as never} />
          </span>
          <span
            className="mono"
            style={{
              fontSize: 'var(--fs-meta)',
              color:
                row.precision === null
                  ? 'var(--faint)'
                  : row.precision >= 0.95
                    ? 'var(--good)'
                    : 'var(--warn)',
            }}
          >
            {percent(row.precision)}
          </span>
          <span className="mono" style={{ fontSize: 'var(--fs-note)', color: 'var(--dim)' }}>
            {row.confirmed} / {row.false_positive}
            {row.suppressed > 0 ? (
              <span
                style={{ color: 'var(--faint)' }}
                title={`${row.suppressed} of ${row.verified} checked alerts were held back by the verifier`}
              >
                {' '}
                · {row.suppressed} held
              </span>
            ) : null}
          </span>
          <span className="mono" style={{ fontSize: 'var(--fs-note)', color: 'var(--dim)' }}>
            {count(row.reviewed)}
          </span>
          <span
            className="mono"
            style={{ fontSize: 'var(--fs-note)', color: row.pending > 0 ? 'var(--warn)' : 'var(--faint)' }}
          >
            {count(row.pending)}
          </span>
        </div>
      ))}
    </div>
  );
}

function CameraFps({ health, target }: { health: Health | null; target: number }) {
  const cameras = useMemo(() => health?.cameras ?? [], [health]);
  if (cameras.length === 0) {
    return <div style={{ fontSize: 'var(--fs-meta)', color: 'var(--faint)' }}>No cameras enrolled.</div>;
  }
  return (
    <>
      <div className="fps">
        {cameras.map((camera) => {
          const achieved = camera.fps ?? 0;
          const height = Math.max(3, Math.min(100, (achieved / Math.max(target, 0.1)) * 100));
          const color =
            camera.fps === null
              ? 'var(--track)'
              : achieved < target * 0.8
                ? 'var(--warn)'
                : 'var(--accent)';
          return (
            <div
              key={camera.id}
              className="fps__col"
              title={`${camera.id} · ${camera.fps === null ? 'no engine data' : `${achieved.toFixed(2)} of ${target} fps`}${
                camera.lag_ms != null ? ` · ${Math.round(camera.lag_ms)} ms behind live` : ''
              }`}
            >
              <div style={{ height: `${height}%`, background: color, width: '100%' }} />
            </div>
          );
        })}
      </div>
      <div className="fps__axis mono">
        <span>{cameras[0]?.id}</span>
        <span>target {target} fps</span>
        <span>{cameras[cameras.length - 1]?.id}</span>
      </div>
    </>
  );
}
