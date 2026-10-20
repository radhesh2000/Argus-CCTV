/**
 * Settings — routing, retention, appliance, users.
 *
 * Mostly read-only, and honest about it. The routing table is the real
 * `smtp.recipients` map from argus.yaml joined to the policies that reference
 * each role, so an operator can see exactly who a confirmed violation reaches.
 * WhatsApp and Slack are in the design and in the roadmap but not in this
 * build, so they are listed as unavailable rather than omitted — a plant
 * deciding whether to buy should see what is wired today.
 *
 * The users block used to be the same discipline pointed the other way: an
 * empty state saying this appliance had no authentication, because a fabricated
 * roster would have implied access control that did not exist. Phase 5 made it
 * exist, so the block is now the real roster — for an ADMIN. Everyone else sees
 * what the appliance's account policy is and who to ask, which is the honest
 * version of "you may not see this".
 */

import { api } from '../api/client';
import { UsersPanel } from '../components/UsersPanel';
import { SeverityGlyph, ErrorBar } from '../components/marks';
import { useResource } from '../lib/hooks';
import { bytesFromMb, duration } from '../lib/format';
import { useAppState } from '../state/AppState';
import { useSession } from '../state/Session';

/**
 * The config file, said the way an operator would say it.
 *
 * The appliance knows its config as an absolute path, and on this machine that
 * path begins `/home/arjunvs/Desktop/Projects/…` — a developer's working copy,
 * printed on the settings screen of a product. On a customer's box it will be a
 * different absolute path and still a filesystem path, which is not what the
 * person reading this screen needs: they need the name of the file, and their
 * installer already told them where the install lives. The full path stays in
 * the tooltip for the one person who does want it.
 */
function shortConfigPath(source: string | null | undefined): string {
  if (!source) return 'config/argus.yaml';
  const parts = source.split('/').filter(Boolean);
  return parts.slice(-2).join('/');
}

export function Settings() {
  const { config, policies, configError } = useAppState();
  const { user, can } = useSession();
  const health = useResource(() => api.health(), []);

  if (!config) return <ErrorBar error={configError ?? 'loading configuration'} />;

  const routed = policies.filter((policy) => policy.routed_to);
  const unrouted = policies.filter((policy) => !policy.routed_to);

  return (
    <div className="rise settings">
      <div className="settings__inner">
        <div className="screen-title">Settings</div>
        <ErrorBar error={health.error} />

        <div className="panel">
          <div style={{ fontSize: 'var(--fs-meta)', fontWeight: 600, marginBottom: 4 }}>Alert routing</div>
          <div style={{ fontSize: 'var(--fs-note)', color: 'var(--faint)', marginBottom: 14 }}>
            Who is notified when an auto-routed policy raises a violation. Defined in{' '}
            <span className="mono">smtp.recipients</span> in{' '}
            <span className="mono" title={config.source ?? undefined}>
              {shortConfigPath(config.source)}
            </span>
            .
          </div>

          <div className="routing__head">
            <span>VIOLATION TYPE</span>
            <span>ROUTES TO</span>
            <span>CHANNEL</span>
          </div>
          {routed.map((policy) => {
            const addresses = config.notify_recipients[policy.routed_to ?? ''] ?? [];
            return (
              <div key={policy.id} className="routing__row">
                <span style={{ fontSize: 'var(--fs-meta)', display: 'flex', alignItems: 'center', gap: 8 }}>
                  <SeverityGlyph severity={policy.severity} size={9} />
                  {policy.label}
                </span>
                <span style={{ fontSize: 'var(--fs-meta)', color: 'var(--dim)' }}>
                  {policy.routed_to}
                  {addresses.length > 0 ? (
                    <span className="mono" style={{ color: 'var(--faint)', fontSize: 'var(--fs-note)' }}>
                      {' '}
                      · {addresses.join(', ')}
                    </span>
                  ) : (
                    <span style={{ color: 'var(--crit)', fontSize: 'var(--fs-note)' }}>
                      {' '}
                      · no recipients configured for this role
                    </span>
                  )}
                </span>
                <span className="mono" style={{ fontSize: 'var(--fs-note)', color: 'var(--accent-text)' }}>
                  {policy.stage === 'auto_routed' ? 'EMAIL' : 'IN-APP ONLY (below auto-routed)'}
                </span>
              </div>
            );
          })}
          {unrouted.map((policy) => (
            <div key={policy.id} className="routing__row">
              <span style={{ fontSize: 'var(--fs-meta)', display: 'flex', alignItems: 'center', gap: 8 }}>
                <SeverityGlyph severity={policy.severity} size={9} />
                {policy.label}
              </span>
              <span style={{ fontSize: 'var(--fs-meta)', color: 'var(--faint)' }}>no role set</span>
              <span className="mono" style={{ fontSize: 'var(--fs-note)', color: 'var(--faint)' }}>
                IN-APP ONLY
              </span>
            </div>
          ))}

          <div className="settings__channels">
            <span className="eyebrow">Channels</span>
            <span className="channel channel--on">
              ● EMAIL · {config.smtp.host}:{config.smtp.port}
            </span>
            <span className="channel">○ WHATSAPP · not enabled (Phase 5)</span>
            <span className="channel">○ SLACK · not enabled (Phase 5)</span>
          </div>
        </div>

        <div className="settings__split">
          {can('admin') ? (
            <UsersPanel minPasswordLength={config.auth.min_password_length} />
          ) : (
            <div className="panel">
              <div style={{ fontSize: 'var(--fs-meta)', fontWeight: 600, marginBottom: 10 }}>
                Users &amp; roles
              </div>
              <div className="users__note">
                You are signed in as <span className="mono">{user?.username}</span> with the{' '}
                <span className="mono">{user?.role}</span> role —{' '}
                {config.auth.role_descriptions[user?.role ?? 'REVIEW']}
                <br />
                <br />
                The roster is visible to administrators only. Sessions on this appliance last{' '}
                <span className="mono">{config.auth.session_ttl_hours} h</span>
                {config.auth.session_idle_minutes > 0 ? (
                  <>
                    {' '}
                    and end after{' '}
                    <span className="mono">{config.auth.session_idle_minutes} min</span> idle
                  </>
                ) : null}
                . Ask an administrator for a role change or a password reset.
              </div>
            </div>
          )}

          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div className="panel">
              <div style={{ fontSize: 'var(--fs-meta)', fontWeight: 600, marginBottom: 10 }}>Retention</div>
              <div style={{ fontSize: 'var(--fs-note)', color: 'var(--dim)', lineHeight: 1.9 }}>
                {config.retention.evidence_tiered ? (
                  <>
                    Evidence is kept for as long as it turns out to be worth: shadow{' '}
                    <span className="mono">{config.retention.evidence_days.shadow} d</span> ·
                    awaiting review{' '}
                    <span className="mono">{config.retention.evidence_days.reviewed} d</span> ·
                    auto-routed{' '}
                    <span className="mono">{config.retention.evidence_days.auto_routed} d</span> ·
                    confirmed{' '}
                    <span className="mono">{config.retention.evidence_days.confirmed} d</span> ·
                    false positive{' '}
                    <span className="mono">
                      {config.retention.evidence_days.false_positive} d
                    </span>
                  </>
                ) : (
                  <>
                    Evidence clips <span className="mono">{config.retention.clips_days} days</span>{' '}
                    · Snapshots{' '}
                    <span className="mono">{config.retention.snapshots_days} days</span>
                  </>
                )}
                <br />
                Event records <span className="mono">{config.retention.events_days} days</span> —
                a record outlives its video, so an aged-out event still reads as an event.
                <br />
                {config.retention.max_clips_gb ? (
                  <>
                    Hard budget:{' '}
                    <span className="mono">{config.retention.max_clips_gb} GB</span> of clips
                    {config.retention.max_snapshots_gb ? (
                      <>
                        {' '}
                        and{' '}
                        <span className="mono">{config.retention.max_snapshots_gb} GB</span> of
                        snapshots
                      </>
                    ) : null}
                    . Over it, the sweep evicts lowest-stage-first.
                    {health.data ? (
                      <>
                        {' '}
                        Holding{' '}
                        <span className="mono">
                          {Math.round(health.data.disk.evidence.used_gb)} GB
                        </span>
                        .
                      </>
                    ) : null}
                    <br />
                  </>
                ) : null}
                Disk guard at <span className="mono">{config.retention.max_disk_percent}%</span>{' '}
                reclaims oldest-first.
                {health.data ? (
                  <>
                    {' '}
                    Currently <span className="mono">{health.data.disk.percent}%</span> full,{' '}
                    <span className="mono">{bytesFromMb(health.data.disk.argus_mb)}</span> of it
                    Argus.
                  </>
                ) : null}
                <br />
                All data stays on the appliance. Nothing leaves the plant network.
              </div>
            </div>

            <div className="panel">
              <div style={{ fontSize: 'var(--fs-meta)', fontWeight: 600, marginBottom: 10 }}>Appliance</div>
              <div style={{ fontSize: 'var(--fs-note)', color: 'var(--dim)', lineHeight: 1.9 }}>
                {health.data?.gpu.gpus[0]?.name ?? 'GPU not probed'} · Argus{' '}
                <span className="mono">{config.version}</span>
                <br />
                Detector <span className="mono">rf-detr-{config.detection.model}</span>{' '}
                {config.detection.precision} at {config.detection.fps} fps ·{' '}
                {config.detection.width}×{config.detection.height}
                <br />
                Engine{' '}
                <span
                  className="mono"
                  style={{ color: health.data?.engine.running ? 'var(--good)' : 'var(--crit)' }}
                >
                  {health.data?.engine.running ? 'running' : 'stopped'}
                </span>
                {health.data?.engine.uptime_sec
                  ? ` · up ${duration(health.data.engine.uptime_sec)}`
                  : ''}
                <br />
                Gateway{' '}
                <span className="mono">
                  go2rtc {config.gateway.api_host}:{config.gateway.api_port}
                </span>{' '}
                {config.gateway.api_host === '127.0.0.1' ? (
                  <span style={{ color: 'var(--good)' }}>· loopback only</span>
                ) : (
                  <span style={{ color: 'var(--crit)' }}>
                    · REACHABLE ON THE NETWORK — unauthenticated
                  </span>
                )}
                <br />
                Live video signalling is relayed through{' '}
                <span className="mono">{config.gateway.relay}</span> on this session. WebRTC media
                arrives on <span className="mono">:{config.gateway.webrtc_port}</span>, which ICE
                connects to directly — see docs/SECURITY.md.
                <br />
                Air-gapped mode <span className="mono">ON</span> — no external requests, fonts and
                assets bundled.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
