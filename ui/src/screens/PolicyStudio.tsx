/**
 * Policy Studio — the staged rollout, made operable, and now writable.
 *
 * The card grid is every policy on the appliance with its real precision from
 * human verdicts and, since schema v5, where it applies. The inspector on the
 * right does three jobs, and they are deliberately three panels rather than
 * three screens:
 *
 * * **Author.** Create and edit policies — prompt, interval, cooldown,
 *   severity, routing, and the camera/zone scope. Nothing here can set `stage`
 *   or `lifecycle`: a new policy is born draft/shadow and the API refuses
 *   anything else, so the console cannot invent a shortcut past the gate.
 * * **Test.** Run the prompt in the box against that camera's current frame and
 *   stream the answer back. This is the panel the design drew and the one that
 *   was disabled until the engine could publish a frame the API can reach.
 * * **Promote.** draft -> pilot -> production, with the precision bar the
 *   promotion decision is supposed to rest on.
 *
 * Everything that writes is behind `can('manage')` — RESPOND and ADMIN. A
 * REVIEW account sees the same policies, read-only, because a reviewer triaging
 * an alert has to be able to see what raised it.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api, testPrompt } from '../api/client';
import type { PolicyDraft, PromptTestEvent } from '../api/client';
import type {
  Camera,
  DetectorAccuracy,
  Policy,
  PolicyScope,
  Severity,
  VlmHealth,
  VlmSchedule,
  Zone,
} from '../api/types';
import { StageBadge, ErrorBar } from '../components/marks';
import { useResource } from '../lib/hooks';
import { LIFECYCLE_LABEL, count, duration, percent } from '../lib/format';
import { useAppState } from '../state/AppState';
import { useSession } from '../state/Session';

/** The bar a policy has to clear before it may notify anybody unattended.
 *
 *  Fallbacks only. The real numbers come from `/api/config` (`rollout:`),
 *  because the appliance enforces them — this used to be the *only* copy, which
 *  made the gate a disabled button rather than a rule. These values are what
 *  the screen draws for the half-second before config lands.
 */
const DEFAULT_PRECISION = 0.95;
const DEFAULT_REVIEWS = 200;

const BLANK_VLM: PolicyDraft = {
  kind: 'vlm',
  detector_key: 'vlm_prompt',
  label: '',
  prompt: '',
  severity: 'medium',
  interval_sec: 8,
  cooldown_sec: 60,
  enabled: true,
  scope: [],
};

export function PolicyStudio() {
  const { policies, config, reloadPolicies } = useAppState();
  const { can } = useSession();
  const mayEdit = can('manage');
  const accuracy = useResource(() => api.accuracy(30), []);
  const fleet = useResource(() => api.cameras(), []);
  // Fetched once rather than polled: this is an authoring screen, and the panel
  // it feeds answers "is this policy running", not "what is happening now".
  const health = useResource(() => api.health(), []);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [draft, setDraft] = useState<PolicyDraft | null>(null);
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);

  const cameras = useMemo(() => fleet.data?.cameras ?? [], [fleet.data]);

  const byDetector = useMemo(() => {
    const map: Record<string, DetectorAccuracy> = {};
    for (const row of accuracy.data?.detectors ?? []) map[row.detector_name] = row;
    return map;
  }, [accuracy.data]);

  const selected = policies.find((policy) => policy.id === selectedId) ?? policies[0] ?? null;
  const stats = selected ? byDetector[selected.id] ?? null : null;

  const bar = {
    precision: config?.rollout?.min_precision ?? DEFAULT_PRECISION,
    reviews: config?.rollout?.min_reviews ?? DEFAULT_REVIEWS,
  };
  const eligible =
    stats !== null &&
    stats.precision !== null &&
    stats.precision >= bar.precision &&
    stats.reviewed >= bar.reviews;

  /** Every write on this screen funnels through here, so the busy flag, the
   *  error bar and the reload happen once rather than in nine handlers. */
  const act = useCallback(
    async (work: () => Promise<unknown>) => {
      setBusy(true);
      setFailure(null);
      try {
        await work();
        reloadPolicies();
        return true;
      } catch (error) {
        setFailure(error instanceof Error ? error.message : String(error));
        return false;
      } finally {
        setBusy(false);
      }
    },
    [reloadPolicies],
  );

  const save = async () => {
    if (!draft) return;
    const creating = !draft.id;
    const ok = await act(async () => {
      if (creating) {
        const { policy } = await api.createPolicy(draft);
        setSelectedId(policy.id);
      } else {
        await api.updatePolicy(draft.id!, {
          label: draft.label,
          enabled: draft.enabled,
          severity: draft.severity,
          routed_to: draft.routed_to,
          prompt: draft.kind === 'vlm' ? draft.prompt : undefined,
          interval_sec: draft.kind === 'vlm' ? draft.interval_sec : undefined,
          cooldown_sec: draft.cooldown_sec,
          threshold: draft.kind === 'detector' ? draft.threshold : undefined,
          scope: draft.scope,
        });
      }
    });
    if (ok) setDraft(null);
  };

  return (
    <div className="rise studio">
      <div className="studio__list">
        <div className="studio__head">
          <div className="screen-title">Policy Studio</div>
          {mayEdit ? (
            <button className="button" onClick={() => setDraft({ ...BLANK_VLM })}>
              + New policy
            </button>
          ) : (
            <button
              className="button button--disabled"
              disabled
              title="Authoring policies needs the RESPOND or ADMIN role."
            >
              + New policy
            </button>
          )}
        </div>

        <ErrorBar error={failure ?? accuracy.error ?? health.error ?? fleet.error} />

        <div className="studio__grid">
          {policies.map((policy) => (
            <PolicyCard
              key={policy.id}
              policy={policy}
              stats={byDetector[policy.id] ?? null}
              cameras={cameras}
              selected={selected?.id === policy.id && draft === null}
              onSelect={() => {
                setDraft(null);
                setSelectedId(policy.id);
              }}
            />
          ))}
          {policies.length === 0 ? (
            <div className="notice">
              <div className="notice__title">No policies defined</div>
              <div className="notice__body">
                Run <code>argus seed</code> to import config/seed.yaml, or write one here.
              </div>
            </div>
          ) : null}
        </div>
      </div>

      <aside className="studio__inspector">
        {draft !== null ? (
          <PolicyEditor
            draft={draft}
            cameras={cameras}
            roles={config?.notify_roles ?? []}
            busy={busy}
            onChange={setDraft}
            onSave={save}
            onCancel={() => setDraft(null)}
          />
        ) : selected ? (
          <>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <span style={{ fontWeight: 600, fontSize: 'var(--fs-body)' }}>{selected.label}</span>
                <StageBadge stage={selected.stage} />
                {mayEdit ? (
                  <button
                    className="button button--quiet"
                    style={{ marginLeft: 'auto' }}
                    onClick={() => setDraft(toDraft(selected))}
                  >
                    Edit
                  </button>
                ) : null}
              </div>
              <div className="mono" style={{ fontSize: 'var(--fs-note)', color: 'var(--faint)', marginTop: 5 }}>
                {selected.kind === 'vlm' ? 'VLM PROMPT' : 'FAST DETECTOR'} ·{' '}
                {LIFECYCLE_LABEL[selected.lifecycle] ?? selected.lifecycle} ·{' '}
                {selected.interval_sec
                  ? `SAMPLING 1 FRAME / ${selected.interval_sec} S`
                  : `DETECTION ${config?.detection.fps ?? '—'} FPS`}
                {' · COOLDOWN '}
                {selected.cooldown_sec}S · {scopeLabel(selected.scope, cameras).toUpperCase()}
              </div>
            </div>

            <ScopePanel scope={selected.scope} cameras={cameras} />

            {selected.kind === 'vlm' ? (
              <>
                <div>
                  <div className="eyebrow" style={{ marginBottom: 7 }}>
                    Prompt
                  </div>
                  <div className="studio__prompt">{selected.prompt ?? '—'}</div>
                </div>
                <div>
                  <div className="eyebrow" style={{ marginBottom: 7 }}>
                    Structured answer
                  </div>
                  <div className="studio__schema mono">
                    {'{"triggered": bool,'}
                    <br />
                    &nbsp;{'"confidence": float,'}
                    <br />
                    &nbsp;{'"reason": string}'}
                  </div>
                </div>
                <PolicyRuntime vlm={health.data?.vlm ?? null} policyId={selected.id} />
                {mayEdit ? (
                  <LiveTest policy={selected} cameras={cameras} prompt={selected.prompt ?? ''} />
                ) : null}
              </>
            ) : (
              <div>
                <div className="eyebrow" style={{ marginBottom: 7 }}>
                  Rule
                </div>
                <div className="studio__prompt">
                  Count people inside each zone on every sampled frame and raise an event when the
                  count exceeds that zone's sanctioned headcount. Boxes, track ids and the zone
                  polygon are stored with the event so a reviewer sees exactly what was counted.
                </div>
              </div>
            )}

            <div className="studio__facts">
              <span>
                Severity <span className="mono">{selected.severity}</span>
              </span>
              <span>
                Routes to <span className="mono">{selected.routed_to ?? 'nobody'}</span>
              </span>
              <span>
                Enabled <span className="mono">{selected.enabled ? 'yes' : 'no'}</span>
              </span>
            </div>

            <PromotionGate
              policy={selected}
              stats={stats}
              bar={bar}
              eligible={eligible}
              busy={busy}
              mayEdit={mayEdit}
              allowOverride={config?.rollout?.allow_override ?? false}
              onStartPilot={() => act(() => api.startPilot(selected.id))}
              onExtend={() => act(() => api.extendPilot(selected.id))}
              onPromote={(reason) =>
                act(() =>
                  api.promotePolicy(selected.id, {
                    stage: 'auto_routed',
                    ...(reason ? { override: true, override_reason: reason } : {}),
                  }),
                )
              }
            />
          </>
        ) : null}
      </aside>
    </div>
  );
}

// --------------------------------------------------------------------------
// scope
// --------------------------------------------------------------------------

function toDraft(policy: Policy): PolicyDraft {
  return {
    id: policy.id,
    kind: policy.kind,
    detector_key: policy.detector_key,
    label: policy.label,
    enabled: Boolean(policy.enabled),
    severity: policy.severity,
    routed_to: policy.routed_to,
    prompt: policy.prompt,
    interval_sec: policy.interval_sec,
    cooldown_sec: policy.cooldown_sec,
    threshold: policy.threshold,
    scope: policy.scope ?? [],
  };
}

/** "Every camera", "2 cameras", "Conveyor side" — what the card shows. */
function scopeLabel(scope: PolicyScope[] | undefined, cameras: Camera[]): string {
  const rows = scope ?? [];
  if (rows.length === 0) return 'every camera';
  const zones = rows.filter((row) => row.zone_id !== null);
  if (rows.length === 1) {
    const row = rows[0]!;
    const camera = cameras.find((entry) => entry.id === row.camera_id);
    if (row.zone_id === null) return camera?.name ?? row.camera_id;
    const zone = camera?.zones?.find((entry) => entry.id === row.zone_id);
    return zone?.name ?? row.zone_id;
  }
  const uniqueCameras = new Set(rows.map((row) => row.camera_id)).size;
  if (zones.length === rows.length) return `${zones.length} zones`;
  return `${uniqueCameras} camera${uniqueCameras === 1 ? '' : 's'}`;
}

function ScopePanel({ scope, cameras }: { scope: PolicyScope[]; cameras: Camera[] }) {
  const rows = scope ?? [];
  return (
    <div>
      <div className="eyebrow" style={{ marginBottom: 7 }}>
        Applies to
      </div>
      {rows.length === 0 ? (
        <div style={{ fontSize: 'var(--fs-note)', color: 'var(--dim)', lineHeight: 1.6 }}>
          <span className="mono">Every camera</span> — including any enrolled later. That is what
          every policy meant before scoping existed, so an empty scope is "everywhere", never
          "nowhere".
        </div>
      ) : (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {rows.map((row) => {
            const camera = cameras.find((entry) => entry.id === row.camera_id);
            const zone = camera?.zones?.find((entry) => entry.id === row.zone_id);
            return (
              <span className="chip-button" key={`${row.camera_id}:${row.zone_id ?? ''}`}>
                {camera?.name ?? row.camera_id}
                {zone ? ` · ${zone.name}` : ''}
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}

function hasScope(scope: PolicyScope[], cameraId: string, zoneId: string | null): boolean {
  return scope.some((row) => row.camera_id === cameraId && row.zone_id === zoneId);
}

function toggleScope(
  scope: PolicyScope[],
  cameraId: string,
  zoneId: string | null,
): PolicyScope[] {
  if (hasScope(scope, cameraId, zoneId)) {
    return scope.filter((row) => !(row.camera_id === cameraId && row.zone_id === zoneId));
  }
  const added = [...scope, { camera_id: cameraId, zone_id: zoneId }];
  // A whole-camera row swallows the zone rows beside it, because that is what
  // the appliance does with them (db.scope_zones). Leaving both would show a
  // scope the engine does not honour.
  return zoneId === null
    ? added.filter((row) => row.camera_id !== cameraId || row.zone_id === null)
    : added;
}

function ScopeEditor({
  scope,
  cameras,
  onChange,
}: {
  scope: PolicyScope[];
  cameras: Camera[];
  onChange: (next: PolicyScope[]) => void;
}) {
  return (
    <div>
      <div className="eyebrow" style={{ marginBottom: 6 }}>
        Applies to
      </div>
      <div className="scope">
        <button
          className={`scope__row ${scope.length === 0 ? 'scope__row--on' : ''}`}
          onClick={() => onChange([])}
        >
          <span>Every camera</span>
          <span className="mono scope__hint">including any enrolled later</span>
        </button>
        {cameras.map((camera) => {
          const wholeCamera = hasScope(scope, camera.id, null);
          return (
            <div className="scope__camera" key={camera.id}>
              <button
                className={`scope__row ${wholeCamera ? 'scope__row--on' : ''}`}
                onClick={() => onChange(toggleScope(scope, camera.id, null))}
              >
                <span>{camera.name}</span>
                <span className="mono scope__hint">{camera.id}</span>
              </button>
              {(camera.zones ?? []).length > 0 ? (
                <div className="scope__zones">
                  {(camera.zones ?? []).map((zone: Zone) => (
                    <button
                      key={zone.id}
                      className={`chip-button ${
                        hasScope(scope, camera.id, zone.id) ? 'chip-button--on' : ''
                      }`}
                      disabled={wholeCamera}
                      title={
                        wholeCamera
                          ? 'The whole camera is selected, which already includes this zone.'
                          : 'Scope to this zone. A prompt policy is shown the crop of it.'
                      }
                      onClick={() => onChange(toggleScope(scope, camera.id, zone.id))}
                    >
                      {zone.name}
                    </button>
                  ))}
                </div>
              ) : null}
            </div>
          );
        })}
      </div>
      <div style={{ fontSize: 'var(--fs-note)', color: 'var(--faint)', marginTop: 7, lineHeight: 1.7 }}>
        A zone narrows a detector policy to that zone, and shows a prompt policy the crop of it
        rather than the whole view — measured at 12/12 against 3/12 on the wide shot for a counting
        question (docs/VLM_EVAL.md).
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------
// the editor
// --------------------------------------------------------------------------

function PolicyEditor({
  draft,
  cameras,
  roles,
  busy,
  onChange,
  onSave,
  onCancel,
}: {
  draft: PolicyDraft;
  cameras: Camera[];
  roles: string[];
  busy: boolean;
  onChange: (next: PolicyDraft) => void;
  onSave: () => void;
  onCancel: () => void;
}) {
  const creating = !draft.id;
  const set = (patch: Partial<PolicyDraft>) => onChange({ ...draft, ...patch });
  const complete =
    draft.label.trim().length > 0 &&
    (draft.kind === 'detector' || (draft.prompt ?? '').trim().length > 0);

  return (
    <>
      <div>
        <div className="eyebrow">{creating ? 'New policy' : `Editing · ${draft.id}`}</div>
        {creating ? (
          <div className="policy-edit__kinds">
            {(['vlm', 'detector'] as const).map((kind) => (
              <button
                key={kind}
                className={`chip-button ${draft.kind === kind ? 'chip-button--on' : ''}`}
                onClick={() =>
                  set(
                    kind === 'vlm'
                      ? { kind, detector_key: 'vlm_prompt', interval_sec: 8, threshold: null }
                      : {
                          kind,
                          detector_key: 'manpower_headcount',
                          prompt: null,
                          interval_sec: null,
                        },
                  )
                }
              >
                {kind === 'vlm' ? 'VLM prompt' : 'Fast detector'}
              </button>
            ))}
          </div>
        ) : null}
      </div>

      <div className="policy-edit">
        <label className="policy-edit__field">
          <span className="eyebrow">Name</span>
          <input
            className="field"
            value={draft.label}
            placeholder="No phones on the packing line"
            onChange={(event) => set({ label: event.target.value })}
          />
        </label>

        {draft.kind === 'vlm' ? (
          <label className="policy-edit__field">
            <span className="eyebrow">Prompt</span>
            <textarea
              className="field policy-edit__prompt"
              rows={6}
              value={draft.prompt ?? ''}
              placeholder="Is any worker visibly using a mobile phone while on the production line? Only answer triggered=true if phone use is clearly visible."
              onChange={(event) => set({ prompt: event.target.value })}
            />
            <span className="policy-edit__hint">
              Sent verbatim, plus the appliance's structured-answer instruction. Write the negative
              cases in: "not for a hand near a pocket" is what stops an alert every eight seconds.
            </span>
          </label>
        ) : (
          <label className="policy-edit__field">
            <span className="eyebrow">Threshold</span>
            <input
              className="field"
              type="number"
              step="0.1"
              value={draft.threshold ?? ''}
              onChange={(event) =>
                set({ threshold: event.target.value === '' ? null : Number(event.target.value) })
              }
            />
            <span className="policy-edit__hint">
              What the detector reads as a limit or a confidence floor. The headcount detector takes
              its limit from the zone's sanctioned count, not from here.
            </span>
          </label>
        )}

        <div className="policy-edit__row">
          {draft.kind === 'vlm' ? (
            <label className="policy-edit__field">
              <span className="eyebrow">Sample every (s)</span>
              <input
                className="field"
                type="number"
                min={1}
                value={draft.interval_sec ?? 8}
                onChange={(event) => set({ interval_sec: Number(event.target.value) })}
              />
            </label>
          ) : null}
          <label className="policy-edit__field">
            <span className="eyebrow">Cooldown (s)</span>
            <input
              className="field"
              type="number"
              min={0}
              value={draft.cooldown_sec ?? 60}
              onChange={(event) => set({ cooldown_sec: Number(event.target.value) })}
            />
          </label>
        </div>

        <div className="policy-edit__row">
          <label className="policy-edit__field">
            <span className="eyebrow">Severity</span>
            <select
              className="field"
              value={draft.severity ?? 'medium'}
              onChange={(event) => set({ severity: event.target.value as Severity })}
            >
              <option value="low">low</option>
              <option value="medium">medium</option>
              <option value="high">high</option>
            </select>
          </label>
          <label className="policy-edit__field">
            <span className="eyebrow">Routes to</span>
            <select
              className="field"
              value={draft.routed_to ?? ''}
              onChange={(event) => set({ routed_to: event.target.value || null })}
            >
              <option value="">nobody</option>
              {roles.map((role) => (
                <option key={role} value={role}>
                  {role}
                </option>
              ))}
            </select>
          </label>
        </div>

        <ScopeEditor
          scope={draft.scope ?? []}
          cameras={cameras}
          onChange={(scope) => set({ scope })}
        />

        <label className="policy-edit__toggle">
          <input
            type="checkbox"
            checked={draft.enabled ?? true}
            onChange={(event) => set({ enabled: event.target.checked })}
          />
          <span>Enabled — a disabled policy is stored and never evaluated.</span>
        </label>

        {creating ? (
          <div className="notice">
            <div className="notice__phase">BORN IN SHADOW</div>
            <div className="notice__body">
              A new policy starts in <span className="mono">draft</span> and{' '}
              <span className="mono">shadow</span>: it is not evaluated against live video until you
              start its pilot, and it notifies nobody until it clears the promotion gate. The API
              refuses any other starting point.
            </div>
          </div>
        ) : null}

        <div style={{ display: 'flex', gap: 8 }}>
          <button className="button" disabled={!complete || busy} onClick={onSave}>
            {busy ? 'Saving…' : creating ? 'Create policy' : 'Save changes'}
          </button>
          <button className="button button--ghost" disabled={busy} onClick={onCancel}>
            Cancel
          </button>
        </div>
      </div>
    </>
  );
}

// --------------------------------------------------------------------------
// the live prompt test
// --------------------------------------------------------------------------

/**
 * The design's "test this prompt against a live frame" panel, wired up.
 *
 * Two things had to exist first and now both do: `POST /api/policies/test`, and
 * a way for the API process to obtain a current frame. The second is the
 * interesting one — live frames exist only inside `argus run`, and rather than
 * a second decode path the engine now publishes the frame it just analysed, on
 * request, through `data/live/` (see argus/live.py). So what this panel shows
 * the model is the same picture the detector saw.
 *
 * When there is no engine, this says so and names the reason. It never invents
 * a verdict.
 */
function LiveTest({
  policy,
  cameras,
  prompt,
}: {
  policy: Policy;
  cameras: Camera[];
  prompt: string;
}) {
  const scoped = policy.scope ?? [];
  const firstScoped = scoped[0];
  const [cameraId, setCameraId] = useState<string>(
    firstScoped?.camera_id ?? cameras[0]?.id ?? '',
  );
  const [zoneId, setZoneId] = useState<string | null>(firstScoped?.zone_id ?? null);
  const [frame, setFrame] = useState<Extract<PromptTestEvent, { name: 'frame' }>['data'] | null>(
    null,
  );
  const [verdict, setVerdict] = useState<
    Extract<PromptTestEvent, { name: 'verdict' }>['data'] | null
  >(null);
  const [asking, setAsking] = useState(false);
  const [problem, setProblem] = useState<{ code: string; detail: string } | null>(null);
  const abort = useRef<(() => void) | null>(null);

  const camera = cameras.find((entry) => entry.id === cameraId) ?? null;
  const zones = camera?.zones ?? [];

  // A panel that is closed (or a policy that is switched) mid-inference must
  // not keep the socket, and must not call back into an unmounted component.
  useEffect(() => () => abort.current?.(), []);
  useEffect(() => {
    setFrame(null);
    setVerdict(null);
    setProblem(null);
  }, [policy.id]);

  const run = () => {
    abort.current?.();
    setFrame(null);
    setVerdict(null);
    setProblem(null);
    setAsking(true);
    abort.current = testPrompt(
      { camera_id: cameraId, prompt, policy_id: policy.id, zone_id: zoneId },
      (event) => {
        if (event.name === 'frame') setFrame(event.data);
        if (event.name === 'verdict') {
          setVerdict(event.data);
          setAsking(false);
        }
        if (event.name === 'error') {
          setProblem(event.data);
          setAsking(false);
        }
      },
    );
  };

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 7 }}>
        <span className="eyebrow">Live test</span>
        <span style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
          <select
            className="field field--inline"
            value={cameraId}
            onChange={(event) => {
              setCameraId(event.target.value);
              setZoneId(null);
            }}
          >
            {cameras.map((entry) => (
              <option key={entry.id} value={entry.id}>
                {entry.name}
              </option>
            ))}
          </select>
          <select
            className="field field--inline"
            value={zoneId ?? ''}
            onChange={(event) => setZoneId(event.target.value || null)}
          >
            <option value="">whole view</option>
            {zones.map((zone) => (
              <option key={zone.id} value={zone.id}>
                {zone.name}
              </option>
            ))}
          </select>
        </span>
      </div>

      <div className="livetest">
        <div className="livetest__frame">
          {frame ? (
            <>
              <img src={frame.image} alt={`current frame from ${frame.camera_id}`} />
              {verdict?.box_2d ? (
                <div
                  className="livetest__box"
                  style={{
                    left: `${verdict.box_2d[0] * 100}%`,
                    top: `${verdict.box_2d[1] * 100}%`,
                    width: `${(verdict.box_2d[2] - verdict.box_2d[0]) * 100}%`,
                    height: `${(verdict.box_2d[3] - verdict.box_2d[1]) * 100}%`,
                  }}
                />
              ) : null}
              <div className="livetest__stamp mono">
                {frame.cropped ? `${frame.zone_name} crop · ` : ''}
                {frame.width}×{frame.height} · {frame.age_sec.toFixed(1)}s old
              </div>
            </>
          ) : (
            <div className="livetest__empty mono">
              {asking ? 'WAITING FOR A FRAME…' : 'NO FRAME YET'}
            </div>
          )}
        </div>
        <div className="livetest__answer mono">
          {verdict ? (
            <>
              {verdict.reason ?? '—'}
              <br />
              <span style={{ color: verdict.triggered ? 'var(--warn)' : 'var(--good)' }}>
                triggered: {String(verdict.triggered)} · confidence{' '}
                {verdict.confidence.toFixed(2)}
              </span>{' '}
              <span style={{ color: 'var(--faint)' }}>
                · {duration(verdict.latency_sec)} · {verdict.model}
              </span>
            </>
          ) : problem ? (
            <span style={{ color: problem.code === 'vlm_disabled' ? 'var(--faint)' : 'var(--crit)' }}>
              {problem.detail}
            </span>
          ) : asking ? (
            <>
              asking the model…
              <span className="caret" />
            </>
          ) : (
            <span style={{ color: 'var(--faint)' }}>
              Runs the prompt above against this camera's current frame. Nothing is stored — no
              event, no snapshot, no alert.
            </span>
          )}
        </div>
      </div>

      <div style={{ display: 'flex', gap: 8, marginTop: 9 }}>
        <button className="button button--soft" disabled={asking || !cameraId} onClick={run}>
          {asking ? 'Running…' : frame ? 'Run again ↻' : 'Test against live frame'}
        </button>
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------
// runtime, cards, gate
// --------------------------------------------------------------------------

/**
 * What the scheduler is doing with this prompt policy right now.
 *
 * One schedule exists per (policy × camera × scoped zone), and each one carries
 * its own counters — so this is the place an author finds out that their
 * 6-second interval is really running at the model's latency, which the
 * `dropped` column says out loud.
 */
function PolicyRuntime({ vlm, policyId }: { vlm: VlmHealth | null; policyId: string }) {
  if (!vlm) return null;

  const schedules = (vlm.schedules ?? []).filter((entry) => entry.policy_id === policyId);

  if (!vlm.enabled) {
    return (
      <div className="notice">
        <div className="notice__phase">NOT EVALUATED</div>
        <div className="notice__title">Prompt policies are switched off on this appliance</div>
        <div className="notice__body">
          <code>vlm.enabled</code> is false in the appliance config, so this policy is stored and
          never asked. Everything else about it — stage, routing, promotion gate — is unchanged.
        </div>
      </div>
    );
  }

  if (!vlm.available) {
    return (
      <div className="notice">
        <div className="notice__phase">DEGRADED</div>
        <div className="notice__title">The model is not answering</div>
        <div className="notice__body">
          This policy is scheduled but raises nothing while{' '}
          <span className="mono">{vlm.configured_model}</span> is down. Detector policies are
          unaffected.
          <br />
          <span className="mono" style={{ fontSize: 'var(--fs-note)', color: 'var(--warn)' }}>
            {vlm.reason ?? 'no reason reported'}
          </span>
        </div>
      </div>
    );
  }

  if (vlm.source === 'probe') {
    return (
      <div className="notice">
        <div className="notice__phase">MODEL UP · ENGINE STOPPED</div>
        <div className="notice__title">Nothing is sampling frames for this policy</div>
        <div className="notice__body">
          <span className="mono">{vlm.model ?? vlm.configured_model}</span> is serving at{' '}
          <span className="mono">{vlm.endpoint}</span>, but the schedule lives inside{' '}
          <code>argus run</code> and it is not running.
        </div>
      </div>
    );
  }

  if (schedules.length === 0) {
    return (
      <div className="notice">
        <div className="notice__phase">NOT SCHEDULED</div>
        <div className="notice__title">Ready, but this policy is not in the schedule</div>
        <div className="notice__body">
          The scheduler runs enabled policies past <code>draft</code> wherever they are scoped —
          every enabled camera if the scope is empty. This one qualifies on neither count, or the
          engine has not picked up the edit yet; it re-reads policies every 15 s.
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="eyebrow" style={{ marginBottom: 7 }}>
        Running against
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
        {schedules.map((schedule) => (
          <ScheduleLine key={schedule.key ?? schedule.camera_id} schedule={schedule} />
        ))}
      </div>
      <div style={{ fontSize: 'var(--fs-note)', color: 'var(--faint)', marginTop: 9, lineHeight: 1.7 }}>
        {vlm.inference
          ? `${vlm.inference.avg_latency_sec.toFixed(2)} s per evaluation on this box. A sample whose
             predecessor has not come back is dropped, not queued — so the effective interval is
             the model's latency whenever it exceeds the policy's.`
          : null}
      </div>
    </div>
  );
}

function ScheduleLine({ schedule }: { schedule: VlmSchedule }) {
  return (
    <div className="studio__schedule">
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
        <span className="mono" style={{ fontSize: 'var(--fs-note)', color: 'var(--text2)' }}>
          {schedule.camera_id}
          {schedule.zone_name ? ` · ${schedule.zone_name}` : ''}
        </span>
        <span className="mono" style={{ fontSize: 'var(--fs-note)', color: 'var(--faint)' }}>
          1 FRAME / {schedule.interval_sec} S · {count(schedule.evaluations)} EVALS ·{' '}
          {count(schedule.events)} EVENTS
          {schedule.cropped ? ' · CROPPED' : ''}
          {schedule.skipped_in_flight > 0 ? (
            <span style={{ color: 'var(--warn)' }}> · {count(schedule.skipped_in_flight)} DROPPED</span>
          ) : null}
        </span>
      </div>
      <div style={{ fontSize: 'var(--fs-note)', color: 'var(--dim)', lineHeight: 1.6 }}>
        {schedule.last_error ? (
          <span style={{ color: 'var(--crit)' }}>{schedule.last_error}</span>
        ) : schedule.last_verdict ? (
          <>
            {schedule.last_verdict.triggered ? '▲ ' : '○ '}
            {schedule.last_verdict.reason ?? '—'}{' '}
            <span className="mono" style={{ fontSize: 'var(--fs-note)', color: 'var(--faint)' }}>
              ({percent(schedule.last_verdict.confidence, 0)} ·{' '}
              {duration(schedule.last_verdict.latency_sec)})
            </span>
          </>
        ) : (
          <span style={{ color: 'var(--faint)' }}>no verdict yet</span>
        )}
      </div>
    </div>
  );
}

function PolicyCard({
  policy,
  stats,
  cameras,
  selected,
  onSelect,
}: {
  policy: Policy;
  stats: DetectorAccuracy | null;
  cameras: Camera[];
  selected: boolean;
  onSelect: () => void;
}) {
  const precision = stats?.precision ?? null;
  const color =
    precision === null
      ? 'var(--faint)'
      : precision >= DEFAULT_PRECISION
        ? 'var(--good)'
        : 'var(--warn)';
  return (
    <button className={`policy ${selected ? 'policy--on' : ''}`} onClick={onSelect}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ fontWeight: 600, fontSize: 'var(--fs-body)', flex: 1, textAlign: 'left' }}>
          {policy.label}
        </span>
        <StageBadge stage={policy.stage} />
      </div>
      <div className="mono policy__engine">
        {policy.kind === 'vlm' ? 'VLM PROMPT' : 'FAST DETECTOR'} ·{' '}
        {LIFECYCLE_LABEL[policy.lifecycle] ?? policy.lifecycle} ·{' '}
        {scopeLabel(policy.scope, cameras).toUpperCase()}
        {policy.enabled ? '' : ' · DISABLED'}
      </div>
      <div className="policy__stats">
        <div>
          <div className="mono policy__figure" style={{ color }}>
            {percent(precision)}
          </div>
          <div className="policy__caption">precision</div>
        </div>
        <div>
          <div className="mono policy__figure">{count(stats?.reviewed ?? 0)}</div>
          <div className="policy__caption">reviewed</div>
        </div>
        <div style={{ marginLeft: 'auto', alignSelf: 'flex-end', color: 'var(--dim)', fontSize: 'var(--fs-note)' }}>
          {policy.routed_to ? `→ ${policy.routed_to}` : 'silent'}
        </div>
      </div>
    </button>
  );
}

function PromotionGate({
  policy,
  stats,
  bar,
  eligible,
  busy,
  mayEdit,
  allowOverride,
  onStartPilot,
  onExtend,
  onPromote,
}: {
  policy: Policy;
  stats: DetectorAccuracy | null;
  bar: { precision: number; reviews: number };
  eligible: boolean;
  busy: boolean;
  mayEdit: boolean;
  allowOverride: boolean;
  onStartPilot: () => void;
  onExtend: () => void;
  /** `reason` set means "promote past the gate". The API refuses an override
   *  without one and writes it onto the policy, so this control cannot be a
   *  bare button. */
  onPromote: (reason?: string) => void;
}) {
  const precision = stats?.precision ?? null;
  const reviewed = stats?.reviewed ?? 0;
  const already = policy.stage === 'auto_routed';
  const draft = policy.lifecycle === 'draft';
  const piloting = policy.lifecycle === 'pilot';
  const [override, setOverride] = useState('');
  const [overriding, setOverriding] = useState(false);

  return (
    <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div className="eyebrow">
        {draft ? 'Draft — not yet running' : 'Promotion to auto-routed'}
      </div>

      {draft ? (
        <>
          <div style={{ fontSize: 'var(--fs-note)', color: 'var(--dim)', lineHeight: 1.6 }}>
            A draft is stored and never evaluated. Starting the pilot puts it in front of live
            video in <span className="mono">shadow</span> — raising events nobody is notified about
            — so the precision below is measured before anybody's phone rings.
          </div>
          <button
            className={mayEdit ? 'button' : 'button button--disabled'}
            disabled={!mayEdit || busy}
            style={{ alignSelf: 'flex-start' }}
            onClick={onStartPilot}
          >
            Start {policy.pilot.days_total}-day pilot
          </button>
        </>
      ) : (
        <>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div className="gate__track">
              <div
                className="gate__fill"
                style={{
                  width: `${Math.min(100, (precision ?? 0) * 100)}%`,
                  background: eligible ? 'var(--good)' : 'var(--warn)',
                }}
              />
              <div className="gate__mark" style={{ left: `${bar.precision * 100}%` }} />
            </div>
            <span
              className="mono"
              style={{ fontSize: 'var(--fs-note)', color: eligible ? 'var(--good)' : 'var(--warn)' }}
            >
              {percent(precision)}
            </span>
          </div>
          <div style={{ fontSize: 'var(--fs-note)', color: 'var(--dim)', lineHeight: 1.6 }}>
            {already
              ? `Live. Confirmed violations notify ${policy.routed_to ?? 'nobody — set a role first'} without waiting for a reviewer.`
              : precision === null
                ? `No verdicts yet. Requires ≥ ${percent(bar.precision, 0)} precision over ${bar.reviews}+ human reviews; ${reviewed} recorded so far.`
                : eligible
                  ? `Eligible: ${percent(precision)} over ${count(reviewed)} reviews.`
                  : `Requires ≥ ${percent(bar.precision, 0)} precision over ${bar.reviews}+ human reviews. Currently ${percent(precision)} over ${count(reviewed)} reviews.`}
            {piloting && policy.pilot.days_left !== null
              ? ` Pilot ${policy.pilot.overdue ? 'overdue' : `has ${policy.pilot.days_left} day(s) left`}.`
              : ''}
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button
              className={eligible && !already && mayEdit ? 'button' : 'button button--disabled'}
              disabled={!mayEdit || !eligible || already || busy || !policy.routed_to}
              onClick={() => onPromote()}
              title={
                policy.routed_to
                  ? undefined
                  : 'A policy with no routed_to role has nobody to notify — set one above'
              }
            >
              {already
                ? 'Already auto-routed'
                : eligible
                  ? 'Promote to auto-routed'
                  : 'Promote — not yet eligible'}
            </button>
            {piloting ? (
              <button
                className="button button--ghost"
                disabled={!mayEdit || busy}
                onClick={onExtend}
                title="Give the pilot two more days of evidence before deciding."
              >
                Extend +2d
              </button>
            ) : null}
            {!eligible && !already && mayEdit && allowOverride && !overriding ? (
              <button className="button button--ghost" onClick={() => setOverriding(true)}>
                Promote anyway…
              </button>
            ) : null}
          </div>

          {overriding && !already ? (
            <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <div style={{ fontSize: 'var(--fs-note)', color: 'var(--warn)', lineHeight: 1.6 }}>
                This policy has not cleared the bar. Promoting it now means confirmed violations
                start emailing {policy.routed_to ?? 'nobody'} with no reviewer in front of them.
                The reason below is stored on the policy — it is the only answer the next person
                gets to "why is this live?".
              </div>
              <input
                className="field"
                value={override}
                placeholder="e.g. validated on the same model at the Pune plant, Jan 2026"
                onChange={(event) => setOverride(event.target.value)}
              />
              <div style={{ display: 'flex', gap: 8 }}>
                <button
                  className="button"
                  disabled={busy || override.trim().length < 8 || !policy.routed_to}
                  onClick={() => onPromote(override.trim())}
                >
                  Promote past the gate
                </button>
                <button
                  className="button button--ghost"
                  onClick={() => {
                    setOverriding(false);
                    setOverride('');
                  }}
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : null}

          {policy.promotion_note ? (
            <div className="mono" style={{ fontSize: 'var(--fs-note)', color: 'var(--warn)', lineHeight: 1.7 }}>
              {policy.promotion_note}
            </div>
          ) : null}
        </>
      )}
    </div>
  );
}
