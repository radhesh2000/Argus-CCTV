/**
 * Cameras & Zones — the fleet, enrolled and edited here, with the zone editor
 * beside it.
 *
 * Both editors are real. Vertices drag on the live picture from that camera,
 * the sanctioned headcount steps up and down, and Save writes normalized
 * coordinates back through PATCH /api/zones/{id}. The engine re-reads zones
 * every fifteen seconds, so a zone redrawn here is a zone the vision loop is
 * using before the operator has put the mouse down twice.
 *
 * Enrolment is new, and the interesting part of it is the test button: it asks
 * *the gateway* to open the URL rather than opening it from the browser. go2rtc
 * is the component that will hold the camera in production — one connection,
 * fanned out — and many cameras allow exactly one RTSP session, so a second
 * opinion from here would be a different answer to a different question. It
 * also means the probe works against a camera the browser cannot route to,
 * which on a segmented plant network is the normal case.
 *
 * Coordinates are normalized (0–1 of the frame) throughout, which is what lets
 * a zone drawn on a 1080p preview keep meaning the same thing to a detector
 * looking at 960×544.
 */

import { Fragment, useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../api/client';
import type { CameraDraft, CameraProbe } from '../api/client';
import type { Camera, LiveCamera, LiveState, Zone } from '../api/types';
import { VideoFrame } from '../components/VideoFrame';
import { ErrorBar } from '../components/marks';
import { useResource } from '../lib/hooks';
import { relativeTime } from '../lib/format';
import { useSession } from '../state/Session';

/** What the appliance puts where a camera's password is (`argus/redact.py`). */
const PASSWORD_MASK = '***';

interface Props {
  live: LiveState | null;
  onReload: () => void;
}

export function CamerasAndZones({ live, onReload }: Props) {
  const fleet = useResource(() => api.cameras(), []);
  const { can } = useSession();
  const mayEdit = can('manage');
  const [selectedCamera, setSelectedCamera] = useState<string | null>(null);
  const [selectedZone, setSelectedZone] = useState<string | null>(null);
  /** The enrolment/edit form, or null when the zone editor has the aside.
   *  `null` id means "adding" — the form is the same either way. */
  const [draft, setDraft] = useState<(CameraDraft & { id?: string }) | null>(null);

  const cameras = fleet.data?.cameras ?? [];
  // Nothing is open until a row is clicked: the editor opens in place under
  // its row, and opening the first camera unasked would push the rest of the
  // fleet below the fold.
  const camera = cameras.find((entry) => entry.id === selectedCamera) ?? null;
  const zones = camera?.zones ?? [];
  const zone = zones.find((entry) => entry.id === selectedZone) ?? zones[0] ?? null;

  const liveCamera = live?.cameras.find((entry) => entry.id === camera?.id) ?? null;
  const liveZone = liveCamera?.zones.find((entry) => entry.id === zone?.id) ?? null;

  const [addingZone, setAddingZone] = useState(false);
  const [zoneFailure, setZoneFailure] = useState<string | null>(null);

  // A row near the bottom of a long fleet would otherwise open its editor
  // below the fold; bring whatever just opened into view.
  const expanded = useRef<HTMLDivElement>(null);
  const formOpen = draft !== null;
  useEffect(() => {
    expanded.current?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }, [camera?.id, formOpen]);

  /** A new zone starts as the middle half of the frame and is then shaped in
   *  the editor it opens in. The API has always accepted a POST here; without
   *  this button a freshly enrolled camera had no way to get its first zone,
   *  and so no zone editor and no picture either. */
  const addZone = async () => {
    if (!camera) return;
    setAddingZone(true);
    setZoneFailure(null);
    try {
      const { zone: created } = await api.createZone(camera.id, {
        name: `Zone ${zones.length + 1}`,
        polygon: [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]],
      });
      setSelectedZone(created.id);
      fleet.reload();
      onReload();
    } catch (error) {
      setZoneFailure(error instanceof Error ? error.message : String(error));
    } finally {
      setAddingZone(false);
    }
  };

  const cameraForm = (current: CameraDraft & { id?: string }) => (
    <div className="cameras__form">
      <CameraForm
        draft={current}
        onChange={setDraft}
        onCancel={() => setDraft(null)}
        onSaved={(saved) => {
          setDraft(null);
          setSelectedCamera(saved.id);
          fleet.reload();
          onReload();
        }}
      />
    </div>
  );

  /** The opened row: its actions, then the picture and its zone controls side
   *  by side at the table's full width. The picture is what the operator draws
   *  on, so it gets the room the old 440px aside could not give it. */
  const openPanel = (entry: Camera) => (
    <div className="cameras__expand" ref={expanded}>
      {draft !== null && draft.id === entry.id ? (
        cameraForm(draft)
      ) : (
        <>
          {mayEdit ? (
            <div className="cameras__actions">
              <button
                className="button button--quiet"
                onClick={() =>
                  setDraft({
                    id: entry.id,
                    name: entry.name,
                    source: entry.source,
                    enabled: Boolean(entry.enabled),
                    detect_fps: entry.detect_fps,
                  })
                }
              >
                Edit camera
              </button>
              <button className="button button--quiet" onClick={addZone} disabled={addingZone}>
                {addingZone ? 'Adding…' : '+ Add zone'}
              </button>
              <span className="mono cameras__source" title={entry.source}>
                {entry.source}
              </span>
            </div>
          ) : null}
          <ErrorBar error={zoneFailure} />
          {zone ? (
            <ZoneEditor
              key={zone.id}
              camera={entry}
              zone={zone}
              zones={zones}
              liveCount={liveZone?.count ?? null}
              streamName={liveCamera?.stream ?? entry.id}
              onPickZone={setSelectedZone}
              onSaved={() => {
                fleet.reload();
                onReload();
              }}
            />
          ) : (
            // No zone to edit is no reason to hide the camera: the operator
            // still needs to see the view to decide where the first zone goes.
            <div className="zonelayout">
              <div className="zonelayout__stage">
                <div className="zonecanvas">
                  <VideoFrame stream={liveCamera?.stream ?? entry.id} />
                </div>
              </div>
              <div className="zonelayout__side">
                <div className="eyebrow">Live view · {entry.id}</div>
                <div style={{ fontSize: 'var(--fs-meta)', color: 'var(--faint)', lineHeight: 1.6 }}>
                  {`${entry.name} has no zones yet. A zone is what turns a camera into a headcount policy — and what lets a prompt policy be asked about one part of the view.`}
                  {mayEdit ? ' Use + Add zone to draw the first one.' : ''}
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );

  return (
    <div className="rise cameras">
      <div className="cameras__main">
        <div className="cameras__head">
          <div className="screen-title">Cameras &amp; Zones</div>
          {mayEdit ? (
            <button
              className="button"
              onClick={() => {
                setSelectedCamera(null);
                setZoneFailure(null);
                setDraft({ name: '', source: 'rtsp://', enabled: true, detect_fps: null });
              }}
            >
              + Add camera (RTSP / ONVIF)
            </button>
          ) : (
            <button
              className="button button--disabled"
              disabled
              title="Enrolling a camera needs the RESPOND or ADMIN role."
            >
              + Add camera (RTSP / ONVIF)
            </button>
          )}
        </div>

        <ErrorBar error={fleet.error} />

        <div className="table">
          <div className="table__head">
            <span>CAMERA</span>
            <span>NAME</span>
            <span>HEALTH</span>
            <span>FPS</span>
            <span>ZONES</span>
            <span>SOURCE</span>
          </div>
          {draft !== null && draft.id === undefined ? (
            <div className="cameras__expand" ref={expanded}>
              {cameraForm(draft)}
            </div>
          ) : null}
          {cameras.map((entry) => {
            const liveEntry = live?.cameras.find((item) => item.id === entry.id) ?? null;
            // A camera that is not watching anything says so with the row, not
            // only with a coloured word in one cell: a left rule and a tint
            // read at a glance and survive a colour-blind operator.
            const exception = cameraState(entry, liveEntry, live?.engine.running ?? false).row;
            const open = entry.id === camera?.id;
            return (
              <Fragment key={entry.id}>
                <button
                  className={`table__row ${exception} ${open ? 'table__row--on' : ''}`}
                  aria-expanded={open}
                  onClick={() => {
                    // A second click on the open row closes it.
                    setSelectedCamera(open ? null : entry.id);
                    setSelectedZone(null);
                    setDraft(null);
                    setZoneFailure(null);
                  }}
                >
                  <span className="mono" style={{ fontSize: 'var(--fs-meta)', fontWeight: 700 }}>
                    {entry.id}
                  </span>
                  <span style={{ fontSize: 'var(--fs-meta)' }}>{entry.name}</span>
                  <HealthCell
                    camera={entry}
                    live={liveEntry}
                    engineRunning={live?.engine.running ?? false}
                  />
                  <span
                    className="mono"
                    style={{
                      fontSize: 'var(--fs-note)',
                      color:
                        liveEntry?.fps == null
                          ? 'var(--faint)'
                          : liveEntry.fps < (liveEntry.target_fps ?? 5) * 0.8
                            ? 'var(--warn)'
                            : 'var(--dim)',
                    }}
                  >
                    {liveEntry?.fps != null ? liveEntry.fps.toFixed(2) : '—'}
                  </span>
                  <span className="mono" style={{ fontSize: 'var(--fs-note)', color: 'var(--dim)' }}>
                    {entry.zones?.length ?? 0}
                  </span>
                  <span
                    className="mono"
                    style={{ fontSize: 'var(--fs-note)', color: 'var(--faint)', overflow: 'hidden', textOverflow: 'ellipsis' }}
                    title={entry.source}
                  >
                    {entry.kind}
                  </span>
                </button>
                {open ? openPanel(entry) : null}
              </Fragment>
            );
          })}
          {cameras.length === 0 && !fleet.loading ? (
            <div style={{ padding: 18, fontSize: 'var(--fs-meta)', color: 'var(--faint)' }}>
              No cameras enrolled. Run <code>argus seed</code> to import the demo fleet.
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

/**
 * Enrol a camera, or edit one. Same form both ways — the only difference is
 * which verb it ends in and whether the id is already spoken for.
 *
 * Deleting is deliberately absent. A camera owns its events, its clips and its
 * zones, and `DELETE /api/cameras/{id}` cascades through all of them; a
 * mis-click here would take a month of evidence with it. Disabling stops the
 * decoder, removes the camera from the gateway config and keeps every frame it
 * ever produced, which is what "we moved that camera" actually means.
 */
function CameraForm({
  draft,
  onChange,
  onCancel,
  onSaved,
}: {
  draft: CameraDraft & { id?: string };
  onChange: (next: CameraDraft & { id?: string }) => void;
  onCancel: () => void;
  onSaved: (camera: Camera) => void;
}) {
  const creating = !draft.id;
  const [saving, setSaving] = useState(false);
  const [probing, setProbing] = useState(false);
  const [probe, setProbe] = useState<CameraProbe | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  const set = (patch: Partial<CameraDraft>) => onChange({ ...draft, ...patch });

  const usable = draft.name.trim().length > 0 && draft.source.trim().length > 4;
  // The console is never given a camera's password, only `user:***@host`. An
  // edit that leaves the placeholder alone keeps what the appliance has stored.
  const keepsPassword = !creating && draft.source.includes(`:${PASSWORD_MASK}@`);

  const save = async () => {
    setSaving(true);
    setFailure(null);
    try {
      const body: CameraDraft = {
        name: draft.name.trim(),
        source: draft.source.trim(),
        enabled: draft.enabled ?? true,
        detect_fps: draft.detect_fps ?? null,
      };
      const { camera } = draft.id
        ? await api.updateCamera(draft.id, body)
        : await api.createCamera(body);
      onSaved(camera);
    } catch (error) {
      setFailure(error instanceof Error ? error.message : String(error));
    } finally {
      setSaving(false);
    }
  };

  const test = async () => {
    setProbing(true);
    setProbe(null);
    setFailure(null);
    try {
      setProbe(
        await api.probeCamera({
          source: draft.source.trim(),
          ...(draft.id ? { password_from: draft.id } : {}),
        }),
      );
    } catch (error) {
      setFailure(error instanceof Error ? error.message : String(error));
    } finally {
      setProbing(false);
    }
  };

  return (
    <>
      <div className="eyebrow">{creating ? 'Enrol a camera' : `Editing · ${draft.id}`}</div>
      <ErrorBar error={failure} />

      <div className="policy-edit">
        <label className="policy-edit__field">
          <span className="eyebrow">Name</span>
          <input
            className="field"
            value={draft.name}
            placeholder="Packing line — west"
            onChange={(event) => set({ name: event.target.value })}
          />
          {creating ? (
            <span className="policy-edit__hint">
              The id is derived from this and never changes afterwards, because events, evidence
              and policy scopes all point at it.
            </span>
          ) : null}
        </label>

        <label className="policy-edit__field">
          <span className="eyebrow">Source</span>
          <input
            className="field mono"
            value={draft.source}
            placeholder="rtsp://user:pass@10.0.0.9:554/Streaming/Channels/102"
            onChange={(event) => set({ source: event.target.value })}
          />
          <span className="policy-edit__hint">
            An <span className="mono">rtsp://</span> / <span className="mono">http://</span> URL, or
            an absolute path to a recording. Point it at the camera's <em>sub-stream</em> where it
            has one: detection runs at 5 fps and 960×544, and a 4K main stream buys nothing but
            decode. Nothing else is accepted — the gateway runs ffmpeg, and a source is an argument
            to it.
          </span>
          {keepsPassword ? (
            <span className="policy-edit__hint">
              <span className="mono">{PASSWORD_MASK}</span> is the stored password, which never
              leaves the appliance. Leave it in place to keep it, or type over it to change it.
              Change the scheme, user, host or port and the password has to be typed again.
            </span>
          ) : null}
        </label>

        <div className="policy-edit__row">
          <label className="policy-edit__field">
            <span className="eyebrow">Detect fps</span>
            <input
              className="field"
              type="number"
              min={1}
              placeholder="appliance default"
              value={draft.detect_fps ?? ''}
              onChange={(event) =>
                set({ detect_fps: event.target.value === '' ? null : Number(event.target.value) })
              }
            />
          </label>
          <label className="policy-edit__toggle" style={{ alignSelf: 'end', paddingBottom: 8 }}>
            <input
              type="checkbox"
              checked={draft.enabled ?? true}
              onChange={(event) => set({ enabled: event.target.checked })}
            />
            <span>Enabled</span>
          </label>
        </div>

        <div>
          <button
            className="button button--soft"
            disabled={probing || draft.source.trim().length < 5}
            onClick={test}
          >
            {probing ? 'Asking the gateway…' : 'Test connection'}
          </button>
          {probe ? (
            <div
              className="probe"
              style={{
                borderColor: probe.ok ? 'var(--good)' : probe.gateway ? 'var(--crit)' : 'var(--border)',
              }}
            >
              <div
                className="mono probe__verdict"
                style={{
                  color: probe.ok ? 'var(--good)' : probe.gateway ? 'var(--crit)' : 'var(--warn)',
                }}
              >
                {probe.ok ? 'OPENED' : probe.gateway ? 'NO STREAM' : 'GATEWAY DOWN'}
                <span style={{ color: 'var(--faint)' }}> · {probe.elapsed_sec.toFixed(2)}s</span>
              </div>
              <div className="probe__detail">{probe.detail}</div>
            </div>
          ) : null}
        </div>

        <div style={{ display: 'flex', gap: 8 }}>
          <button className="button" disabled={!usable || saving} onClick={save}>
            {saving ? 'Saving…' : creating ? 'Enrol camera' : 'Save changes'}
          </button>
          <button className="button button--ghost" disabled={saving} onClick={onCancel}>
            Cancel
          </button>
        </div>
      </div>
    </>
  );
}

/**
 * The health of one camera, including the state that has no status code.
 *
 * `awaiting restart` is the one worth explaining. The engine reads the camera
 * table once, at startup — zones and policies are re-read every 15 s, a camera
 * is not, because it means a decoder process and a regenerated gateway config.
 * So a camera enrolled from the form on this very screen is stored, is
 * enabled, has just answered a probe with `OPENED`, and is not being decoded by
 * anything. Its row said `offline`, which is also what a camera with a cut
 * cable says, and the operator's next move for those two facts is not the same.
 */
/**
 * One reading of a camera's state, used by both the health cell and the row it
 * sits in.
 *
 * It lives in one function because the two were derived separately and
 * disagreed: the cell read the camera table and the row read the live snapshot,
 * so on an appliance whose engine is stopped every row was flagged as an
 * exception while every cell said "live". A row treatment that contradicts the
 * word next to it is worse than no row treatment.
 *
 * `row` is the class the exception states earn — a left rule and a tint, so the
 * state reads at a glance and at peripheral vision, not only in one cell's hue.
 */
function cameraState(camera: Camera, live: LiveCamera | null, engineRunning: boolean) {
  // Never delivered a frame *and* has no decoder attached, while an engine is
  // up: that is a camera this engine was never told about. A camera it is
  // trying and failing reads `connecting` or `error` and keeps its own message;
  // one that died has a `last_seen_at` and is honestly `offline`.
  const awaitingRestart =
    engineRunning &&
    Boolean(camera.enabled) &&
    camera.status === 'offline' &&
    camera.last_seen_at === null &&
    (live?.decode_mode ?? null) === null;
  if (awaitingRestart) {
    return {
      glyph: '◆',
      color: 'var(--warn)',
      text: 'awaiting restart',
      row: 'row--warn',
      title:
        'Enrolled, but the running engine has never been told about it — it reads the ' +
        'camera table at startup. Restart the engine to begin decoding this camera.',
    };
  }
  const detail =
    camera.status_detail ??
    (camera.last_seen_at
      ? `last frame ${relativeTime(new Date(camera.last_seen_at).getTime() / 1000)}`
      : 'never seen');
  if (camera.status === 'live') {
    // No fps while the camera is otherwise healthy means the engine is not
    // sampling it. Worth saying, not worth flagging the row for: nothing is
    // wrong with the camera.
    const text = (live?.fps ?? null) === null ? 'live (no engine)' : 'live';
    return { glyph: '●', color: 'var(--good-dot)', text, row: '', title: detail };
  }
  if (camera.status === 'connecting') {
    return { glyph: '◆', color: 'var(--warn)', text: 'connecting', row: 'row--warn', title: detail };
  }
  if (camera.status === 'error') {
    return { glyph: '◆', color: 'var(--crit)', text: 'error', row: 'row--crit', title: detail };
  }
  // A disabled camera is not a fault: nobody asked the engine to watch it.
  return {
    glyph: '○',
    color: 'var(--dim)',
    text: 'offline',
    row: camera.enabled ? 'row--warn' : '',
    title: detail,
  };
}

function HealthCell({
  camera,
  live,
  engineRunning,
}: {
  camera: Camera;
  live: LiveCamera | null;
  engineRunning: boolean;
}) {
  const look = cameraState(camera, live, engineRunning);
  return (
    <span
      style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 'var(--fs-note)', color: look.color }}
      title={look.title}
    >
      <span style={{ fontSize: 'var(--fs-label)' }}>{look.glyph}</span>
      {look.text}
    </span>
  );
}

function ZoneEditor({
  camera,
  zone,
  zones,
  liveCount,
  streamName,
  onPickZone,
  onSaved,
}: {
  camera: Camera;
  zone: Zone;
  zones: Zone[];
  liveCount: number | null;
  streamName: string;
  onPickZone: (zoneId: string) => void;
  onSaved: () => void;
}) {
  const [polygon, setPolygon] = useState<[number, number][]>(zone.polygon);
  const [limit, setLimit] = useState<number | null>(zone.sanctioned_limit);
  const [name, setName] = useState(zone.name);
  const [dragging, setDragging] = useState<number | null>(null);
  const [saving, setSaving] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);
  const surface = useRef<HTMLDivElement>(null);

  const dirty =
    name !== zone.name ||
    limit !== zone.sanctioned_limit ||
    JSON.stringify(polygon) !== JSON.stringify(zone.polygon);

  const pointFromEvent = useCallback((event: { clientX: number; clientY: number }): [number, number] => {
    const box = surface.current?.getBoundingClientRect();
    if (!box) return [0, 0];
    return [
      Math.min(1, Math.max(0, (event.clientX - box.left) / box.width)),
      Math.min(1, Math.max(0, (event.clientY - box.top) / box.height)),
    ];
  }, []);

  useEffect(() => {
    if (dragging === null) return;
    const move = (event: MouseEvent) => {
      const point = pointFromEvent(event);
      setPolygon((current) => current.map((entry, index) => (index === dragging ? point : entry)));
    };
    const up = () => setDragging(null);
    window.addEventListener('mousemove', move);
    window.addEventListener('mouseup', up);
    return () => {
      window.removeEventListener('mousemove', move);
      window.removeEventListener('mouseup', up);
    };
  }, [dragging, pointFromEvent]);

  const save = async () => {
    setSaving(true);
    setFailure(null);
    try {
      await api.updateZone(zone.id, { name, polygon, sanctioned_limit: limit });
      onSaved();
    } catch (error) {
      setFailure(error instanceof Error ? error.message : String(error));
    } finally {
      setSaving(false);
    }
  };

  /** Insert a vertex at the midpoint of the nearest edge — the "dbl-click to
   *  add" the design promises, placed where the operator actually clicked. */
  const addVertex = (event: React.MouseEvent) => {
    const point = pointFromEvent(event);
    let bestIndex = 0;
    let bestDistance = Infinity;
    for (let index = 0; index < polygon.length; index += 1) {
      const a = polygon[index]!;
      const b = polygon[(index + 1) % polygon.length]!;
      const midpoint: [number, number] = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
      const distance = Math.hypot(midpoint[0] - point[0], midpoint[1] - point[1]);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestIndex = index + 1;
      }
    }
    setPolygon((current) => [...current.slice(0, bestIndex), point, ...current.slice(bestIndex)]);
  };

  return (
    <div className="zonelayout">
      <div className="zonelayout__stage">
        <div className="zonecanvas" onDoubleClick={addVertex}>
          <VideoFrame stream={streamName} boxRef={surface}>
            <svg
              viewBox="0 0 100 100"
              preserveAspectRatio="none"
              style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}
            >
              <polygon
                points={polygon.map(([x, y]) => `${x * 100},${y * 100}`).join(' ')}
                style={{ fill: 'rgba(179,52,31,.14)', stroke: 'var(--accent)', strokeWidth: 1.6 }}
                vectorEffect="non-scaling-stroke"
              />
            </svg>
            {polygon.map(([x, y], index) => (
              <div
                key={index}
                className="zonecanvas__vertex"
                style={{ left: `${x * 100}%`, top: `${y * 100}%` }}
                onMouseDown={(event) => {
                  event.preventDefault();
                  setDragging(index);
                }}
                onContextMenu={(event) => {
                  // Right-click removes, while a polygon remains a polygon.
                  event.preventDefault();
                  if (polygon.length > 3) {
                    setPolygon((current) => current.filter((_, position) => position !== index));
                  }
                }}
                title="Drag to move · right-click to remove"
              />
            ))}
          </VideoFrame>
          <div className="zonecanvas__hint mono">DRAG VERTICES · DBL-CLICK TO ADD · RIGHT-CLICK TO REMOVE</div>
        </div>

      </div>
      <div className="zonelayout__side">
        <div className="eyebrow">Zone editor · {camera.id}</div>
        {zones.length > 1 ? (
          <select
            className="focus__zonepick"
            value={zone.id}
            onChange={(event) => onPickZone(event.target.value)}
          >
            {zones.map((entry) => (
              <option key={entry.id} value={entry.id}>
                {entry.name}
              </option>
            ))}
          </select>
        ) : null}
        <ErrorBar error={failure} />

        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div>
            <div className="eyebrow" style={{ marginBottom: 6 }}>
              Zone name
            </div>
            <input className="field" value={name} onChange={(event) => setName(event.target.value)} />
          </div>

          <div style={{ display: 'flex', gap: 12 }}>
            <div style={{ flex: 1 }}>
              <div className="eyebrow" style={{ marginBottom: 6 }}>
                Sanctioned headcount
              </div>
              <div className="stepper">
                <button
                  onClick={() => setLimit((value) => (value === null ? 0 : Math.max(0, value - 1)))}
                >
                  −
                </button>
                <span className="mono">{limit ?? '—'}</span>
                <button onClick={() => setLimit((value) => (value === null ? 1 : value + 1))}>+</button>
              </div>
            </div>
            <div style={{ flex: 1 }}>
              <div className="eyebrow" style={{ marginBottom: 6 }}>
                Present now
              </div>
              <div className="field field--readonly mono">{liveCount === null ? '—' : liveCount}</div>
            </div>
          </div>

          <div style={{ fontSize: 'var(--fs-note)', color: 'var(--faint)', lineHeight: 1.6 }}>
            {limit === null
              ? 'No limit set: this zone is counted but never raises an alert.'
              : `Headcount policies raise an event when more than ${limit} ${limit === 1 ? 'person is' : 'people are'} inside this outline.`}
          </div>

          <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
            <button className="button" disabled={!dirty || saving} onClick={save}>
              {saving ? 'Saving…' : 'Save zone'}
            </button>
            <button
              className="button button--ghost"
              disabled={!dirty}
              onClick={() => {
                setPolygon(zone.polygon);
                setLimit(zone.sanctioned_limit);
                setName(zone.name);
              }}
            >
              Discard
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
