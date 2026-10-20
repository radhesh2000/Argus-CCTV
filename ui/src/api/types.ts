/**
 * The shapes the Argus API actually returns.
 *
 * Hand-written rather than generated from the OpenAPI document on purpose: the
 * API answers with database rows as they are (see argus/api/schemas.py — only
 * *requests* are modelled server-side), so there is no response schema to
 * generate from. These declarations are the contract as read from
 * argus/db.py's tables and the routers, and they are the one place to look
 * when the backend changes.
 *
 * SQLite has no boolean type: `enabled` and friends arrive as 0 or 1.
 */

export type Severity = 'low' | 'medium' | 'high';
export type Stage = 'shadow' | 'reviewed' | 'auto_routed';
export type Lifecycle = 'draft' | 'pilot' | 'production' | 'archived';
export type Verdict = 'confirmed' | 'false_positive';
export type CameraStatus = 'offline' | 'connecting' | 'live' | 'error';

/** The four roles the Settings screen has named since the design. Their
 *  meanings live in argus/auth.py; the appliance ships the descriptions in
 *  `/api/config` so the console never has a second, drifting copy. */
export type Role = 'ADMIN' | 'RESPOND' | 'REVIEW' | 'SYSTEM';
export type Permission = 'view' | 'triage' | 'manage' | 'admin' | 'system';

/** Who this browser is signed in as. */
export interface Session {
  id: number;
  username: string;
  display_name: string;
  role: Role;
  permissions: Permission[];
  /** ISO-8601. The absolute deadline, not a countdown — the server decides. */
  expires_at: string;
}

export interface User {
  id: number;
  username: string;
  display_name: string;
  role: Role;
  /** SQLite has no boolean type. */
  disabled: 0 | 1;
  last_login_at: string | null;
  created_at: string;
  updated_at: string;
  permissions: Permission[];
}

export interface UserRoster {
  users: User[];
  roles: Role[];
  role_descriptions: Record<Role, string>;
  /** So the console can grey out the control that would lock the plant out of
   *  its own appliance rather than offering it and then showing a 409. */
  admin_count: number;
  self: string;
}

export interface Shift {
  id: string;
  start_hour: number;
  end_hour: number;
}

export interface AppConfig {
  version: string;
  source: string | null;
  site: { name: string; shifts: Shift[] };
  features: {
    search: boolean;
    /** This appliance evaluates prompt policies at all. Whether the model is
     *  answering right now is a health question — see `Health.vlm`. */
    vlm: boolean;
    vlm_verification: boolean;
    auth: boolean;
    retraining_exports: boolean;
    channels: { email: boolean; whatsapp: boolean; slack: boolean };
  };
  detection: {
    fps: number;
    width: number;
    height: number;
    model: string;
    precision: string;
  };
  evidence: { clip_pad_sec: number; enabled: boolean };
  /** The promotion gate, as the appliance defines it. Read from here rather
   *  than hard-coded, because a bar that lives in the console is a bar the API
   *  does not enforce — which is exactly what it was. */
  rollout: { min_precision: number; min_reviews: number; allow_override: boolean };
  index: {
    model: string;
    sample_interval_sec: number;
    min_score: number;
    default_limit: number;
  };
  retention: {
    enabled: boolean;
    clips_days: number;
    snapshots_days: number;
    events_days: number;
    occupancy_days: number;
    index_days: number;
    max_disk_percent: number;
    /** Per-stage evidence windows, already resolved through the fallback chain
     *  by the appliance — see argus/config.py:EvidenceRetentionConfig. */
    evidence_days: Record<
      'shadow' | 'reviewed' | 'auto_routed' | 'confirmed' | 'false_positive',
      number
    >;
    /** False when the config sets no per-stage block and the flat windows rule. */
    evidence_tiered: boolean;
    /** null = no hard ceiling configured for that media kind. */
    max_clips_gb: number | null;
    max_snapshots_gb: number | null;
  };
  auth: {
    roles: Role[];
    role_descriptions: Record<Role, string>;
    session_ttl_hours: number;
    session_idle_minutes: number;
    min_password_length: number;
  };
  gateway: {
    url: string;
    /** Loopback since Phase 5 — reported so Settings can state the posture, not
     *  so the browser can dial it. Signalling goes through `relay`. */
    api_host: string;
    api_port: number;
    /** The exception: ICE connects the browser here directly. docs/SECURITY.md. */
    webrtc_port: number;
    relay: string;
  };
  stages: Stage[];
  severities: Severity[];
  notify_roles: string[];
  notify_recipients: Record<string, string[]>;
  smtp: { host: string; port: number; from_address: string };
}

export interface Zone {
  id: string;
  camera_id: string;
  name: string;
  polygon: [number, number][];
  sanctioned_limit: number | null;
  color: string;
  created_at: string;
  updated_at: string;
}

export interface Camera {
  id: string;
  name: string;
  /** Never the real thing: the appliance masks the password before a camera
   *  row leaves it — `rtsp://user:***@host/...` — for every role. Sent back
   *  unchanged on an edit, the `***` keeps the stored password. */
  source: string;
  kind: 'rtsp' | 'http' | 'file';
  enabled: number;
  status: CameraStatus;
  status_detail: string | null;
  detect_fps: number | null;
  last_seen_at: string | null;
  created_at: string;
  updated_at: string;
  zones?: Zone[];
}

export interface PolicyScope {
  camera_id: string;
  zone_id: string | null;
}

export interface Policy {
  id: string;
  kind: 'detector' | 'vlm';
  detector_key: string;
  label: string;
  enabled: number;
  stage: Stage;
  lifecycle: Lifecycle;
  severity: Severity;
  routed_to: string | null;
  prompt: string | null;
  interval_sec: number | null;
  cooldown_sec: number;
  threshold: number | null;
  pilot_started_at: string | null;
  pilot_duration_days: number;
  promoted_at: string | null;
  /** Set only when somebody promoted past the gate: the timestamp, the numbers
   *  as they were, and the reason they gave. */
  promotion_note: string | null;
  created_at: string;
  updated_at: string;
  /** Where this policy applies. `[]` means every camera — which is what every
   *  policy meant before schema v5, so an empty list is not "nowhere". A row
   *  with `zone_id` set narrows a detector policy to that zone and shows a VLM
   *  policy the crop of it. */
  scope: PolicyScope[];
  pilot: {
    days_elapsed: number | null;
    days_total: number;
    days_left: number | null;
    overdue: boolean;
  };
}

/** Normalized `[x1, y1, x2, y2, track_id, confidence]`. */
export type Box = [number, number, number, number, number, number];

export interface EventMeta {
  count?: number;
  limit?: number | null;
  excess?: number;
  zone_name?: string;
  camera_name?: string;
  track_ids?: number[];
  boxes?: Box[];
  inside?: boolean[];
  frame?: { width: number; height: number };
  detector?: string;
  // --- written by the VLM policy engine (argus/vlm/scheduler.py::_emit) ---
  /** The prompt policy's human label. */
  policy_label?: string;
  /** The model's own sentence — the whole reason the event exists. */
  reason?: string;
  /** The exact prompt that was asked of the frame. */
  prompt?: string;
  /** Which model answered, e.g. "Qwen/Qwen3-VL-8B-Instruct". */
  model?: string;
  latency_sec?: number;
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  /** Grounding box, normalized like `boxes`, when the model located its answer. */
  box_2d?: [number, number, number, number];
  coords?: string;
  /** What the model said when an ongoing condition *started*. `description`
   *  carries the newest sentence, so without this the opening claim — the one
   *  that decided the alert was worth raising — would be overwritten by the
   *  fortieth restatement of it. Written on the first extension only. */
  opened?: { ts: number; reason: string | null; confidence: number | null };
}

export interface ArgusEvent {
  id: number;
  camera_id: string;
  zone_id: string | null;
  policy_id: string | null;
  detector_name: string;
  /** `vlm` = raised by a prompt policy; `detector` = by the fast detector. */
  detector_type: 'detector' | 'vlm';
  severity: Severity;
  confidence: number;
  stage: Stage;
  routed_to: string | null;
  description: string;
  ts: number;
  snapshot_path: string | null;
  clip_path: string | null;
  clip_start: number | null;
  tracks_path: string | null;
  meta: EventMeta | null;
  created_at: string;
  // --- the VLM verification pass (schema migration 2). A vetoed event keeps
  // its row and its evidence; these columns are the audit trail. ---
  suppressed?: number;
  suppressed_reason?: string | null;
  verified_by?: string | null;
  verified_at?: string | null;
  verify_confidence?: number | null;
  /** Present on list responses (LEFT JOIN reviews). */
  review_verdict?: Verdict | null;
  reviewer?: string | null;
  reviewed_at?: string | null;
  // --- ongoing conditions (schema migration 7) ------------------------------
  // A prompt policy asks about a *state*, and a state persists. When the model
  // answers "yes" again about the same thing in the same place, the appliance
  // extends this row instead of writing another one — so a VLM event is not a
  // moment, it is a span with a count of how often it was seen.
  //
  // Set on detector-origin events too, since `detection.continuity`: a zone
  // over its sanctioned limit is a state for exactly the same reason, and the
  // debounce that was supposed to hold it steady had written 112 151 rows about
  // it. Such a row's `condition_key` is `det:policy:camera:zone`, and its
  // `description` is "Packing floor: 12 people · limit 8 · peak 14 · 6 min".
  /** `policy:camera`, `policy:camera:zone`, or `det:policy:camera:zone`. */
  condition_key?: string | null;
  /** `open` = still true as far as the appliance knows. */
  condition_state?: 'open' | 'closed' | null;
  /** Triggered verdicts folded into this row, including the one that opened it. */
  observations?: number | null;
  /** Epoch of the newest observation. `ts` is the first — the card's "since". */
  last_seen_ts?: number | null;
  closed_at?: number | null;
  /** `cleared` = the model answered no; `stale` = nobody looked for long
   *  enough; `moved` = a triggered verdict landed somewhere else and started
   *  its own condition. The three are not the same news. */
  close_reason?: 'cleared' | 'stale' | 'moved' | null;
  /** Epoch of the last write to the condition. The SSE stream's cursor. */
  condition_updated_at?: number | null;
}

export interface EventPage {
  events: ArgusEvent[];
  total: number;
  limit: number;
  offset: number;
  detectors: string[];
}

export interface Review {
  id: number;
  event_id: number;
  verdict: Verdict;
  reviewer: string | null;
  note: string | null;
  created_at: string;
  updated_at: string;
}

export type BoxVerdict = 'true_positive' | 'false_positive';

/** One reviewer's judgement on a single detection within an event's
 *  `meta.boxes` — distinct from `Review`, which judges the whole alert. */
export interface BoxCorrection {
  id: number;
  event_id: number;
  track_id: number;
  verdict: BoxVerdict;
  reviewer: string | null;
  note: string | null;
  created_at: string;
  updated_at: string;
}

export interface DetectorAccuracy {
  detector_name: string;
  detector_type: 'detector' | 'vlm';
  total: number;
  confirmed: number;
  false_positive: number;
  pending: number;
  reviewed: number;
  precision: number | null;
  fp_rate: number | null;
  /** Held back by the VLM verification pass (kept, not deleted). */
  suppressed: number;
  /** Events the verifier actually looked at — the denominator of the rate. */
  verified: number;
  suppression_rate: number | null;
}

export interface AccuracyReport {
  detectors: DetectorAccuracy[];
  trend: { day: string; confirmed: number; false_positive: number }[];
  total_reviewed: number;
  total_pending: number;
}

export interface LiveZone {
  id: string;
  name: string;
  polygon: [number, number][];
  color: string;
  sanctioned_limit: number | null;
  /** null when nobody is watching — never a fabricated zero. */
  count: number | null;
  over_limit: boolean;
}

export interface LiveCamera {
  id: string;
  name: string;
  enabled: boolean;
  status: CameraStatus;
  status_detail: string | null;
  last_seen_at: string | null;
  stream: string;
  target_fps: number;
  fps: number | null;
  lag_ms: number | null;
  decode_mode: string | null;
  zones: LiveZone[];
  tracks: Box[] | null;
  live_ts: number | null;
  people: number | null;
}

/**
 * The appliance pause switch — the console's honest read on whether the GPU
 * plane is running, paused on purpose, or warming up.
 *
 * `status` is the headline: `paused` (the flag is off, engine + VLM down by
 * intent), `starting` (resuming — the engine is not yet up, or it is and the
 * VLM is still within its cold-start grace: "models warming up"), or `running`.
 * `vlm` is the model's own sub-status, for the panels that show it in full.
 * A stopped engine while `paused` is a deliberate switch, not a failure, and the
 * console renders it calm rather than red — that is what this block is for.
 */
export type ProcessingStatus = 'paused' | 'starting' | 'running';
export type ProcessingVlm = 'disabled' | 'starting' | 'ready' | 'down' | 'paused';

export interface Processing {
  processing_enabled: boolean;
  status: ProcessingStatus;
  vlm: ProcessingVlm;
  /** Whether this box has a VLM worth waiting for — enabled in config and
   *  actually installed. False means the model is not part of the resume. */
  vlm_required: boolean;
  updated_at: string | null;
  updated_by: string | null;
}

/** The full control-plane view (`GET`/`POST /api/system/processing`): the light
 *  block above, plus each toggled unit's actual systemd state. */
export interface ProcessingReport extends Processing {
  /** False on a box with no user systemd — the flag still persists, but nothing
   *  here can start or stop a unit. */
  systemctl: boolean;
  services: {
    engine: ProcessingService;
    vlm: ProcessingService & { available: boolean };
  };
  /** Present on the `POST` response: whether the units were actually actuated,
   *  the command run, and any note (e.g. systemctl unavailable). */
  actuated?: boolean;
  command?: string | null;
  detail?: string | null;
}

export interface ProcessingService {
  unit: string;
  active_state: string | null;
  sub_state: string | null;
  /** The word the console renders: running | paused | starting | failed |
   *  stopped | disabled. Derived from the desired flag *and* the live unit
   *  state, because an ExecCondition-skipped unit is indistinguishable from a
   *  clean stop by systemd alone. */
  state: string;
}

export interface LiveState {
  time: number;
  engine: { running: boolean; reason: string | null; age_sec: number | null };
  processing: Processing;
  gateway: { enabled: boolean; running: boolean; url: string; last_error: string | null } | null;
  shift: Shift & { local_time: string };
  cameras: LiveCamera[];
}

export interface CameraHealth {
  id: string;
  name: string;
  enabled: boolean;
  status: CameraStatus;
  status_detail: string | null;
  last_seen_at: string | null;
  target_fps: number;
  fps: number | null;
  fps_avg: number | null;
  frames: number | null;
  processed: number | null;
  lag_ms: number | null;
  events: number | null;
  decode_mode: string | null;
  restarts: number | null;
  last_error: string | null;
  ring: { mode: string; segments: number } | null;
}

/** One prompt policy against one camera, as the scheduler runs it. */
export interface VlmSchedule {
  policy_id: string;
  camera_id: string;
  /** Set when the policy is scoped to a zone: one schedule per zone, with its
   *  own interval, cooldown and counters. */
  zone_id: string | null;
  zone_name: string | null;
  /** Whether the last evaluation was actually shown the crop. A zone covering
   *  most of the frame is scoped but asked wide — there is no sub-picture. */
  cropped: boolean | null;
  /** `policy:camera` or `policy:camera:zone` — unique, and the React key. */
  key: string;
  label: string;
  interval_sec: number;
  cooldown_sec: number;
  stage: Stage;
  evaluations: number;
  /** Conditions *opened* — one per incident, since schema migration 7. */
  events: number;
  /** Triggered verdicts folded into a condition already on file. */
  observations: number;
  /** Conditions this schedule has ended. */
  closed: number;
  last_close_reason: string | null;
  failures: number;
  skipped_no_frame: number;
  /** Samples dropped because the previous one had not come back. A rising
   *  number means the model is slower than the policy's interval. */
  skipped_in_flight: number;
  last_verdict: {
    triggered: boolean;
    confidence: number;
    reason: string | null;
    box_2d: [number, number, number, number] | null;
    latency_sec: number;
    prompt_tokens: number | null;
    completion_tokens: number | null;
  } | null;
  last_error: string | null;
  degraded: boolean;
}

export interface VlmInference {
  requests: number;
  failures: number;
  last_error: string | null;
  last_inference_age_sec: number | null;
  last_latency_sec: number;
  avg_latency_sec: number;
  prompt_tokens: number;
  completion_tokens: number;
  in_flight: number;
  queue_depth: number;
}

/**
 * The VLM policy engine's condition.
 *
 * `source` says who answered: `engine` means `argus run` published these — the
 * only process that can count a queue — and `probe` means the API asked the
 * endpoint directly because the engine is not running. On the probe path every
 * counter is `null`, and null must be rendered as "unknown", never as 0.
 */
export interface VlmHealth {
  source: 'engine' | 'probe';
  enabled: boolean;
  available: boolean;
  endpoint: string;
  /** What is actually being served (probe/engine), null if nothing answered. */
  model: string | null;
  /** What argus.yaml asks for — differs from `model` only if pointed elsewhere. */
  configured_model: string;
  reason: string | null;
  degraded_for_sec: number | null;
  policies: number | null;
  schedules: VlmSchedule[] | null;
  evaluations: number | null;
  events: number | null;
  /** Ongoing conditions (schema migration 7), as the scheduler is running them. */
  continuity: {
    enabled: boolean;
    min_iou: number;
    stale_after_sec: number;
    observations: number | null;
    closed: number | null;
  } | null;
  verification: {
    enabled: boolean;
    stages: string[];
    checked: number | null;
    suppressed: number | null;
    failures: number | null;
  };
  inference: VlmInference | null;
  /** Read from disk, not the network: what is installed on this box. */
  install: {
    venv: string;
    venv_ready: boolean;
    model_path: string;
    model_ready: boolean;
    pid: number | null;
    log: string;
  };
}

export interface Health {
  version: string;
  status: string;
  time: number;
  /** The pause switch, alongside the state it governs. A `paused` engine here is
   *  a deliberate switch, so Model Health reads it calm rather than as failure. */
  processing: Processing;
  engine: {
    running: boolean;
    reason: string | null;
    age_sec: number | null;
    started_at: number | null;
    uptime_sec: number | null;
    detector: Record<string, unknown> | null;
    gateway: { enabled: boolean; running: boolean; url: string; last_error: string | null } | null;
    retention: Record<string, unknown> | null;
  };
  cameras: CameraHealth[];
  vlm: VlmHealth;
  gpu: {
    available: boolean;
    reason?: string;
    memory_source?: string;
    memory_note?: string;
    gpus: {
      index: number;
      name: string;
      utilization_percent: number | null;
      memory_used_mb: number | null;
      memory_total_mb: number | null;
      memory_percent: number | null;
      temperature_c: number | null;
      power_w: number | null;
    }[];
  };
  memory: { total_mb?: number; used_mb?: number; available_mb?: number; percent?: number };
  disk: {
    data_dir: string;
    total_gb: number;
    used_gb: number;
    free_gb: number;
    percent: number;
    max_percent: number;
    over_threshold: boolean;
    argus_mb: number;
    by_directory_mb: Record<string, number>;
    /** Evidence held against the budget it is allowed. `budget_gb` and
     *  `percent` are null where the site configured no ceiling — a number
     *  there would be an invention, not a measurement.
     *
     *  `at_budget` is true from the moment the ceiling binds; `over_budget`
     *  carries a tolerance and means the sweep has NOT kept up. A busy site
     *  lives permanently at 100%, so only the second one is worth amber. */
    evidence: {
      kinds: Record<
        'clips' | 'snapshots',
        {
          used_gb: number;
          budget_gb: number | null;
          percent: number | null;
          over_budget: boolean;
          at_budget: boolean;
        }
      >;
      used_gb: number;
      budget_gb: number | null;
      over_budget: boolean;
      at_budget: boolean;
      tiered: boolean;
    };
  };
  database: { path: string; schema_version: number } & Record<string, number | string>;
}

export interface AnalyticsTotals {
  /** Incidents — one per row, however long it went on. */
  events: number;
  /** Sampled verdicts behind those incidents. A VLM condition that was true all
   *  night is 1 event and ~40 observations; before schema migration 7 it was 40
   *  of both, which is why the trend used to be a picture of the sampling
   *  interval rather than of the plant. */
  observations: number;
  /** Conditions still open at the moment this summary was generated. */
  ongoing: number;
  confirmed: number;
  false_positive: number;
  pending: number;
  open_high: number;
  reviewed: number;
  precision: number | null;
  median_response_sec: number | null;
}

export interface AnalyticsSummary {
  generated_at: number;
  window: {
    days: number;
    since_ts: number;
    until_ts: number;
    since_day: string;
    until_day: string;
  };
  shifts: Shift[];
  totals: AnalyticsTotals;
  previous: AnalyticsTotals;
  by_day: {
    day: string;
    raw: number;
    observations: number;
    confirmed: number;
    false_positive: number;
  }[];
  by_type: {
    key: string;
    label: string;
    detector_type: 'detector' | 'vlm';
    raw: number;
    observations: number;
    confirmed: number;
    false_positive: number;
    /** Real severity (the same field triage sorts by), not a per-policy
     *  invented scale. Only the severities actually seen appear as keys. */
    by_severity: Partial<Record<Severity, number>>;
    /** Same shift ids as `shifts` above, folded from local hour the same way
     *  `by_zone_shift` is — every configured shift present, zero-filled. */
    by_shift: Record<string, { raw: number; observations: number; confirmed: number }>;
    /** Dense daily series across the whole window, this policy only — the
     *  per-card trend line, zero-filled the same way the screen's own
     *  `by_day` is. */
    by_day: { day: string; raw: number; confirmed: number }[];
  }[];
  by_zone_shift: {
    zone_id: string | null;
    zone_name: string;
    camera_id: string;
    raw: Record<string, number>;
    observations: Record<string, number>;
    confirmed: Record<string, number>;
    total: number;
  }[];
}

// --------------------------------------------------------------------------
// semantic search (Phase 3)
// --------------------------------------------------------------------------

/** What the index knows about, and how far back. Backs the honest version of
 *  the design's "SEARCHED 214 H OF FOOTAGE" line. */
export interface IndexCoverage {
  frames: number;
  events: number;
  samples: number;
  cameras: number;
  span_hours: number;
  oldest_ts: number | null;
  newest_ts: number | null;
  model: string;
}

/** The alert a result belongs to, when it is one. A sampled frame has none —
 *  it is an ordinary minute that nothing fired on, which is exactly what
 *  search is for. */
export interface SearchResultEvent {
  id: number;
  description: string;
  severity: Severity;
  detector_type: 'detector' | 'vlm';
  detector_name: string;
  stage: Stage;
  confidence: number;
  clip_path: string | null;
  clip_start: number | null;
  suppressed?: number;
  review_verdict?: Verdict | null;
}

export interface SearchResult {
  id: number;
  camera_id: string;
  camera_name: string;
  ts: number;
  kind: 'event' | 'sample';
  /** Cosine similarity between the query and the frame, reported raw. On real
   *  footage a good match is ~0.10-0.17 and noise sits below 0.05 — the bar is
   *  drawn against a fixed ceiling rather than rescaled per query, so the same
   *  score means the same thing every day. */
  score: number;
  thumb_url: string;
  width: number | null;
  height: number | null;
  event: SearchResultEvent | null;
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
  count: number;
  took_ms: number;
  embed_ms: number;
  min_score: number;
  filters: {
    camera_id: string | null;
    since_ts: number | null;
    until_ts: number | null;
    kind: string | null;
  };
  index: IndexCoverage;
}

export interface SearchStatus {
  enabled: boolean;
  model: string;
  model_path: string;
  model_ready: boolean;
  query_device: string;
  query_model_loaded: boolean;
  store_path: string;
  store_exists: boolean;
  index: (IndexCoverage & { size_mb: number }) | null;
  error: string | null;
}

export interface SavedSearch {
  id: number;
  label: string;
  query: string;
  camera_id: string | null;
  window_hours: number | null;
  /** Set once this search has been promoted to a draft VLM policy. */
  policy_id: string | null;
  policy_label?: string | null;
  policy_lifecycle?: Lifecycle | null;
  run_count: number;
  last_run_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface Notification {
  id: number;
  event_id: number;
  channel: string;
  target: string;
  status: 'sent' | 'failed';
  detail: string | null;
  created_at: string;
  detector_name: string;
  severity: Severity;
  description: string;
}
