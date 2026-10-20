/**
 * Everything this SPA knows how to ask the appliance.
 *
 * Same origin in production (the API serves this bundle), and the vite dev
 * server proxies the same paths, so there is no base URL to configure and no
 * environment variable that can point a plant's browser at somebody else's box.
 */

import type {
  AccuracyReport,
  AnalyticsSummary,
  AppConfig,
  ArgusEvent,
  BoxCorrection,
  BoxVerdict,
  Camera,
  DetectorAccuracy,
  EventPage,
  Health,
  LiveState,
  Notification,
  Policy,
  PolicyScope,
  ProcessingReport,
  Review,
  Role,
  SavedSearch,
  SearchResponse,
  SearchStatus,
  Session,
  Severity,
  Stage,
  User,
  UserRoster,
  Verdict,
  Zone,
} from './types';

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
    readonly url: string,
    /** The API's machine-readable reason for a 401: `no_accounts` on a freshly
     *  imaged appliance, `session_required` once one has been created. */
    readonly code?: string,
  ) {
    super(`${status} ${detail}`);
    this.name = 'ApiError';
  }
}

/**
 * Told when any request comes back 401.
 *
 * A session can end between two polls — it expired, an admin disabled the
 * account, somebody revoked it — and when it does, *every* screen starts
 * failing at once. Handling that per screen would give an operator nine
 * different error bars for one cause. Instead the app registers one listener
 * (see AppState) that drops the whole console back to the login screen with a
 * sentence explaining why.
 */
let onUnauthenticated: ((code: string, detail: string) => void) | null = null;

export function setUnauthenticatedHandler(
  handler: ((code: string, detail: string) => void) | null,
): void {
  onUnauthenticated = handler;
}

/** One pydantic validation error, as FastAPI serialises it in a 422 body. */
interface ValidationError {
  loc?: unknown[];
  msg?: string;
}

function isValidationError(entry: unknown): entry is ValidationError {
  return (
    typeof entry === 'object' &&
    entry !== null &&
    typeof (entry as ValidationError).msg === 'string'
  );
}

/**
 * Turn an error body that is not a plain sentence into one.
 *
 * A 422 from FastAPI carries `detail` as an *array* of pydantic errors, so the
 * old `JSON.stringify` fallback put this in front of an operator who mistyped a
 * camera source:
 *
 *     API ERROR · 422 [{"type":"value_error","loc":["body","source"],"msg":
 *     "Value error, a file source must be an absolute path: './x.mp4'", ...}]
 *
 * The sentence the appliance wanted to say is in there, wrapped in three keys
 * that mean nothing to the person reading it. Every authoring form in the
 * console — policies, cameras, zones, accounts — funnels its errors through
 * here, so this one function is the difference between a console that explains
 * a rejection and one that shows its homework.
 */
function describeDetail(detail: unknown): string {
  if (!Array.isArray(detail)) return JSON.stringify(detail);
  const sentences = detail.filter(isValidationError).map((entry) => {
    // Pydantic prefixes anything raised as ValueError with "Value error, ".
    const message = (entry.msg ?? '').replace(/^Value error,\s*/i, '');
    // loc is ["body", "source"] or ["body", "polygon", 0] — the field is the
    // last string in it that is not the part of the request it arrived in.
    const field = (entry.loc ?? [])
      .filter((part): part is string => typeof part === 'string')
      .filter((part) => part !== 'body' && part !== 'query' && part !== 'path')
      .pop();
    return field ? `${field}: ${message}` : message;
  });
  return sentences.length > 0 ? sentences.join('; ') : JSON.stringify(detail);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    // The session cookie is HttpOnly, so there is no token to attach here —
    // `same-origin` (the default) is what sends it. Stated explicitly because
    // the absence of an Authorization header otherwise reads like an omission.
    credentials: 'same-origin',
    headers: init?.body ? { 'Content-Type': 'application/json', ...init?.headers } : init?.headers,
  });
  if (!response.ok) {
    // FastAPI puts the useful sentence in `detail`; falling back to the status
    // text alone produces "Unprocessable Entity", which helps nobody.
    let detail = response.statusText;
    let code: string | undefined;
    try {
      const body = (await response.json()) as { detail?: unknown; code?: unknown };
      if (typeof body.detail === 'string') detail = body.detail;
      else if (body.detail) detail = describeDetail(body.detail);
      if (typeof body.code === 'string') code = body.code;
    } catch {
      /* a non-JSON error body is not worth a second failure */
    }
    if (response.status === 401 && !path.startsWith('/api/auth/')) {
      // The auth routes are excluded: a failed login is a message on the login
      // form, not a session that just ended. Everything else means the console
      // is holding a session the appliance no longer honours.
      onUnauthenticated?.(code ?? 'session_required', detail);
    }
    throw new ApiError(response.status, detail, path, code);
  }
  return (await response.json()) as T;
}

function query(params: object): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== null && value !== undefined && value !== '') search.set(key, String(value));
  }
  const text = search.toString();
  return text ? `?${text}` : '';
}

export interface EventFilters {
  camera_id?: string | null;
  zone_id?: string | null;
  policy_id?: string | null;
  detector?: string | null;
  detector_type?: 'detector' | 'vlm' | null;
  stage?: Stage | null;
  severity?: Severity | null;
  verdict?: Verdict | 'unreviewed' | null;
  since_ts?: number | null;
  until_ts?: number | null;
  limit?: number;
  offset?: number;
}

/** The fields the enrolment form owns. `id` is derived from the name unless
 *  the fleet is being kept reproducible from a seed file. */
export interface CameraDraft {
  name: string;
  source: string;
  enabled?: boolean;
  detect_fps?: number | null;
  id?: string;
}

export interface CameraProbe {
  ok: boolean;
  /** False means go2rtc itself is not answering — it lives inside `argus run`,
   *  so this is not the camera's fault and the console must not say it is. */
  gateway: boolean;
  kind: string | null;
  medias?: string[];
  detail: string;
  elapsed_sec: number;
}

/** Everything Policy Studio's editor can set. Deliberately not `Policy`:
 *  `stage`, `lifecycle` and the pilot timestamps are the rollout's business
 *  and move only through the lifecycle endpoints. */
export interface PolicyDraft {
  kind: 'detector' | 'vlm';
  detector_key: string;
  label: string;
  enabled?: boolean;
  severity?: Severity;
  routed_to?: string | null;
  prompt?: string | null;
  interval_sec?: number | null;
  cooldown_sec?: number;
  threshold?: number | null;
  pilot_duration_days?: number;
  scope?: PolicyScope[];
  id?: string;
}

export interface PromptTestRequest {
  camera_id: string;
  prompt?: string;
  policy_id?: string;
  zone_id?: string | null;
  wait_sec?: number;
}

export type PromptTestEvent =
  | {
      name: 'frame';
      data: {
        camera_id: string;
        zone_id: string | null;
        zone_name: string | null;
        cropped: boolean;
        width: number;
        height: number;
        frame_width: number;
        frame_height: number;
        age_sec: number;
        image: string;
      };
    }
  | { name: 'asking'; data: { model: string; endpoint: string } }
  | {
      name: 'verdict';
      data: {
        triggered: boolean;
        confidence: number;
        reason: string | null;
        box_2d: [number, number, number, number] | null;
        latency_sec: number;
        prompt_tokens: number | null;
        completion_tokens: number | null;
        model: string;
        cropped: boolean;
      };
    }
  | { name: 'error'; data: { code: string; detail: string } };

export interface SearchQuery {
  q: string;
  camera_id?: string | null;
  since_ts?: number | null;
  until_ts?: number | null;
  kind?: 'event' | 'sample' | null;
  limit?: number;
  min_score?: number | null;
  /** Counts this run against a saved search, so the chips can order themselves
   *  by what actually gets used. */
  saved_id?: number | null;
}

export const api = {
  config: () => request<AppConfig>('/api/config'),
  health: () => request<Health>('/api/health'),
  liveState: () => request<LiveState>('/api/live/state'),

  // --- the appliance pause switch (system control plane) ---
  //
  // GET is readable by any signed-in role; POST is ADMIN-only and audited. The
  // POST returns immediately — resuming does not wait for the VLM to load — and
  // the console watches readiness by polling live state / health.
  processing: () => request<ProcessingReport>('/api/system/processing'),
  setProcessing: (enabled: boolean, note?: string) =>
    request<ProcessingReport>('/api/system/processing', {
      method: 'POST',
      body: JSON.stringify({ enabled, note: note ?? null }),
    }),

  // --- accounts (Phase 5) ---
  //
  // No token is ever handled here. The session lives in an HttpOnly cookie the
  // browser attaches on its own, which is what stops an XSS bug from walking
  // off with credentials that can download the plant's footage.
  session: () => request<{ user: Session; roles: Record<Role, string> }>('/api/auth/session'),
  login: (username: string, password: string) =>
    request<{ user: Session }>('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    }),
  logout: () => request<{ logged_out: boolean }>('/api/auth/logout', { method: 'POST' }),
  changeOwnPassword: (current_password: string, new_password: string) =>
    request<{ changed: boolean }>('/api/auth/password', {
      method: 'POST',
      body: JSON.stringify({ current_password, new_password }),
    }),

  users: () => request<UserRoster>('/api/users'),
  createUser: (body: { username: string; password: string; role: Role; display_name?: string }) =>
    request<{ user: User }>('/api/users', { method: 'POST', body: JSON.stringify(body) }),
  updateUser: (
    id: number,
    body: { role?: Role; disabled?: boolean; display_name?: string; password?: string },
  ) => request<{ user: User }>(`/api/users/${id}`, { method: 'PATCH', body: JSON.stringify(body) }),
  revokeSessions: (id: number) =>
    request<{ revoked: number }>(`/api/users/${id}/sessions/revoke`, { method: 'POST' }),

  cameras: () => request<{ cameras: Camera[] }>('/api/cameras?with_zones=true'),
  createCamera: (body: CameraDraft) =>
    request<{ camera: Camera }>('/api/cameras', { method: 'POST', body: JSON.stringify(body) }),
  updateCamera: (cameraId: string, body: Partial<CameraDraft>) =>
    request<{ camera: Camera }>(`/api/cameras/${cameraId}`, {
      method: 'PATCH',
      body: JSON.stringify(body),
    }),
  /** Ask the *gateway* to open this source, rather than opening it here. It is
   *  the component that will hold the camera in production, and many cameras
   *  allow exactly one RTSP session — a second opinion from the browser would
   *  be a different answer to a different question.
   *
   *  `password_from` is for testing an edit: the form only ever holds a masked
   *  source, and this names the enrolled camera whose stored password fills
   *  the `***`. */
  probeCamera: (body: { source?: string; camera_id?: string; password_from?: string }) =>
    request<CameraProbe>('/api/cameras/probe', { method: 'POST', body: JSON.stringify(body) }),
  zones: () => request<{ zones: Zone[] }>('/api/zones'),
  createZone: (
    cameraId: string,
    body: Pick<Zone, 'name'> & Partial<Pick<Zone, 'sanctioned_limit' | 'color'>> & {
      polygon: [number, number][];
    },
  ) =>
    request<{ zone: Zone }>(`/api/cameras/${encodeURIComponent(cameraId)}/zones`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  updateZone: (
    zoneId: string,
    body: Partial<Pick<Zone, 'name' | 'sanctioned_limit' | 'color'>> & {
      polygon?: [number, number][];
    },
  ) => request<{ zone: Zone }>(`/api/zones/${zoneId}`, { method: 'PATCH', body: JSON.stringify(body) }),

  policies: () => request<{ policies: Policy[] }>('/api/policies'),
  policyAnalytics: (policyId: string) =>
    request<{ policy: Policy; analytics: Record<string, unknown> }>(
      `/api/policies/${policyId}/analytics`,
    ),

  // --- policy authoring (Phase 5) ---
  //
  // No `stage` or `lifecycle` on create: a new policy is always born
  // draft/shadow and walks the promotion gate like every other one. The
  // console cannot skip it because the API will not let it.
  createPolicy: (body: PolicyDraft) =>
    request<{ policy: Policy }>('/api/policies', { method: 'POST', body: JSON.stringify(body) }),
  updatePolicy: (policyId: string, body: Partial<PolicyDraft> & { stage?: Stage }) =>
    request<{ policy: Policy }>(`/api/policies/${policyId}`, {
      method: 'PATCH',
      body: JSON.stringify(body),
    }),
  deletePolicy: (policyId: string) =>
    request<{ deleted: string }>(`/api/policies/${policyId}`, { method: 'DELETE' }),
  startPilot: (policyId: string) =>
    request<{ policy: Policy }>(`/api/policies/${policyId}/pilot`, { method: 'POST' }),
  extendPilot: (policyId: string, extra_days = 2) =>
    request<{ policy: Policy }>(`/api/policies/${policyId}/pilot/extend`, {
      method: 'POST',
      body: JSON.stringify({ extra_days }),
    }),
  promotePolicy: (
    policyId: string,
    body: {
      stage?: Stage;
      routed_to?: string | null;
      /** Promote past the gate. The API refuses it without a reason, and stores
       *  the reason on the policy — see `rollout:` in argus.yaml. */
      override?: boolean;
      override_reason?: string;
    } = {},
  ) =>
    request<{ policy: Policy }>(`/api/policies/${policyId}/promote`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  events: (filters: EventFilters = {}) => request<EventPage>(`/api/events${query(filters)}`),
  event: (id: number) =>
    request<{
      event: ArgusEvent;
      camera: Camera | null;
      zone: Zone | null;
      review: Review | null;
      notifications: Notification[];
      box_corrections: BoxCorrection[];
    }>(`/api/events/${id}`),
  /**
   * The triage queue, and how deep it really is.
   *
   * `total` is the whole backlog rather than this page — the nav badge asks for
   * `limit: 1` and gets the number without the quarter of a megabyte of events
   * it used to download every 15 s to count them.
   */
  triage: (options?: { limit?: number; order?: 'oldest' | 'newest' }) => {
    const params = new URLSearchParams();
    if (options?.limit !== undefined) params.set('limit', String(options.limit));
    if (options?.order) params.set('order', options.order);
    const query = params.toString();
    return request<{ events: ArgusEvent[]; total: number; order: 'oldest' | 'newest' }>(
      `/api/triage${query ? `?${query}` : ''}`,
    );
  },
  review: (id: number, verdict: Verdict, reviewer?: string) =>
    request<{ event: ArgusEvent; review: Review; accuracy: DetectorAccuracy | null }>(
      `/api/events/${id}/review`,
      { method: 'POST', body: JSON.stringify({ verdict, reviewer: reviewer ?? null }) },
    ),
  /** Judges one detection within an event, not the whole alert — see
   *  `BoxCorrection`. */
  correctBox: (eventId: number, trackId: number, verdict: BoxVerdict, note?: string) =>
    request<{ box_corrections: BoxCorrection[] }>(
      `/api/events/${eventId}/boxes/${trackId}/correction`,
      { method: 'POST', body: JSON.stringify({ track_id: trackId, verdict, note: note ?? null }) },
    ),

  // --- semantic search (Phase 3) ---
  //
  // `search` sends the operator's sentence verbatim; the caption template that
  // helps SigLIP is applied server-side, so what a saved search stores is what
  // the operator typed and not a prompt-engineered variant of it.
  search: (params: SearchQuery) => request<SearchResponse>(`/api/search${query({ ...params })}`),
  searchStatus: () => request<SearchStatus>('/api/search/status'),
  savedSearches: () => request<{ searches: SavedSearch[] }>('/api/searches'),
  saveSearch: (body: {
    query: string;
    label?: string | null;
    camera_id?: string | null;
    window_hours?: number | null;
  }) => request<{ search: SavedSearch }>('/api/searches', {
    method: 'POST',
    body: JSON.stringify(body),
  }),
  deleteSavedSearch: (id: number) =>
    request<{ deleted: number }>(`/api/searches/${id}`, { method: 'DELETE' }),
  /** Search -> draft VLM policy. Always lands in draft/shadow; the pilot gate
   *  is the product, and this button does not skip it. */
  promoteSearch: (id: number, body: { label?: string; severity?: Severity } = {}) =>
    request<{ policy: Policy; search: SavedSearch }>(`/api/searches/${id}/promote`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  accuracy: (days = 14) => request<AccuracyReport>(`/api/accuracy?days=${days}`),
  analytics: (days = 30) => request<AnalyticsSummary>(`/api/analytics/summary?days=${days}`),
  notifications: (limit = 50) =>
    request<{ notifications: Notification[] }>(`/api/notifications?limit=${limit}`),
};

/**
 * Run a prompt against a camera's current frame, streaming what happens.
 *
 * `POST` with a body rules out `EventSource`, which only does GET — so this
 * reads the response body itself and splits SSE frames out of it. The stream
 * is worth the extra thirty lines: the frame arrives immediately and the model
 * takes ~5 s on this appliance, and showing the picture first answers half the
 * questions an author has ("is that even the right camera?") without waiting.
 *
 * `onFrame` is called for every SSE event in order. The returned function
 * aborts — a panel that is closed mid-inference must not keep the socket, and
 * must not call back into an unmounted component.
 */
export function testPrompt(
  body: PromptTestRequest,
  onFrame: (event: PromptTestEvent) => void,
): () => void {
  const controller = new AbortController();
  void (async () => {
    try {
      const response = await fetch('/api/policies/test', {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (!response.ok || !response.body) {
        const detail = (await response.json().catch(() => ({}))) as { detail?: string };
        onFrame({ name: 'error', data: { code: 'refused', detail: detail.detail ?? response.statusText } });
        return;
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        // SSE frames are separated by a blank line. Anything after the last
        // separator is a partial frame and stays in the buffer — a 130 KB
        // data: URI arrives in several chunks, and parsing half of one would
        // throw away the picture.
        const blocks = buffer.split('\n\n');
        buffer = blocks.pop() ?? '';
        for (const block of blocks) {
          const name = block.match(/^event: (.+)$/m)?.[1];
          const data = block.match(/^data: (.*)$/m)?.[1];
          if (!name || data === undefined) continue;
          try {
            onFrame({ name, data: JSON.parse(data) } as PromptTestEvent);
          } catch {
            /* a malformed frame must not end the stream */
          }
        }
      }
    } catch (error) {
      if (controller.signal.aborted) return;
      onFrame({
        name: 'error',
        data: { code: 'network', detail: error instanceof Error ? error.message : String(error) },
      });
    }
  })();
  return () => controller.abort();
}

/** Evidence URLs. The API mounts these directories with Range support, which is
 *  what lets a reviewer scrub an mp4 rather than wait for the whole file. */
export const evidence = {
  /**
   * `version` busts the browser cache for an **ongoing condition**, whose
   * snapshot is rewritten in place at the same path every time the model sees
   * the mess again (schema migration 7 — one evidence file per condition rather
   * than one per sighting, which for the laminate that started all this would
   * have been forty JPEGs of the same floor). Pass the row's `last_seen_ts`.
   * Without it the card would faithfully report "12 observations" beside the
   * picture taken at the first.
   */
  snapshot: (path: string | null, version?: number | null): string | null =>
    path ? `/snapshots/${path}${version ? `?v=${Math.trunc(version)}` : ''}` : null,
  clip: (path: string | null): string | null => (path ? `/clips/${path}` : null),
  tracks: (path: string | null): string | null => (path ? `/tracks/${path}` : null),
};
