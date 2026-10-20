/**
 * Semantic Search — the index service, live.
 *
 * The layout is the one the design specifies and the one this screen already
 * had while it was inert: query bar, saved chips, a chronological strip of
 * matches with a confidence bar, and a detail panel with the clip and the
 * promote-to-policy affordance. What changed in Phase 3 is that the results
 * are real.
 *
 * Three decisions worth stating, because each is a place where a search UI can
 * quietly lie to an operator:
 *
 * **The strip is chronological; the cut is by relevance.** The API ranks by
 * similarity and returns the best `limit` frames; this sorts those newest-first
 * for reading. Sorting by time *before* cutting would return the most recent
 * frames rather than the best ones, and an operator would never know the
 * difference.
 *
 * **The score is shown raw, against a fixed ceiling.** SigLIP cosine
 * similarities are small absolute numbers — 0.10-0.17 is a strong match on real
 * footage. Rescaling per query so the best hit always fills the bar would make
 * a search that found nothing look identical to one that found everything.
 *
 * **The header says what was actually searched.** The design's line reads
 * "SEARCHED 214 H OF FOOTAGE IN 3.1 S"; the honest version is the index's own
 * span and frame count, which is what /api/search returns. An appliance that
 * indexed nothing yesterday says so rather than implying it looked.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { api } from '../api/client';
import type { SearchQuery } from '../api/client';
import type { Camera, SavedSearch, SearchResponse, SearchResult } from '../api/types';
import { EvidenceViewer } from '../components/EvidenceViewer';
import { ErrorBar, NotEnabled, OriginBadge, SeverityGlyph } from '../components/marks';
import { useResource } from '../lib/hooks';
import { dayAndTime, relativeTime } from '../lib/format';
import { useAppState } from '../state/AppState';

const EXAMPLES = [
  'person near the conveyor',
  'stacked crates blocking a walkway',
  'someone alone on the floor',
  'a forklift moving through the aisle',
];

/** Similarity the bar treats as full. Not the maximum a cosine can reach —
 *  0.18 is about the best a SigLIP query scores against real CCTV, so drawing
 *  against 1.0 would leave every bar looking empty and every match looking
 *  doubtful. Fixed rather than per-query, so two searches are comparable. */
const SCORE_CEILING = 0.18;

const WINDOWS: { label: string; hours: number | null }[] = [
  { label: 'Any time', hours: null },
  { label: 'Last hour', hours: 1 },
  { label: 'Last shift', hours: 8 },
  { label: 'Last 24 h', hours: 24 },
  { label: 'Last 7 days', hours: 24 * 7 },
];

export function SearchScreen() {
  const { config } = useAppState();
  const enabled = config?.features.search ?? false;

  const [text, setText] = useState('');
  const [cameraId, setCameraId] = useState<string | null>(null);
  const [windowHours, setWindowHours] = useState<number | null>(null);
  const [eventsOnly, setEventsOnly] = useState(false);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<SearchResult | null>(null);
  const [openEvent, setOpenEvent] = useState<number | null>(null);
  const [note, setNote] = useState<string | null>(null);

  const status = useResource(() => api.searchStatus(), []);
  const cameras = useResource(() => api.cameras(), []);
  const saved = useResource(() => api.savedSearches(), []);

  const run = useCallback(
    async (query: string, options: Partial<SearchQuery> = {}) => {
      const trimmed = query.trim();
      if (!trimmed) return;
      setBusy(true);
      setError(null);
      setSelected(null);
      try {
        const params: SearchQuery = {
          q: trimmed,
          camera_id: options.camera_id ?? cameraId,
          kind: (options.kind ?? (eventsOnly ? 'event' : null)) as SearchQuery['kind'],
          limit: config?.index.default_limit ?? 24,
          saved_id: options.saved_id ?? null,
        };
        const hours = options.since_ts === undefined ? windowHours : null;
        if (hours) params.since_ts = Date.now() / 1000 - hours * 3600;
        if (options.since_ts !== undefined) params.since_ts = options.since_ts;
        setResponse(await api.search(params));
      } catch (cause) {
        setResponse(null);
        setError(cause instanceof Error ? cause.message : String(cause));
      } finally {
        setBusy(false);
      }
    },
    [cameraId, windowHours, eventsOnly, config],
  );

  // Re-run on a filter change, but only once a search exists: changing the
  // camera before typing anything should not fire an empty query.
  useEffect(() => {
    if (response) void run(response.query);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cameraId, windowHours, eventsOnly]);

  const ordered = useMemo(
    () => [...(response?.results ?? [])].sort((a, b) => b.ts - a.ts),
    [response],
  );

  const runSaved = (entry: SavedSearch) => {
    setText(entry.query);
    setCameraId(entry.camera_id);
    setWindowHours(entry.window_hours);
    void run(entry.query, {
      camera_id: entry.camera_id,
      saved_id: entry.id,
      since_ts: entry.window_hours ? Date.now() / 1000 - entry.window_hours * 3600 : null,
    });
  };

  const save = async () => {
    if (!response) return;
    await api.saveSearch({
      query: response.query,
      camera_id: cameraId,
      window_hours: windowHours,
    });
    saved.reload();
    setNote('Saved. It will run with the same camera and window next time.');
  };

  const promote = async () => {
    if (!response) return;
    setNote(null);
    try {
      // Saving first is not a shortcut: a policy has to come *from* a saved
      // search so the provenance ("this prompt started life as this query")
      // has a row to live on.
      const entry =
        saved.data?.searches.find((s) => s.query === response.query) ??
        (await api.saveSearch({ query: response.query, camera_id: cameraId, window_hours: windowHours }))
          .search;
      const { policy } = await api.promoteSearch(entry.id);
      saved.reload();
      setNote(
        `Created draft policy “${policy.label}”. It is in Policy Studio, in shadow — ` +
          'start a pilot there when you want it evaluated against live frames.',
      );
    } catch (cause) {
      setNote(cause instanceof Error ? cause.message : String(cause));
    }
  };

  const coverage = response?.index ?? status.data?.index ?? null;
  const emptyIndex = (coverage?.frames ?? 0) === 0;

  return (
    <div className="rise search">
      <div className="search__inner">
        <div className="screen-title">Search recorded footage</div>

        <form
          className="search__bar"
          onSubmit={(event) => {
            event.preventDefault();
            void run(text);
          }}
        >
          <div className="search__field">
            <span style={{ color: 'var(--faint)' }}>⌕</span>
            <input
              value={text}
              disabled={!enabled}
              placeholder="Describe what to look for, in plain English"
              onChange={(event) => setText(event.target.value)}
              autoFocus
            />
          </div>
          <button
            className="button"
            type="submit"
            disabled={!enabled || busy || !text.trim()}
            style={{ padding: '14px 24px' }}
          >
            {busy ? 'Searching…' : 'Search'}
          </button>
        </form>

        <div className="search__saved">
          <span className="eyebrow">{saved.data?.searches.length ? 'Saved' : 'Try'}</span>
          {(saved.data?.searches ?? []).map((entry) => (
            <span key={entry.id} className="search__savedchip">
              <button
                className="search__chip"
                disabled={!enabled}
                onClick={() => runSaved(entry)}
                title={
                  entry.policy_id
                    ? `Promoted to policy ${entry.policy_id} · run ${entry.run_count}×`
                    : `Run ${entry.run_count}×`
                }
              >
                {entry.policy_id ? '◆ ' : ''}
                {entry.label}
              </button>
              <button
                className="search__chipx"
                title="Forget this search"
                onClick={async () => {
                  await api.deleteSavedSearch(entry.id);
                  saved.reload();
                }}
              >
                ×
              </button>
            </span>
          ))}
          {saved.data?.searches.length === 0
            ? EXAMPLES.map((example) => (
                <button
                  key={example}
                  className="search__chip"
                  disabled={!enabled}
                  onClick={() => {
                    setText(example);
                    void run(example);
                  }}
                >
                  {example}
                </button>
              ))
            : null}
        </div>

        {enabled ? (
          <div className="search__filters">
            <select
              className="search__select"
              value={cameraId ?? ''}
              onChange={(event) => setCameraId(event.target.value || null)}
            >
              <option value="">All cameras</option>
              {(cameras.data?.cameras ?? []).map((camera: Camera) => (
                <option key={camera.id} value={camera.id}>
                  {camera.name}
                </option>
              ))}
            </select>
            <select
              className="search__select"
              value={windowHours ?? ''}
              onChange={(event) =>
                setWindowHours(event.target.value ? Number(event.target.value) : null)
              }
            >
              {WINDOWS.map((option) => (
                <option key={option.label} value={option.hours ?? ''}>
                  {option.label}
                </option>
              ))}
            </select>
            <label className="search__toggle">
              <input
                type="checkbox"
                checked={eventsOnly}
                onChange={(event) => setEventsOnly(event.target.checked)}
              />
              Alerts only
            </label>
          </div>
        ) : null}

        <ErrorBar error={error ?? status.error} />

        {!enabled ? (
          <NotEnabled title="Search index not enabled on this appliance" phase="index.enabled">
            Natural-language search needs the index service: SigLIP 2 embeddings of frames written
            to a local vector store as footage is recorded. It is switched off in this appliance's
            configuration, so there is nothing to search and this screen will not pretend otherwise.
            <br />
            <br />
            Set <span className="mono">index.enabled: true</span> in{' '}
            <span className="mono">config/argus.yaml</span> and restart the engine. Everything it
            will search is already being kept.
          </NotEnabled>
        ) : status.data && !status.data.model_ready ? (
          <NotEnabled title="The embedding model is not installed" phase={status.data.model}>
            The index is enabled but the SigLIP 2 checkpoint is not on this box, so no frame can be
            embedded and no query can be answered. Run{' '}
            <span className="mono">scripts/fetch_siglip.sh</span>, then{' '}
            <span className="mono">argus index backfill</span> to make the evidence already here
            searchable.
          </NotEnabled>
        ) : emptyIndex && !response ? (
          <NotEnabled title="Nothing has been indexed yet" phase="empty index">
            The model is installed and search is on, but the store is empty. Frames are indexed as{' '}
            <span className="mono">argus run</span> decodes them — one sample per camera every{' '}
            {config?.index.sample_interval_sec ?? 10} s, plus the frame behind every alert. To make
            the evidence already on this box searchable now, run{' '}
            <span className="mono">argus index backfill</span>.
          </NotEnabled>
        ) : null}

        {response ? (
          <>
            <div className="mono search__summary">
              {response.count} MATCHED {response.count === 1 ? 'MOMENT' : 'MOMENTS'}
              {response.filters.camera_id ? ` · ${response.filters.camera_id.toUpperCase()}` : ''}
              {windowHours ? ` · LAST ${windowHours} H` : ''} · SEARCHED{' '}
              {coverage?.frames.toLocaleString('en-GB') ?? 0} INDEXED FRAMES SPANNING{' '}
              {coverage?.span_hours ?? 0} H IN {response.took_ms.toFixed(0)} MS
            </div>

            {response.count === 0 ? (
              <div className="notice">
                <div className="notice__title">Nothing scored above {response.min_score}</div>
                <div className="notice__body">
                  The index holds {coverage?.frames ?? 0} frames and none of them resembles this
                  description closely enough to be worth showing. Try fewer words, a wider time
                  window, or a different camera — and remember the index only covers what has been
                  recorded since it was switched on.
                </div>
              </div>
            ) : null}

            <div className="search__results">
              {ordered.map((result) => (
                <ResultTile
                  key={result.id}
                  result={result}
                  on={selected?.id === result.id}
                  onClick={() => setSelected(selected?.id === result.id ? null : result)}
                />
              ))}
            </div>
          </>
        ) : null}

        {selected ? (
          <Detail
            result={selected}
            query={response?.query ?? ''}
            model={coverage?.model ?? ''}
            interval={config?.index.sample_interval_sec ?? 10}
            note={note}
            onOpenEvidence={() => setOpenEvent(selected.event?.id ?? null)}
            onSave={save}
            onPromote={promote}
          />
        ) : null}
      </div>

      {openEvent !== null ? (
        <EvidenceViewer
          eventId={openEvent}
          title="Search result"
          onClose={() => setOpenEvent(null)}
          onReviewed={() => undefined}
        />
      ) : null}
    </div>
  );
}

function ResultTile({
  result,
  on,
  onClick,
}: {
  result: SearchResult;
  on: boolean;
  onClick: () => void;
}) {
  const fill = Math.max(2, Math.min(100, (result.score / SCORE_CEILING) * 100));
  return (
    <button className={`search__tile ${on ? 'search__tile--on' : ''}`} onClick={onClick}>
      <div className="search__tileframe">
        <img src={result.thumb_url} alt="" loading="lazy" />
        {result.event ? (
          <span className="search__tilebadge mono">
            <SeverityGlyph severity={result.event.severity} size={9} /> ALERT
          </span>
        ) : null}
      </div>
      <div className="search__tilemeta">
        <div className="mono search__tileline">
          <span>{result.camera_name}</span>
          <span style={{ color: 'var(--faint)' }}>{dayAndTime(result.ts)}</span>
        </div>
        <div className="search__tilescore">
          <div className="search__confbar">
            <div style={{ width: `${fill}%` }} />
          </div>
          <span className="mono" style={{ fontSize: 'var(--fs-note)', color: 'var(--accent-text)' }}>
            {result.score.toFixed(3)}
          </span>
        </div>
      </div>
    </button>
  );
}

function Detail({
  result,
  query,
  model,
  interval,
  note,
  onOpenEvidence,
  onSave,
  onPromote,
}: {
  result: SearchResult;
  query: string;
  model: string;
  interval: number;
  note: string | null;
  onOpenEvidence: () => void;
  onSave: () => void;
  onPromote: () => void;
}) {
  return (
    <div className="search__detail">
      <div className="search__detailframe">
        <img src={result.thumb_url} alt="" />
      </div>
      <div className="search__detailbody">
        <div style={{ fontWeight: 600, fontSize: 'var(--fs-body)', display: 'flex', gap: 9, alignItems: 'center' }}>
          Match · {result.camera_name} · {dayAndTime(result.ts)}
          {result.event ? <OriginBadge type={result.event.detector_type} short /> : null}
        </div>
        <div style={{ fontSize: 'var(--fs-meta)', color: 'var(--dim)', lineHeight: 1.6 }}>
          {result.event ? (
            <>
              This frame is the evidence behind alert #{result.event.id}: “
              {result.event.description}”.{' '}
            </>
          ) : (
            <>
              A routine sample — no policy fired on this frame. The index takes one per camera every{' '}
              {interval} s, which is how footage nobody was alerted about stays findable.{' '}
            </>
          )}
          Similarity to “{query}” is {result.score.toFixed(3)}, {relativeTime(result.ts)}. Embedded
          on this box by {model || 'the local index model'}; nothing about this search left the
          appliance.
        </div>
        {note ? <div className="search__note">{note}</div> : null}
        <div className="search__actions">
          <button className="button button--soft" onClick={onPromote}>
            Promote to draft policy →
          </button>
          <button className="button button--quiet" onClick={onSave}>
            Save search
          </button>
          {result.event ? (
            <button className="button button--ghost" onClick={onOpenEvidence}>
              Open evidence ↗
            </button>
          ) : (
            <span className="mono" style={{ fontSize: 'var(--fs-note)', color: 'var(--faint)', alignSelf: 'center' }}>
              NO CLIP — SAMPLED FRAME, NOT AN ALERT
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
