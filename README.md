Video Demo: https://www.youtube.com/watch?v=BDxK0rsxUMo


<img width="3840" height="2160" alt="image" src="https://github.com/user-attachments/assets/d6935707-8f80-41ba-a1cb-c5b3aee2def8" />



# Argus

**Industrial safety video analytics that runs entirely on one box in your plant.**

Argus watches the cameras you already have, counts people against sanctioned
limits per zone, and raises evidence-backed alerts — an annotated frame plus the
±8 seconds of video around it. Every frame is decoded, analysed and stored on an
NVIDIA DGX Spark on the plant LAN. No raw video leaves site, no per-camera
licence, no cloud account, and nothing stops working when the internet does.

> **Status: v2 rebuild in progress.** The ingest, vision and API planes are
> built and measured, the local VLM policy engine runs on box, natural-language
> search works over everything recorded, the operator console is built against
> all of them, and the whole thing is now **behind a login** with four roles and
> one authenticated door to the video gateway (Phases 0–4 and the first half of
> Phase 5 of [`docs/REBUILD_PLAN.md`](docs/REBUILD_PLAN.md); posture and residual
> exposure in [`docs/SECURITY.md`](docs/SECURITY.md)). Cameras and policies are
> **written in the console**, not only in a seed file: a policy names the cameras
> and zones it applies to, its prompt can be run against that camera's current
> frame before anyone commits to it, and it reaches auto-routed only by clearing
> the precision bar — which the appliance enforces, rather than the browser. Ask *"person near the
> conveyor"* and get ranked frames back in ~110 ms, from a SigLIP 2 index this
> box wrote itself — measured at 0.3% of one GPU at demo scale. Both the VLM and
> the index are optional equipment: with either stopped or uninstalled the
> console says so, with the reason and the command that fixes it, and nothing
> else on the appliance changes. It now **installs as an appliance rather than a
> checkout**: one resumable, air-gappable installer, systemd user units that
> restart the engine 5 s after a `kill -9`, nightly SQLite backups with a
> measured 0.29 s restore, and `argus doctor` — one command that says whether
> this box is healthy and exits non-zero when it is not. **v1 — the working prototype this
> replaces — has been removed from this branch** and is preserved in full on
> **`v1-archive`** (`git show v1-archive:pipeline/live.py`, and the same for
> `dashboard/`, `detectors/`, `demo.sh` and its docs). Removing it is what let
> the AGPL dependency go: see [Licensing](#licensing).

---

## Why an appliance

The video-analytics market sells enterprise SaaS at $30K–$200K per year with
per-camera licence lock-in, or dumb NVRs with no analytics at all. Between them
sits the plant with 20–100 cameras that wants its footage to stay in the
building. Argus is built for that gap:

- **Fully local.** RTSP in, WebRTC out, inference and storage on the same
  machine. Air-gapped installs are the design point, not a configuration option.
- **Flat per-site pricing.** One appliance covers the plant. Adding a camera
  costs decode capacity, not a licence.
- **Staged rollout, not blind trust.** Every detector runs silently first, then
  to a human reviewer, and only reaches an HOD's inbox once its measured
  false-positive rate says it has earned that.
- **Evidence-first.** An alert nobody can verify is an alert nobody acts on, so
  every event carries an annotated snapshot, a clip, and the track overlays.
- **Bounded storage.** Retention is a hard requirement — see below.

Hardware: **NVIDIA DGX Spark** (GB10 Grace Blackwell, 128 GB unified memory,
20 Grace CPU cores, one NVDEC, one NVENC), ~$4,699.

---

## Quickstart

```bash
./scripts/install.sh --services   # everything below, in order, resumable
```

That is the appliance install — environment, gateway, ffmpeg, model weights,
console, database, first account, first backup, systemd units — and it ends by
running `argus doctor`. Full walkthrough, including air-gapped pre-staging:
[`docs/INSTALL.md`](docs/INSTALL.md). By hand, it is:

```bash
uv sync                       # Python 3.12; torch 2.12.1+cu130 is in the lock
./scripts/fetch_gateway.sh    # vendors go2rtc (MIT) into bin/, pinned by sha256
./scripts/build_ffmpeg.sh     # builds LGPL ffmpeg with NVDEC/NVENC into bin/
./scripts/fetch_detector.sh   # RF-DETR weights, so first boot needs no network
./scripts/fetch_siglip.sh     # SigLIP 2, for natural-language search

(cd ui && npm ci && npm run build)   # the operator console -> ui/dist

uv run argus seed             # import config/seed.yaml -> cameras, zones, policies
uv run argus user add ops --role ADMIN   # the first account — see below
uv run argus run              # gateway + decode + detect + track + evidence
uv run argus api              # console + JSON API on :8080
```

Then:

- `http://<appliance>:8080/` — **the console**: sign in, then live wall, alerts, triage, search, analytics
- `http://<appliance>:8080/api/docs` — interactive API reference, served locally
- `http://<appliance>:8080/api/health` — liveness; engine, per-camera fps, GPU and disk once signed in

Both `argus run` and `argus api` are meant to stay up — the first is the vision
engine, the second serves the console and the API. Neither needs the other to
start, and the console says plainly when the engine is not running rather than
showing a still picture behind a green light. On a real appliance neither is
run by hand: `argus install-services` puts both behind systemd (below).

### The first account

**A freshly installed appliance has no accounts and refuses every request**, and
there is no default password and no open first-run window. The only credential a
fresh box has is physical access to it, so the bootstrap happens on the box:

```bash
uv run argus user add ops --role ADMIN --name "J. Whitfield"
```

It prompts for the password twice, and reads one from stdin when there is no
terminal — so an offline installer can do this unattended:

```bash
printf '%s\n' "$PASSWORD" | uv run argus user add ops --role ADMIN
```

Until it is run, `argus api` prints a warning at startup, `/api/health` reports
`"accounts": 0`, and the console shows that command instead of a login form no
password could satisfy. Afterwards, everything else is done from Settings →
Users & roles in the console; the CLI stays as the recovery path:

```bash
uv run argus user list                       # roster, roles, last sign-in
uv run argus user passwd rc                  # reset a password (ends its sessions)
uv run argus user role rc --role RESPOND     # change a role
uv run argus user disable rc                 # sign out everywhere and refuse
```

**Four roles**, exactly as the console's Settings screen names them:

| Role | Can |
|---|---|
| `ADMIN` | Everything, including accounts and roles |
| `RESPOND` | Triage alerts, and manage policies, cameras and zones |
| `REVIEW` | Triage alerts: confirm or reject, and search evidence |
| `SYSTEM` | Health and metrics only — no footage, no events |

Every route except the `/api/health` liveness probe and the login endpoint needs
a session, and so do the evidence mounts (`/snapshots`, `/clips`, `/tracks`).
Verdicts are recorded against the person who signed in. Sessions last 12 hours,
log out after 8 idle, and end immediately when an account is disabled, demoted or
has its password changed. Login is rate-limited per account and per source
address. The full posture — including what is **not** protected, which is the
WebRTC media port — is [`docs/SECURITY.md`](docs/SECURITY.md).

### The VLM policy engine

Plain-English policies ("is the emergency exit blocked?") are evaluated on box by
a local vision-language model. It is **optional equipment**: the appliance runs
without it, and the console reports it as down with the command that fixes it
rather than failing. Two install-time steps, both of which need the network and
neither of which runs while the appliance is serving:

```bash
./scripts/setup_vlm_server.sh   # builds .venv-vlm: vLLM nightly cu130 aarch64 + its own torch
./scripts/fetch_vlm.sh          # 17 GB of Qwen3-VL-8B-Instruct (Apache-2.0) -> models/
```

Then, whenever you want prompt policies running:

```bash
uv run argus vlm start          # loads the model; ~4 min cold, ~2 min warm
uv run argus vlm health         # model, endpoint, latency — or why it is not up
uv run argus vlm stop           # frees ~18 GB of GPU
```

`argus run` picks the server up on its own and starts sampling the enabled prompt
policies; nothing else needs restarting. The server binds `127.0.0.1:8000` and is
never exposed. Model Health in the console shows evaluations, average latency and
per-policy dropped samples once it is up.

Both scripts are install-time only and are documented in their own headers —
including why this is a wheel install rather than the NGC container, and why the
model is Qwen3-VL-8B rather than the plan's Cosmos-Reason2-8B. See
[`docs/PHASE2_NOTES.md`](docs/PHASE2_NOTES.md) for what shipped, what it costs,
and why event verification ships switched off.

**Running a different model is one line.** `vlm.model` in `config/argus.yaml`,
plus `./scripts/fetch_vlm.sh <that id>` once. Two alternatives are measured on
this hardware and both are documented as *held, not shipped*, with the trade in
[`docs/VLM_EVAL.md`](docs/VLM_EVAL.md): `Qwen/Qwen3.8-27B-FP8` answers headcount
claims correctly and costs 8.8 s an evaluation, and `Qwen/Qwen3-VL-8B-Instruct-FP8`
costs 2.7 s and answers them worse.

### Natural-language search

Ask for what you remember seeing — *"person near the conveyor"*, *"stacked
crates blocking a walkway"* — and get ranked frames out of everything the box has
recorded. One install-time step, 0.77 GB:

```bash
./scripts/fetch_siglip.sh       # google/siglip2-base-patch16-384 (Apache-2.0) -> models/
uv run argus index backfill     # make the evidence already on disk searchable
uv run argus index status       # how much is searchable, by which model
```

`argus run` then indexes as it goes — one frame per camera every 10 s plus the
frame behind every alert, reusing frames the vision loop has already decoded, so
there is no second ffmpeg anywhere in this path. A query costs 105–129 ms and
indexing costs 0.3% of one GPU at the two-camera demo; the expensive half was
paid once, when the frame was recorded. Also optional equipment: with
`index.enabled: false` search still answers from whatever is already indexed,
and with the checkpoint absent the Search screen names the script above.

A search that keeps finding the same violation can be **promoted to a draft VLM
policy** from the same screen — the query becomes the prompt, verbatim, and the
policy starts in shadow and walks the ordinary pilot gate.
[`docs/PHASE3_NOTES.md`](docs/PHASE3_NOTES.md) has the costs, the checkpoint
comparison, and the two things worth knowing before trusting a result: SigLIP
does not do negation, and it reads text burned into a frame — including the
headline Argus itself draws on evidence snapshots.

### Choosing a model on evidence

`argus eval` scores candidate vision models on *this box's* policies, footage and
triage verdicts, and writes [`docs/VLM_EVAL.md`](docs/VLM_EVAL.md) plus a
per-case JSON dump:

```bash
uv run argus eval --build-labels   # sample frames into evals/labels.yaml
$EDITOR evals/labels.yaml          # a human fills in headcounts and verdicts
uv run argus eval                  # score whatever `argus vlm start` is serving
```

Three slices: reviewed events (labels the deployment produces for free),
hand-labelled frames, and a **counting slice** that generates headcount claims in
the engine's own wording and asks each one under both the production
verification prompt and a frozen count-aware copy of it. That slice exists
because `vlm.verify.enabled` ships `false` on the strength of one 2026-08-09
result, and running it is how that decision stops being folklore.

It has already paid twice. The verifier's failure was never that it could not
count — it was that nothing asked it to, and that it was shown a whole camera
view when the claim was about a zone. Both are fixed and both were measured on
the same model and the same frames: adding one clause took over-claims from
15/24 to 24/24 vetoed, and the same claims on a zone crop rather than the wide
shot they were cut from went 3/12 to 12/12. The ladders are in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md), the before/after is in
[`docs/VLM_EVAL.md`](docs/VLM_EVAL.md), and `vlm.verify.enabled` still ships
`false` — now as an opt-in with a table behind it rather than a disabled feature
with a caveat.

Cloud candidates (OpenRouter — Qwen 3.8, Gemini 3.7 Flash, Kimi K3) are
**development-only** and behind two gates: `--cloud` *and* `OPENROUTER_API_KEY`.
A key alone is not consent. No module on any runtime path imports that code, and
three tests assert it — importing the engine, the API and the scheduler leaves
`argus.eval` out of `sys.modules` entirely. See
[`docs/EVAL_NOTES.md`](docs/EVAL_NOTES.md).

### The console

Vite + React + TypeScript in [`ui/`](ui/), built to `ui/dist` and served at `/`
by `argus api` (any deep link reloads correctly; `/api` stays JSON). Fonts are
bundled — the page makes no external request at all, which is checkable:

```bash
grep -roE "https?://[a-zA-Z0-9.-]+" ui/dist --include="*.js" --include="*.css"
```

Two hosts appear and neither is fetched: `https://reactjs.org` is the URL in
React's own error-message text, and `http://www.w3.org` is the SVG XML
namespace, which is an identifier rather than an address. Anything else in that
output is a regression.

For UI development, run the vite dev server alongside the appliance. It proxies
`/api` and the evidence mounts — and since the gateway relay lives under
`/api/gateway`, there is no second proxy rule and no second port:

```bash
uv run argus api                                  # :8080, as usual
cd ui && npm run dev                              # :5173, proxying to :8080
# ARGUS_API_URL points it at another box if you want
```

`npm run build` typechecks first (`tsc -b`, strict), so a type error fails the
build rather than shipping.

If the console loads but every tile is black, the browser could not complete the
WebRTC handshake. Signalling is relayed through the API and is therefore
already reachable if the console loaded at all — so the usual cause is the
**media** path: `webrtc_port` (8555 by default) must be reachable from the
operator's machine, because ICE connects the browser to it directly. See
[`docs/SECURITY.md`](docs/SECURITY.md).

Useful variations:

```bash
uv run argus run --duration 60 --report-every 10   # a bounded smoke run
uv run argus run --no-gateway                      # decode straight from sources
uv run argus status                                # config, database, fleet
uv run argus retention --dry-run                   # what storage is being used
uv run argus eval                                  # score the VLM on this box's labels
uv run argus detector-eval                         # score the detector on this site's labels
uv run argus user list                             # who can sign in, and as what
uv run pytest -q                                   # the test suite (GPU-optional)
uv run python scripts/license_audit.py --check     # no copyleft in the product tree

uv run python scripts/benchmark.py detector        # throughput vs batch size
uv run python scripts/benchmark.py decode          # NVDEC vs software decode ceiling
uv run python scripts/benchmark.py capacity        # N cameras through the real engine
```

`bin/ffmpeg` matters: distribution ffmpeg builds are usually GPL *and* built
without CUDA, and `argus/ingest/decoder.py` prefers the vendored one precisely so
the appliance is neither. Without it Argus still runs — decode falls back to
software and the engine reports `mode=cpu`.

Configuration lives in [`config/argus.yaml`](config/argus.yaml). Every scalar can
be overridden from the environment by its dotted path — `detection.fps` becomes
`ARGUS_DETECTION_FPS`. Secrets are never stored in the file: `smtp.password_env`
names the variable that holds the password.

---

## Architecture

Everything runs on the Spark. One process (`argus run`) owns the live path; a
second (`argus api`) serves JSON; SQLite in WAL mode is the bus between them.

| Service | Tech | Role |
|---|---|---|
| stream-gateway | go2rtc (MIT) | RTSP/ONVIF in; WebRTC out to the UI; one connection per physical camera, fanned out. Its HTTP API binds **loopback only**; the console reaches signalling and poster stills through the authenticated relay at `/api/gateway/*` |
| vision-engine | Python 3.12 + torch FP16 | NVDEC decode; RF-DETR detection batched across cameras at 5 fps on sub-streams; ByteTrack via `supervision`; zone/occupancy analytics; debounce; evidence |
| vlm-service | vLLM in its own venv (`argus vlm start`) | Qwen3-VL-8B-Instruct (Apache-2.0; Cosmos Reason 2 is a gated fine-tune of the same base), structured-JSON outputs via xgrammar; prompt-policy sampling + event verification (verification ships off — see docs/BENCHMARKS.md). `vlm.model` is a one-line swap; Qwen3.8-27B-FP8 was trialled on the same venv and held for latency (docs/VLM_EVAL.md) |
| index-service | SigLIP 2 + sqlite-vec (`argus/index/`) | Embeds one frame per camera every 10 s plus the frame behind every alert, into a local vector store beside the appliance database; `GET /api/search` ranks them against plain English, saved searches promote to draft VLM policies. `argus index backfill` makes existing evidence searchable |
| eval-harness | `argus eval` (`argus/eval/`) | **Development only, not a runtime path.** Scores candidate vision models on this box's triage verdicts, hand-labelled frames and generated headcount claims; writes `docs/VLM_EVAL.md` with a ship/hold verdict per candidate. `--baseline` merges an earlier run so two local models land in one table (vLLM serves one at a time). Cloud candidates need `--cloud` *and* an API key, and no runtime module can import that code |
| core-api | FastAPI | Cameras, zones, policies, events, verdicts, SSE, health, retention — and, since Phase 5, local accounts (Argon2id), server-side sessions, an RBAC matrix over four roles, and the go2rtc relay. Deny-by-default: a route is protected by existing, including the evidence mounts |
| alerts | SMTP | Email to the role a policy is routed to, once it reaches `auto_routed` |
| ui | Vite + React + TS SPA | WebRTC live wall, alert feed, keyboard triage, zone editor, analytics; served by the API at `/`; self-hosted fonts, zero CDN |
| ops | systemd / Compose (arm64) | Offline installer, metrics, model manager *(rest of Phase 5)* |

Deliberately **not** adopted wholesale: NVIDIA's VSS blueprint (heavy reference
microservices, ~10-minute cold starts — components and patterns borrowed
instead) and DeepStream (revisit at 9.1 if stream counts demand it).

### Measured on the target hardware

Not extrapolated — every number below came off this box. Full tables and method
in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

| | |
|---|---|
| **Cameras at 5 fps detection, one box** | **48** — 4.99 fps achieved, 38 ms lag, 3.5 of 20 cores, 52% GPU |
| Decode chain | frames stay on the GPU until after the 5 fps sample — **10.5×** the old chain at 1080p, byte-identical output |
| NVDEC decode ceiling | ~5 600 fps of sub-stream H.264 (**~220 cameras**), or ~42 streams of 1080p |
| RF-DETR Nano, FP16 traced, end to end | **1.61 ms/frame** at batch 8 (621 img/s); 3.13 ms single-image p50 |
| What actually limits the box | the vision loop, at **~330 analysed frames/s** — and it is the GIL, so a worker pool measured *slower* |
| Ring buffer | stream-copy, ~28 MB per camera per 5-minute window, no NVENC |
| Semantic search | **105–129 ms** a query over 14 891 indexed frames; indexing costs **0.3% of one GPU** at 2 cameras, ~4% at twice the frame rate 48 cameras would ask for |

The Phase-1 exit criterion is met, and it moved the plan's estimate: 24–40
cameras was the guess and 48 is the measurement. The 2026-08-12 pass made the
vision loop ~11% faster, which bought margin at 48 (80 ms of lag down to 38)
rather than a bigger fleet — because the constraint is neither NVDEC nor the GPU
but the one interpreter lock every camera's decoder thread competes for. Method,
the full sweep, and the worker pool that was built and then switched off are in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

---

## Detector build order

Detectors are built and validated **one at a time**, not in parallel. The order
is a dependency chain, not a difficulty ranking — each step needs the one before
it to be solid.

### Where each class stands today

Measured on a customer's own footage — six cameras from Anmol Industries'
biscuit plant — rather than on the demo clip. Nothing below `reviewed` sends
anybody an email.

| # | Class | How it is done | Stage | Measured |
|---|---|---|---|---|
| 1 | Person detection + tracking | RF-DETR Nano + ByteTrack | **reviewed** | AP@50 **0.595** on this plant's `val`, presence recall 44.6%. Optional per-site fine-tune: **0.782 / 80.0%** in 8.7 min, `detection.model.checkpoint` — [`docs/DETECTOR_EVAL.md`](docs/DETECTOR_EVAL.md) |
| 2 | Manpower vs. sanctioned limit | zone counts on 1 | **reviewed** | Fires correctly on all six plant scenes; the limits are placeholders until the site supplies a manning sheet |
| 3a | Broken-biscuit trays overflowing | VLM prompt policy | **shadow** | 8B **0/10 recall**; 27B-FP8 **1.000 / 0.800** on the same frames. Needs a model this appliance does not ship — [`docs/VLM_EVAL.md`](docs/VLM_EVAL.md) |
| 3b | Packed biscuits / laminate on the floor | VLM prompt policy | **shadow** | **1.000 / 1.000** on 13 labelled crops incl. 3 cross-camera negatives |
| 3c | Carton-loading backlog at dispatch | VLM prompt policy | **shadow** | 8B precision 1.000, recall 0.333. **Pending site review** |
| 3d | Idle / conversing manpower | VLM prompt policy | **shadow** | Zero false accusations in 9 negatives; missed its 1 positive. **Pending site review** |
| 4 | Hair-net | OWLv2 bootstrap → fine-tune | *labels only* | 2 832 candidate head crops from 139 frames in 21.4 s; the VLM cannot do it as a prompt — [`docs/HAIRNET_BOOTSTRAP.md`](docs/HAIRNET_BOOTSTRAP.md) |
| 5 | Phone usage | VLM prompt policy | **reviewed** (demo fleet) | 0 of 26 labelled frames are positives — the policy has never been shown one |
| 6–8 | Fall, inactivity, jewelry | not started | — | — |

The four plant classes are `stage: shadow`, `lifecycle: pilot` in
[`config/seed.anmol.yaml`](config/seed.anmol.yaml): they record events and route
to nobody. Promotion needs a plant supervisor's judgement on the labels, which
nobody has given yet.

1. **Person detection + tracking** — a stable box on every person, same ID
   across frames, no double-count on movement. Everything else sits on this, and
   "person" needs no custom labelling to get working. *(Built: RF-DETR +
   ByteTrack.)*
2. **Manpower / headcount vs. sanctioned limit** — a configuration layer on top
   of 1: per-zone thresholds, not a new detection problem. *(Built.)*
3. **Floor mishaps and the customer's own classes** — first custom-labelled
   classes. VLM policy first, distilled to a fast detector once labelled data
   accrues. *(Four written and measured, all `shadow` — table above.)*
4. **Hair-net** — runs on the person box. Bootstrap labels with an open-vocab
   model, then fine-tune RF-DETR-Nano on-box. *(Bootstrap built; the VLM was
   measured and cannot do it as a prompt.)*
5. **Phone usage** — runs on the person box; duration comes from tracking.
   *(A prompt policy exists on the demo fleet; no positive has ever been seen.)*
6. **Fall detection** — pose (RTMO) on top of the person box, VLM verification,
   tuned for high recall.
7. **Inactivity / sleep** — track velocity + VLM verification, tuned for high
   precision.
8. **Jewelry** — VLM close-up zone policies.

The open-vocab → distil loop (auto-label with a slow zero-shot model, train a
fast closed-set one on the Spark) is the standing pattern for every new class,
and it is what makes "false positives feed retraining" real rather than
export-only.

**The on-box half of that loop exists for class 1 already.**
`scripts/train_detector.py` fine-tunes RF-DETR Nano on a site's own labelled
frames in its own virtualenv, and `argus detector-eval --checkpoint <path>`
scores the result against the stock weights over the same frames, per scene.
Measured on 63 labelled frames from one plant: `val` presence recall **44.6% →
80.0%** in **8.7 minutes**, for seven points of precision — `docs/DETECTOR_EVAL.md`.
It ships as an **option** (`detection.model.checkpoint`, empty by default),
never as a new default: weights fitted to one plant's cameras are a liability on
another's, and the appliance refuses to load any checkpoint that arrives without
the `provenance.json` saying what it was trained on and which logit is a person.

### Staged rollout

Each detector passes through four stages before the next one starts. This is the
core of the product, and the API enforces it: a policy cannot be created
directly into `auto_routed`.

| Stage | What happens |
|---|---|
| 1. Shadow | Runs silently, logs only, no alerts. Not reviewable — shadow output never enters the precision statistics. |
| 2. Reviewed | Alerts go to a human reviewer, not the HOD. False positives are tagged and feed retraining. |
| 3. Auto-routed | Alerts go live by email to the owning role, once the measured false-positive rate is proven acceptable. |
| 4. Scaled | Expand to more cameras and zones. |

Policies carry a second axis alongside stage — a **lifecycle** of
`draft → pilot → production` — so a policy can be authored, run against live
video for a validation window, and promoted with its analytics on the screen.
Both transitions are timestamped, which is the evidence that a policy served its
window before anyone's phone started buzzing.

---

## Retention

**v1 kept everything and produced 186 GB from a demo loop.** v2 treats bounded
storage as a product requirement, not a maintenance task. Housekeeping runs on a
thread inside `argus run` — no cron, no systemd timer, nothing an installer can
forget:

- **Age windows** (`retention.*_days`): clips 14 d, snapshots 30 d, events
  180 d, occupancy 7 d, search-index samples 14 d. Files are deleted *and* the
  event rows that referenced them are cleared, so the API never serves a path to
  a file that is gone. The vector index is pruned in the same sweep — by age, by
  vanished event, by deleted camera and by deleted thumbnail — because it lives
  in a second SQLite file where no foreign key can reach it, and a search
  result that resolves to nothing is worse than one that was never returned.
- **A disk guard** (`retention.max_disk_percent`, default 85%): reclaims
  oldest-first whenever the filesystem crosses the threshold, because a fixed
  age window is no defence against a shift that simply produces more video than
  usual. It never touches anything younger than `min_keep_hours`, so an alert
  raised minutes ago always still has its clip.

`uv run argus retention --dry-run` reports what is on disk without deleting
anything; `uv run argus retention` runs one housekeeping tick now.

---

## Running as an appliance

An appliance is not a checkout somebody remembers to start. Three commands turn
one into the other, and none of them needs root:

```bash
./scripts/install.sh --services      # install, then write + enable + start the units
argus doctor                         # is this box a working appliance? exit 1 if not
argus backup list                    # the nightly snapshots, newest first
```

**systemd *user* units** (`~/.config/systemd/user/`): `argus-engine.service`,
`argus-api.service`, an optional `argus-vlm.service` that is written but *not*
enabled, and `argus.target` to handle all of them at once. Everything logs to
journald. Written by `argus install-services`, which compares unit text before
writing, refuses to overwrite a file it did not generate, and prints the one
command in the whole install that does need root:

```bash
sudo loginctl enable-linger $USER    # or every service stops when you log out
```

Measured on this box: `systemctl --user start argus.target` brought the engine
and API up together and held **5.06–5.15 fps** against a 5.00 target at 32–51 ms
behind the live edge; `kill -9` on the engine was replaced **5 s** later with
`NRestarts=1`; `systemctl --user stop argus.target` left **no** orphaned ffmpeg,
go2rtc or argus process and a clear GPU.

**Nightly backups** run on the same housekeeping thread as retention — no cron
entry to forget. Both SQLite files are copied with the **online backup API**,
because the vision engine is writing to one of them in WAL mode and a plain `cp`
of that is a database that is *sometimes* fine:

```bash
argus backup now --reason pre-upgrade
argus backup verify latest        # re-hash against the manifest
argus backup restore latest       # engine stopped; nothing is ever deleted
```

Measured live, engine running and writing: **71.8 MB copied and sha256-verified
in 0.1 s**; a drill that deleted all 7 599 events restored them in **0.29 s**,
`quick_check: ok`, with the damaged database kept as `*.pre-restore-<stamp>`.
Backups default to `data/backups` — which survives a mistake, not a dead disk;
point `backup.directory` at a share and `argus doctor` stops warning.

**`argus doctor`** is the support tool: GPU driver *and* whether torch sees it,
ffmpeg's NVDEC *and* its licence, every model checkpoint, schema version and
SQLite `quick_check`, disk against the retention guard, backup age, log sizes,
unit states, linger — and **which ports answer from the LAN**, by opening a
socket rather than reading the config. Every warning and failure carries the
command that fixes it. `--json` for a monitoring probe, `--strict` to fail on
warnings.

---

## Alerting

Email only, and only for events whose stage is `auto_routed`. Slack, generic
webhooks and WhatsApp are deliberately out of scope: a webhook to a cloud
endpoint quietly contradicts the "raw video never leaves site" claim the product
is sold on, and plant SMTP already exists and already reaches the phones that
matter.

Roles map to recipients in `config/argus.yaml` (`smtp.recipients`), a policy is
routed to a role, and **every attempt is recorded** in the `notifications`
table — sent or failed, one row per recipient. "Was anyone actually told?" is
answerable from the database alone, months later. The annotated snapshot is
attached to the message.

---

## Repository layout

```
argus/              # v2 — the product
  api/              #   FastAPI app, routers, request schemas
  ingest/           #   go2rtc gateway supervision, ffmpeg decoders
  vision/           #   RF-DETR detector, ByteTrack, zones, zone state, debounce
  media/            #   ring buffer, clip cutting, annotated snapshots
  vlm/              #   vLLM server supervision, policy scheduler, JSON schemas
  index/            #   SigLIP 2 embedder, sqlite-vec store, the writer, backfill
  auth.py accounts.py   #   roles, Argon2id, session tokens; account rules
  services.py backup.py doctor.py   #   systemd units; SQLite snapshots + log
                    #   rotation; the one command support asks a site to run
  config.py db.py engine.py health.py notify.py retention.py seed.py cli.py
ui/                 # the operator console (Vite + React + TS), built to ui/dist
  src/screens/      #   the nine screens, one file each
  src/components/   #   camera tiles, overlays, evidence viewer, ⌘K, Wall TV
  src/api/          #   typed client for the JSON API and the gateway relay
  src/state/        #   the session (who is signed in) and the shared app state
  design-reference/ #   the approved Claude Design mockup this implements
config/             # argus.yaml (deployment config), seed.yaml (the fleet)
docs/               # REBUILD_PLAN.md, INSTALL.md, BENCHMARKS.md, SECURITY.md,
                    # PHASE2_NOTES.md, PHASE3_NOTES.md, LICENSES.md,
                    # DETECTOR_EVAL.md, PRODUCTION_READINESS.md, design/
scripts/            # install.sh (orchestrates the rest), fetch_gateway.sh,
                    # build_ffmpeg.sh, fetch_detector.sh, fetch_siglip.sh,
                    # fetch_vlm.sh, setup_vlm_server.sh, benchmark.py,
                    # license_audit.py, setup_detector_training.sh +
                    # train_detector.py (optional per-site detector fine-tune)
tests/              # pytest; GPU-optional, no network
```

v1 is on the **`v1-archive`** branch (and on the GitHub remote), not in this
tree. `git show v1-archive:<path>` reads any of it without a checkout.

## Licensing

Every runtime dependency is **MIT, BSD, Apache-2.0, PSF or ISC**. This is a hard
rule, not a preference: v1 depended on Ultralytics (AGPL-3.0), which is
incompatible with shipping this as a commercial product, and replacing it with
Apache-licensed RF-DETR was one of the five reasons for the rebuild. **Both
`ultralytics` and v1 are now gone from this branch.**

The rule is checkable rather than asserted. [`docs/LICENSES.md`](docs/LICENSES.md)
is an audit of the *resolved* trees — every package actually installed, in the
appliance venv, in `ui/node_modules` and in the VLM serving venv — regenerated
from the metadata already on disk, with no network and no extra tool:

```bash
uv run python scripts/license_audit.py            # rewrite docs/LICENSES.md
uv run python scripts/license_audit.py --check    # exit 1 if copyleft appears
```

Three things it reports that a one-line claim would hide, all covered in full
there: `certifi` and `tqdm` are MPL-2.0 (file-level copyleft — combining is
allowed, publishing *their* source with the appliance image is owed); the CUDA
runtime libraries torch depends on are NVIDIA proprietary and redistributable
under the CUDA EULA; and `bin/ffmpeg` is LGPL by construction, which is the whole
reason `scripts/build_ffmpeg.sh` exists. Before adding a dependency, check its
licence — then run `--check`.

## Contributing

- Before adding a detector class, check it against the build order above — don't
  jump ahead.
- Any model or library choice needs a note in `docs/` with the reasoning, not
  just a PR with a new dependency. Include its licence, and re-run
  `scripts/license_audit.py`.
- A labelled dataset ships with a dataset card next to it (source clips, class
  list, frame count, date) — provenance is what makes a retraining export
  usable a year later.
- Alert-routing changes need a note on which role or threshold changed and why.
- No facial-identity or biometric matching code without sign-off — legal review
  required. Workplace emotion recognition is prohibited in the EU (Feb 2025) and
  is out of scope everywhere.

**Not in scope:** production/yield verification (needs PLC integration) and
facial identity / access control.

## License / access

Internal — Anmol Industries.
