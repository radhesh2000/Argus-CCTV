"""The eval harness, tested without a GPU, without weights and without a network.

Both runners are exercised against **real HTTP servers on loopback** rather than
mocked clients, for the reason ``tests/test_vlm.py`` gives and this package
doubles: the thing most likely to break is the wire contract, and a mocked
client cannot fail that way. The cloud runner gets its own fake, standing in for
OpenRouter with a catalog endpoint that publishes prices — because the cost
column in the report is derived from that shape, and a hard-coded price would
make the test pass while the report lies.

The most important test in this file is not about a score. It is
``test_the_runtime_tree_never_imports_the_cloud_runner``: it starts a fresh
interpreter, imports the engine, the API and the VLM scheduler, and asserts that
nothing named ``argus.eval`` came with them. That is the "no cloud calls in the
product" claim expressed as an assertion instead of a promise.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from argus.config import ArgusConfig
from argus.db import Camera, Database, Event, Policy
from argus.eval.cases import EvalCase, EvalSet, FrameRef
from argus.eval.dataset import (
    COUNT_STRICT_POLICY,
    COUNT_VERIFY_POLICY,
    build_eval_set,
    counting_claims,
    headcount_claim,
    policy_applies,
    strict_verification_prompt,
)
from argus.eval.frames import MAX_SCAN_DEPTH, find_videos
from argus.eval.harness import (
    EvalRun, build_labels, cloud_gate, load_baseline, run_eval, write_json,
)
from argus.eval.labels import blank_frame_entry, counts, load, merge, save
from argus.eval.metrics import Confusion, Latency, disagreements, score
from argus.eval.report import EXCLUDED_CANDIDATES, render
from argus.eval.runners import LocalRunner, RunnerResult, run_cases
from argus.vlm.schema import VlmResponseError, verification_prompt

MODEL = "test/eval-vlm"
CLOUD_MODEL = "vendor/cloud-vlm"


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------


class Reply:
    """A provider reply spelled out at the wire level, not at the verdict level.

    The cloud candidates fail in ways a verdict dict cannot express — a null
    ``content`` with the answer in ``reasoning``, a completion cut off by the
    token budget, a 400 that names the parameter it objects to. Those are the
    shapes ``argus/eval/openrouter.py`` exists to survive, so the fake has to be
    able to produce them verbatim.
    """

    def __init__(self, *, message: dict | None = None, finish_reason: str = "stop",
                 usage: dict | None = None, status: int = 200, error: str = ""):
        self.message = {"role": "assistant", **(message or {})}
        self.finish_reason = finish_reason
        self.usage = usage or {"prompt_tokens": 1000, "completion_tokens": 50}
        self.status = status
        self.error = error


class FakeOpenAIServer:
    """An OpenAI-compatible endpoint that answers from a scripted queue.

    ``replies`` entries are verdict dicts, raw strings (to test a model that
    wraps its JSON in prose), ints (an HTTP status to fail with), or
    :class:`Reply` objects (the whole message, spelled out). The last entry
    repeats, so a test that does not care how many calls happen need not count
    them.

    ``pricing`` turns the ``/models`` endpoint into an OpenRouter-shaped catalog
    with per-token prices and input modalities.
    """

    def __init__(self, replies=None, *, model: str = MODEL, pricing: dict | None = None,
                 modalities: list[str] | None = None, demands_reasoning: bool = False):
        self.replies = list(replies or [
            {"triggered": False, "confidence": 0.9, "reason": "nothing here", "box_2d": None}
        ])
        self.model = model
        self.pricing = pricing
        self.modalities = modalities if modalities is not None else ["text", "image"]
        # google/gemini-3.7-flash and qwen/qwen3.8-max both answer 400 to
        # `reasoning: {"effort": "none"}`. Modelling it as a rule rather than a
        # scripted reply is what makes the concurrent-degrade test deterministic:
        # the refusal depends on what was asked, not on how many calls preceded it.
        self.demands_reasoning = demands_reasoning
        self.requests: list[dict] = []
        self.auth: list[str] = []
        self.status = 200
        self._index = 0
        self._lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def _json(self, payload, status=200):
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if server.status != 200:
                    self._json({"error": "down"}, status=server.status)
                    return
                if not self.path.endswith("/models"):
                    self._json({"error": "not found"}, status=404)
                    return
                entry = {"id": server.model,
                         "architecture": {"input_modalities": list(server.modalities)}}
                if server.pricing is not None:
                    entry["pricing"] = server.pricing
                self._json({"object": "list", "data": [entry]})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                with server._lock:
                    server.requests.append(payload)
                    server.auth.append(self.headers.get("Authorization", ""))
                    if server.demands_reasoning and \
                            (payload.get("reasoning") or {}).get("effort") == "none":
                        self._json({"error": {
                            "message": "Reasoning is mandatory for this endpoint and "
                                       "cannot be disabled.", "code": 400}}, status=400)
                        return
                    reply = server._next()
                if isinstance(reply, int):
                    self._json({"error": {"message": "scripted failure"}}, status=reply)
                    return
                if isinstance(reply, Reply):
                    if reply.status != 200:
                        self._json({"error": {"message": reply.error, "code": reply.status}},
                                   status=reply.status)
                        return
                    self._json({
                        "id": "chatcmpl-eval",
                        "model": server.model,
                        "choices": [{"index": 0, "message": reply.message,
                                     "finish_reason": reply.finish_reason}],
                        "usage": reply.usage,
                    })
                    return
                content = reply if isinstance(reply, str) else json.dumps(reply)
                self._json({
                    "id": "chatcmpl-eval",
                    "model": server.model,
                    "choices": [{"index": 0,
                                 "message": {"role": "assistant", "content": content},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1000, "completion_tokens": 50},
                })

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def _next(self):
        reply = self.replies[min(self._index, len(self.replies) - 1)]
        self._index += 1
        return reply

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    @property
    def calls(self) -> int:
        return len(self.requests)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def server():
    with FakeOpenAIServer() as running:
        yield running


def write_frame(path: Path, *, width: int = 320, height: int = 240, shade: int = 40) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), (shade, shade + 10, shade + 20)).save(path, "JPEG")
    return path


def make_case(path: Path, *, expected: bool | None = True, policy: str = "p",
              slice_name: str = "policy", case_id: str = "c1", **frame_kwargs) -> EvalCase:
    return EvalCase(
        id=case_id, slice=slice_name, policy=policy, prompt="is anything wrong?",
        frame=FrameRef(path=path, origin="test", **frame_kwargs), expected=expected,
    )


# --------------------------------------------------------------------------
# frames and cases
# --------------------------------------------------------------------------


def test_a_frame_loads_as_bgr(tmp_path: Path):
    path = write_frame(tmp_path / "f.jpg")
    image = FrameRef(path=path).load()
    assert image.shape == (240, 320, 3)
    assert image.dtype == np.uint8
    # Written as (40, 50, 60) RGB, so BGR order puts the largest channel first.
    assert image[0, 0, 0] > image[0, 0, 2]


def test_a_region_crops_the_frame(tmp_path: Path):
    path = write_frame(tmp_path / "f.jpg", width=400, height=200)
    full = FrameRef(path=path).load()
    cropped = FrameRef(path=path, region=(0.0, 0.5, 0.5, 1.0)).load()
    assert full.shape == (200, 400, 3)
    assert cropped.shape == (100, 200, 3)


def test_a_reversed_or_out_of_range_region_is_clamped_not_rejected(tmp_path: Path):
    path = write_frame(tmp_path / "f.jpg", width=400, height=200)
    # Corners the wrong way round and running off the edge: a hand-written crop
    # in a YAML file will be wrong like this, and losing the frame over it is
    # worse than clamping.
    image = FrameRef(path=path, region=(0.9, 1.4, 0.2, -0.3)).load()
    assert image.shape[0] > 0 and image.shape[1] > 0


def test_cropping_the_banner_removes_the_headline_strip(tmp_path: Path):
    from argus.media.annotate import banner_height, draw_banner

    picture = Image.new("RGB", (640, 360), (30, 30, 30))
    draw_banner(picture, "Zone 'Packing floor' over sanctioned limit: 5 people (limit 3)")
    path = tmp_path / "snap.jpg"
    picture.save(path, "JPEG")

    kept = FrameRef(path=path, crop_banner=False).load()
    cropped = FrameRef(path=path, crop_banner=True).load()
    assert kept.shape[0] == 360
    assert cropped.shape[0] == 360 - banner_height(640)


def test_unlabelled_cases_are_counted_but_not_scored(tmp_path: Path):
    path = write_frame(tmp_path / "f.jpg")
    eval_set = EvalSet(cases=[
        make_case(path, expected=True, case_id="a"),
        make_case(path, expected=None, case_id="b"),
    ])
    assert len(eval_set) == 2
    assert len(eval_set.scored) == 1
    assert eval_set.summary()["policy"] == {"cases": 2, "labelled": 1}


# --------------------------------------------------------------------------
# the counting slice
# --------------------------------------------------------------------------


def test_the_counting_slice_reproduces_the_phase_2_failure():
    """Five people, a claim of thirty: the exact case that keeps verify off."""
    variants = dict((name, (count, limit, expected))
                    for name, count, limit, expected in counting_claims(5, exact=True))
    assert variants["truthful"] == (5, 4, True)
    inflated_count, _limit, expected = variants["inflated"]
    assert inflated_count >= 30 and expected is False


def test_the_moderate_variant_probes_near_the_boundary_not_past_it():
    """Measured on this box, the model vetoes a 6x over-claim on a crowded frame
    and upholds a 2.5x one. A slice that only asks 6x reports a pass."""
    variants = dict((name, count) for name, count, _limit, _e in counting_claims(16))
    assert variants["moderate"] < variants["inflated"]
    # Twice the most people who could be there: 16 labelled approximately means
    # up to 19, so 38 — not 32, which a generous labelling error could excuse.
    assert variants["moderate"] == 38


def test_every_wrong_count_claim_is_measured_from_the_generous_end_of_the_label():
    approximate = dict((name, count) for name, count, _l, _e in counting_claims(9))
    exact = dict((name, count) for name, count, _l, _e in counting_claims(9, exact=True))
    assert approximate["moderate"] > exact["moderate"]
    assert approximate["inflated"] > exact["inflated"]


def test_a_truthful_claim_is_only_generated_from_an_exact_count():
    """An off-by-one label would silently invert this one variant, so it is the
    only one gated on the labeller saying they are sure."""
    assert "truthful" not in dict.fromkeys(v[0] for v in counting_claims(9, exact=False))
    assert "truthful" in [v[0] for v in counting_claims(9, exact=True)]
    # The wrong-count variants survive either way — they are wrong by a margin
    # no labelling error covers.
    assert {"inflated", "understated"} <= {v[0] for v in counting_claims(9, exact=False)}


def test_wrong_count_claims_stay_wrong_under_a_three_person_labelling_error():
    for truth in (2, 5, 9, 16):
        for name, count, _limit, expected in counting_claims(truth):
            if name == "truthful":
                continue
            assert expected is False
            # Wrong against the most generous reading of an approximate label.
            assert count > truth + 3 or count < truth - 3, (truth, name, count)


def test_an_empty_frame_only_produces_a_phantom_claim():
    assert [v[0] for v in counting_claims(0, exact=True)] == ["phantom"]


def test_the_generated_claim_matches_the_engines_own_wording():
    """The claim text is what `Engine._raise_occupancy_event` writes verbatim; a
    paraphrase would evaluate a sentence the appliance never emits."""
    assert headcount_claim(5, 3).endswith("over sanctioned limit: 5 people (limit 3)")


def test_the_production_wording_is_the_count_aware_wording_that_was_measured():
    """These two were the A/B: production, and production plus the count clause.
    The clause won 24/24 to 15/24 and moved into production on 2026-08-15, so
    the two are now the same question — and this is what keeps them that way.

    ``strict_verification_prompt`` is deliberately a frozen literal rather than a
    call to the production function. Delegating would make the counting slice
    track whatever the prompt happens to say, and the slice exists to notice when
    that changes: the day these diverge, `headcount_verify` and `headcount_strict`
    separate in docs/VLM_EVAL.md, on the same page as the number that paid for
    the clause."""
    claim = headcount_claim(5, 3)
    production, strict = verification_prompt(claim), strict_verification_prompt(claim)

    assert claim in production
    assert production == strict
    assert "count" in production.lower()


# --------------------------------------------------------------------------
# sampling frames out of video
# --------------------------------------------------------------------------


def test_sampling_starts_half_an_interval_in():
    """The first frame of a clip is often a fade-in or a black frame, and a
    black frame is a case nobody can label."""
    from argus.eval.frames import sample_offsets

    assert sample_offsets(60.0, 20.0, 12) == [10.0, 30.0, 50.0]
    assert sample_offsets(60.0, 20.0, 2) == [10.0, 30.0]
    assert sample_offsets(0.0, 20.0, 12) == []
    assert sample_offsets(60.0, 0.0, 12) == []


def test_a_frame_id_survives_a_rebuild_and_a_working_directory():
    from argus.eval.frames import frame_id

    identifier = frame_id(Path("/somewhere/else/packing_section_recording.mp4"), 22.5)
    assert identifier == frame_id(Path("packing_section_recording.mp4"), 22.5)
    assert identifier == "packing-section-recording-0002250"
    assert "/" not in identifier and " " not in identifier


def test_a_labelled_frame_is_re_cut_from_its_recipe(tmp_path: Path, sample_video, ffmpeg):
    """`evals/frames/` is gitignored, so this is what every fresh checkout does:
    the labels name a source video and an offset, and the picture comes back."""
    from argus.eval.frames import cut_frame, ensure_frame

    frames_dir = tmp_path / "frames"
    entry = {"id": "clip-0001000", "source": sample_video.name, "ts_sec": 10.0}
    first = cut_frame(sample_video, 10.0, frames_dir / "clip-0001000.jpg")
    digest = FrameRef(path=first).digest()

    first.unlink()
    again = ensure_frame(entry, frames_dir, sample_video.parent)
    assert again is not None and again.is_file()
    # Byte-identical, which is what makes the digest in labels.yaml an audit of
    # the *source* rather than of the extraction.
    assert FrameRef(path=again).digest() == digest


def test_a_missing_source_video_returns_none_rather_than_raising(tmp_path: Path):
    from argus.eval.frames import ensure_frame

    entry = {"id": "gone", "source": "footage/not-here.mp4", "ts_sec": 1.0}
    assert ensure_frame(entry, tmp_path / "frames", tmp_path) is None


# --------------------------------------------------------------------------
# the label file
# --------------------------------------------------------------------------


def test_a_starter_label_file_has_null_labels(tmp_path: Path):
    entry = blank_frame_entry("f-1", "clip.mp4", 2.5, ["exit", "phone"])
    assert entry["people"] is None
    assert entry["labels"] == {"exit": None, "phone": None}
    assert entry["labelled_by"] is None


def test_rebuilding_labels_never_overwrites_a_human_answer(tmp_path: Path):
    path = tmp_path / "labels.yaml"
    save(path, {"version": 1, "policies": {"exit": {"label": "Exit", "prompt": "old"}},
                "frames": [{"id": "f-1", "source": "clip.mp4", "ts_sec": 2.5,
                            "people": 4, "people_exact": True, "labelled_by": "aisha",
                            "labels": {"exit": True}}]})

    merged, added = merge(
        load(path),
        [blank_frame_entry("f-1", "clip.mp4", 2.5, ["exit"]),
         blank_frame_entry("f-2", "clip.mp4", 7.5, ["exit"])],
        {"exit": {"label": "Exit", "prompt": "REWRITTEN"}},
    )
    assert added == 1
    kept = {entry["id"]: entry for entry in merged["frames"]}
    assert kept["f-1"]["people"] == 4
    assert kept["f-1"]["labels"]["exit"] is True
    assert kept["f-1"]["labelled_by"] == "aisha"
    assert kept["f-2"]["labels"]["exit"] is None
    # A prompt somebody edited by hand survives too.
    assert merged["policies"]["exit"]["prompt"] == "old"


def test_a_new_policy_reaches_frames_that_predate_it(tmp_path: Path):
    merged, _ = merge(
        {"version": 1, "policies": {}, "frames": [
            {"id": "f-1", "labels": {"exit": True}}]},
        [],
        {"exit": {"label": "Exit", "prompt": "p"}, "phone": {"label": "Phone", "prompt": "q"}},
    )
    labels = merged["frames"][0]["labels"]
    assert labels["exit"] is True and labels["phone"] is None


def test_two_frames_sharing_a_crop_do_not_become_a_yaml_alias(tmp_path: Path):
    """PyYAML emits `&id001` / `*id001` for a repeated object, which in a
    hand-edited file means changing one frame's crop silently changes another's."""
    path = tmp_path / "labels.yaml"
    shared = [0.0, 0.55, 0.4, 1.0]
    save(path, {"version": 1, "policies": {}, "frames": [
        {"id": "f-1", "crop": shared, "people": 2},
        {"id": "f-2", "crop": shared, "people": 3},
    ]})
    text = path.read_text()
    assert "*id" not in text and "&id" not in text
    # And inline, so a crop rectangle reads as one value rather than five lines.
    assert text.count("crop: [0.0, 0.55, 0.4, 1.0]") == 2
    frames = {entry["id"]: entry for entry in load(path)["frames"]}
    assert frames["f-1"]["crop"] == frames["f-2"]["crop"] == shared


def test_the_label_file_round_trips_through_disk(tmp_path: Path):
    path = tmp_path / "labels.yaml"
    document = {"version": 1,
                "policies": {"exit": {"label": "Exit", "prompt": "a long " * 40}},
                "frames": [blank_frame_entry("f-1", "clip.mp4", 1.0, ["exit"])]}
    save(path, document)
    assert path.read_text().startswith("# Argus eval labels")
    assert load(path) == document
    assert counts(load(path))["frames"] == 1


# --------------------------------------------------------------------------
# building the set from a database
# --------------------------------------------------------------------------


@pytest.fixture
def seeded(config: ArgusConfig, database: Database, tmp_path: Path) -> ArgusConfig:
    """Two reviewed events with snapshots, and an empty label file."""
    database.upsert_camera(Camera(id="cam", name="Cam", source="/dev/null", kind="file"))
    database.upsert_policy(Policy(id="manpower_headcount", kind="detector",
                                  detector_key="manpower_headcount", label="Headcount"))
    for index, (verdict, description) in enumerate((
        ("confirmed", "Zone 'Packing floor' over sanctioned limit: 5 people (limit 3)"),
        ("false_positive", "Zone 'Packing floor' over sanctioned limit: 4 people (limit 3)"),
    )):
        name = f"snap-{index}.jpg"
        write_frame(config.snapshots_dir / name)
        event_id = database.insert_event(Event(
            camera_id="cam", policy_id="manpower_headcount",
            detector_name="manpower_headcount", detector_type="detector",
            severity="medium", stage="reviewed", description=description,
            ts=1000.0 + index, snapshot_path=name,
        ))
        database.record_review(event_id, verdict, reviewer="aisha")
    config.eval.labels_path = tmp_path / "labels.yaml"
    config.eval.frames_dir = tmp_path / "frames"
    return config


def test_triage_verdicts_become_expected_verdicts(seeded: ArgusConfig, database: Database):
    eval_set = build_eval_set(seeded, database, slices=("triage",))
    expected = {case.id: case.expected for case in eval_set.cases}
    assert sorted(expected.values(), key=str) == [False, True]
    case = eval_set.cases[0]
    # The production prompt, imported rather than retyped.
    assert case.prompt == verification_prompt(case.note)
    assert case.label_source.startswith("triage verdict")


def test_the_triage_slice_keeps_the_banner_because_the_appliance_does(
    seeded: ArgusConfig, database: Database,
):
    """`VlmRunner.dispatch` hands the verifier the annotated frame. Cropping it
    here would score a model on an input the product never sends."""
    eval_set = build_eval_set(seeded, database, slices=("triage",))
    assert all(case.frame.crop_banner is False for case in eval_set.cases)


def test_crop_banner_can_be_forced_over_the_whole_set(seeded: ArgusConfig, database: Database):
    eval_set = build_eval_set(seeded, database, slices=("triage",), crop_banner=True)
    assert all(case.frame.crop_banner for case in eval_set.cases)
    assert any("banner crop forced on" in note for note in eval_set.notes)


def test_an_event_without_a_snapshot_on_disk_is_skipped(
    seeded: ArgusConfig, database: Database,
):
    (seeded.snapshots_dir / "snap-0.jpg").unlink()
    eval_set = build_eval_set(seeded, database, slices=("triage",))
    assert len(eval_set) == 1
    assert any("without a snapshot" in note for note in eval_set.notes)


def test_labelled_frames_build_policy_and_counting_cases(
    seeded: ArgusConfig, database: Database, tmp_path: Path,
):
    frame = write_frame(tmp_path / "frames" / "f-1.jpg")
    save(seeded.eval_labels_path, {
        "version": 1,
        "policies": {"exit": {"label": "Exit", "prompt": "is the exit blocked?"}},
        "frames": [{"id": "f-1", "source": str(frame), "ts_sec": 0.0,
                    "people": 5, "people_exact": True, "labelled_by": "aisha",
                    "labels": {"exit": False}}],
    })
    eval_set = build_eval_set(seeded, database, slices=("policy", "counting"))
    policies = {case.policy for case in eval_set.cases}
    assert policies == {"exit", COUNT_VERIFY_POLICY, COUNT_STRICT_POLICY}
    counting = eval_set.by_slice("counting")
    # Every counting claim is asked under both prompts.
    assert len(counting) == 2 * len(counting_claims(5, exact=True))
    assert all(case.frame.crop_banner is False for case in counting)


def test_a_frame_whose_source_video_is_gone_is_dropped_with_a_note(
    seeded: ArgusConfig, database: Database,
):
    save(seeded.eval_labels_path, {
        "version": 1, "policies": {"exit": {"label": "Exit", "prompt": "p"}},
        "frames": [{"id": "missing", "source": "footage/not-here.mp4", "ts_sec": 1.0,
                    "people": 3, "labels": {"exit": True}}],
    })
    eval_set = build_eval_set(seeded, database, slices=("policy", "counting"))
    assert len(eval_set) == 0
    assert any("skipped" in note for note in eval_set.notes)


def test_a_counting_slice_without_a_positive_control_says_so(
    seeded: ArgusConfig, database: Database, tmp_path: Path,
):
    frame = write_frame(tmp_path / "frames" / "f-1.jpg")
    save(seeded.eval_labels_path, {
        "version": 1, "policies": {},
        "frames": [{"id": "f-1", "source": str(frame), "ts_sec": 0.0,
                    "people": 9, "people_exact": False, "labels": {}}],
    })
    eval_set = build_eval_set(seeded, database, slices=("counting",))
    assert all(case.expected is False for case in eval_set.cases)
    assert any("no positive control" in note for note in eval_set.notes)


def test_an_unreadable_headcount_costs_one_frame_not_the_run(
    seeded: ArgusConfig, database: Database, tmp_path: Path,
):
    frames = [write_frame(tmp_path / "frames" / f"f-{i}.jpg") for i in (1, 2)]
    save(seeded.eval_labels_path, {
        "version": 1, "policies": {},
        "frames": [
            {"id": "f-1", "source": str(frames[0]), "ts_sec": 0.0, "people": "five",
             "labels": {}},
            {"id": "f-2", "source": str(frames[1]), "ts_sec": 0.0, "people": 6,
             "people_exact": True, "labels": {}},
        ],
    })
    eval_set = build_eval_set(seeded, database, slices=("counting",))
    assert eval_set.cases and all("f-2" in case.id for case in eval_set.cases)
    assert any("unreadable `people:`" in note for note in eval_set.notes)


def test_limit_keeps_the_labelled_cases(seeded: ArgusConfig, database: Database, tmp_path: Path):
    frame = write_frame(tmp_path / "frames" / "f-1.jpg")
    save(seeded.eval_labels_path, {
        "version": 1,
        "policies": {"exit": {"label": "Exit", "prompt": "p"},
                     "phone": {"label": "Phone", "prompt": "q"}},
        "frames": [{"id": "f-1", "source": str(frame), "ts_sec": 0.0, "people": None,
                    "labels": {"exit": True, "phone": None}}],
    })
    eval_set = build_eval_set(seeded, database, slices=("policy",), limit=1)
    assert len(eval_set) == 1
    assert eval_set.cases[0].scored


# --------------------------------------------------------------------------
# the local runner
# --------------------------------------------------------------------------


def test_the_local_runner_speaks_the_production_wire_shape(
    config: ArgusConfig, server: FakeOpenAIServer, tmp_path: Path,
):
    path = write_frame(tmp_path / "f.jpg")
    runner = LocalRunner(name=MODEL, base_url=server.base_url)
    result = asyncio.run(runner.evaluate(make_case(path)))

    assert result.triggered is False and result.error is None
    payload = server.requests[0]
    assert payload["model"] == MODEL
    assert payload["response_format"]["json_schema"]["name"] == "argus_policy_verdict"
    content = payload["messages"][0]["content"]
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "single JSON object" in content[0]["text"]


def test_an_unreachable_model_becomes_a_result_not_an_exception(
    server: FakeOpenAIServer, tmp_path: Path,
):
    server.replies = [503]
    path = write_frame(tmp_path / "f.jpg")
    runner = LocalRunner(name=MODEL, base_url=server.base_url)
    result = asyncio.run(runner.evaluate(make_case(path)))
    assert result.triggered is None
    assert "VlmUnavailable" in (result.error or "")
    assert result.latency_sec >= 0.0


def test_an_unparseable_answer_becomes_a_result_not_an_exception(
    server: FakeOpenAIServer, tmp_path: Path,
):
    server.replies = ["I would rather not answer that."]
    path = write_frame(tmp_path / "f.jpg")
    runner = LocalRunner(name=MODEL, base_url=server.base_url)
    result = asyncio.run(runner.evaluate(make_case(path)))
    assert result.triggered is None and "VlmResponseError" in (result.error or "")


def test_run_cases_preserves_order_under_concurrency(
    server: FakeOpenAIServer, tmp_path: Path,
):
    path = write_frame(tmp_path / "f.jpg")
    cases = [make_case(path, case_id=f"c{index}") for index in range(6)]
    runner = LocalRunner(name=MODEL, base_url=server.base_url, max_concurrency=3)
    results = asyncio.run(run_cases(runner, cases, concurrency=3, log=lambda *_: None))
    assert [result.case_id for result in results] == [case.id for case in cases]


# --------------------------------------------------------------------------
# the cloud runner and its gate
# --------------------------------------------------------------------------


def test_the_cloud_gate_needs_both_the_flag_and_the_key(config: ArgusConfig):
    variable = config.eval.cloud_key_env
    allowed, reason = cloud_gate(config, cloud=False, env={})
    assert allowed is False and "--cloud was not passed" in reason and variable in reason

    allowed, reason = cloud_gate(config, cloud=False, env={variable: "sk-x"})
    assert allowed is False and "a key alone is not consent" in reason

    allowed, reason = cloud_gate(config, cloud=True, env={})
    assert allowed is False and "is not set" in reason

    allowed, reason = cloud_gate(config, cloud=True, env={variable: "sk-x"})
    assert allowed is True


def test_the_gate_message_is_one_line(config: ArgusConfig):
    for cloud, env in ((False, {}), (False, {"OPENROUTER_API_KEY": "sk"}), (True, {})):
        _allowed, reason = cloud_gate(config, cloud=cloud, env=env)
        assert "\n" not in reason and reason


def test_the_cloud_runner_refuses_to_exist_without_a_key(monkeypatch):
    from argus.eval.openrouter import CloudDisabled, OpenRouterRunner

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(CloudDisabled):
        OpenRouterRunner(name=CLOUD_MODEL)


def test_the_cloud_runner_sends_the_key_and_reads_the_verdict(monkeypatch, tmp_path: Path):
    from argus.eval.openrouter import OpenRouterRunner

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-value")
    with FakeOpenAIServer(
        replies=[{"triggered": True, "confidence": 0.8, "reason": "a person", "box_2d": None}],
        model=CLOUD_MODEL,
    ) as cloud:
        runner = OpenRouterRunner(name=CLOUD_MODEL, base_url=cloud.base_url)
        result = asyncio.run(runner.evaluate(make_case(write_frame(tmp_path / "f.jpg"))))

    assert result.triggered is True and result.confidence == pytest.approx(0.8)
    assert result.prompt_tokens == 1000 and result.completion_tokens == 50
    assert cloud.auth[0] == "Bearer sk-test-value"


def test_a_cloud_model_that_wraps_its_json_in_a_fence_is_still_read(monkeypatch, tmp_path: Path):
    """OpenRouter passes `response_format` through only to providers that
    support it, so prose-wrapped JSON is a normal reply, not a broken one."""
    from argus.eval.openrouter import OpenRouterRunner

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-value")
    fenced = '```json\n{"triggered": true, "confidence": 0.7, "reason": "yes", "box_2d": null}\n```'
    with FakeOpenAIServer(replies=[fenced], model=CLOUD_MODEL) as cloud:
        runner = OpenRouterRunner(name=CLOUD_MODEL, base_url=cloud.base_url)
        result = asyncio.run(runner.evaluate(make_case(write_frame(tmp_path / "f.jpg"))))
    assert result.triggered is True and result.error is None


# --------------------------------------------------------------------------
# reading what four vendors actually send back
#
# The strings below are shortened but otherwise verbatim from OpenRouter on
# 2026-08-25, one per failure mode that cost the first cloud run 387-420 of its
# 424 cases. Each is a payload no local vLLM reply can be, which is why the
# leniency lives in the eval module and `argus/vlm/schema.py` stays strict.
# --------------------------------------------------------------------------


FENCED = (
    '```json\n{"triggered": true, "confidence": 0.88, "reason": "Crushed biscuit '
    'waste is visible in a tray cart.", "box_2d": [566, 322, 755, 447]}\n```'
)
IN_PROSE = (
    "Looking at the frame I can see product on the line. Here is my assessment:\n\n"
    '{"triggered": false, "confidence": 0.9, "reason": "Product appears intact.", '
    '"box_2d": null}\n\nLet me know if you need anything else.'
)
# qwen/qwen3.8-max and moonshotai/kimi-k3 both do this: content is null and the
# whole attempt is in `reasoning`.
REASONED = {
    "role": "assistant",
    "content": None,
    "reasoning": (
        "The user wants me to judge whether broken biscuits are visible. The trays "
        "look intact and the instruction says prefer false when ambiguous. So:\n"
        '{"triggered": false, "confidence": 0.8, "reason": "Biscuits in trays appear '
        'intact.", "box_2d": null}'
    ),
}
# Cut off by the token budget mid-`reason`. The verdict is already on the wire.
TRUNCATED = (
    '{"triggered": true, "confidence": 0.72, "reason": "Spilled biscuit pieces are '
    "visible scattered on the floor beside the conveyo"
)
GARBAGE = "I'm sorry, I can't help with analysing images of people at work."


@pytest.mark.parametrize("text, triggered", [
    (FENCED, True),
    (IN_PROSE, False),
    (REASONED["reasoning"], False),
    (TRUNCATED, True),
])
def test_every_shape_a_provider_sends_is_read_back_to_a_verdict(text: str, triggered: bool):
    from argus.eval.openrouter import extract_verdict_json

    payload = json.loads(extract_verdict_json(text))
    assert payload["triggered"] is triggered


def test_prose_after_the_json_does_not_cost_the_reply_its_last_field():
    """The salvage must not answer "extra data" by deleting a real field.

    A model that signs off after its JSON has produced a complete verdict.
    Trimming back to the last comma to make `json.loads` stop complaining would
    silently drop `box_2d` from a reply that had nothing wrong with it, so the
    object is read with `raw_decode` and the sign-off ignored.
    """
    from argus.eval.openrouter import extract_verdict_json

    text = (
        '{"triggered": true, "confidence": 0.7, "reason": "a phone", '
        '"box_2d": [10, 20, 30, 40]}\n\nLet me know, if you need anything else.'
    )
    payload = json.loads(extract_verdict_json(text))
    assert payload["box_2d"] == [10, 20, 30, 40]
    assert payload["confidence"] == 0.7


def test_a_refusal_is_an_error_and_not_a_negative_verdict():
    """The failure that matters: garbage must not score as `triggered: false`."""
    from argus.eval.openrouter import extract_verdict_json

    with pytest.raises(VlmResponseError):
        extract_verdict_json(GARBAGE)


def test_a_truncation_that_never_reached_the_verdict_is_an_error():
    """`{}` handed to `parse_verdict` reads as "the model said no". It did not.

    This is the one place the salvage has to say no. A reply cut off before
    `triggered` contains no answer, and inventing `false` for it would move the
    model's recall without the model having done anything.
    """
    from argus.eval.openrouter import extract_verdict_json

    for fragment in ('{', '{"confi', '{"confidence": 0.9, "reason": "some'):
        with pytest.raises(VlmResponseError):
            extract_verdict_json(fragment)


def test_the_verdict_is_found_past_an_earlier_object_in_the_reasoning():
    """Reasoning text quotes the requested shape before answering in it."""
    from argus.eval.openrouter import extract_verdict_json

    text = (
        'I must answer with {"format": "as specified"} exactly. Checking the frame '
        'now.\n{"triggered": true, "confidence": 0.6, "reason": "a phone", "box_2d": null}'
    )
    assert json.loads(extract_verdict_json(text))["reason"] == "a phone"


def test_an_answer_left_in_the_reasoning_field_is_still_an_answer(monkeypatch, tmp_path: Path):
    from argus.eval.openrouter import OpenRouterRunner

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-value")
    with FakeOpenAIServer(replies=[Reply(message=REASONED)], model=CLOUD_MODEL) as cloud:
        runner = OpenRouterRunner(name=CLOUD_MODEL, base_url=cloud.base_url)
        result = asyncio.run(runner.evaluate(make_case(write_frame(tmp_path / "f.jpg"))))
    assert result.error is None and result.triggered is False
    assert result.reason.startswith("Biscuits in trays")


def test_an_unreadable_reply_is_asked_once_more_before_it_becomes_an_error(
    monkeypatch, tmp_path: Path,
):
    from argus.eval.openrouter import OpenRouterRunner

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-value")
    good = {"triggered": True, "confidence": 0.9, "reason": "second time", "box_2d": None}
    with FakeOpenAIServer(replies=[GARBAGE, good], model=CLOUD_MODEL) as cloud:
        runner = OpenRouterRunner(name=CLOUD_MODEL, base_url=cloud.base_url, backoff_sec=0.0)
        result = asyncio.run(runner.evaluate(make_case(write_frame(tmp_path / "f.jpg"))))
        assert cloud.calls == 2
    assert result.triggered is True and result.error is None and runner.retried == 1


def test_a_model_that_stays_unreadable_is_recorded_with_why(monkeypatch, tmp_path: Path):
    """The error string has to carry the diagnosis, not just the symptom.

    "empty response" 396 times is what the first run recorded, and it named
    neither the token budget nor the reasoning that ate it.
    """
    from argus.eval.openrouter import OpenRouterRunner

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-value")
    starved = Reply(
        message={"content": "", "reasoning": "Let me look at the image carefully"},
        finish_reason="length",
        usage={"prompt_tokens": 900, "completion_tokens": 192,
               "completion_tokens_details": {"reasoning_tokens": 192}},
    )
    with FakeOpenAIServer(replies=[starved], model=CLOUD_MODEL) as cloud:
        runner = OpenRouterRunner(name=CLOUD_MODEL, base_url=cloud.base_url, backoff_sec=0.0)
        result = asyncio.run(runner.evaluate(make_case(write_frame(tmp_path / "f.jpg"))))
        assert cloud.calls == 2  # one retry, then recorded
    assert result.triggered is None
    assert "finish=length" in result.error and "reasoning_tokens=192" in result.error
    assert "read=reasoning" in result.error


def test_reasoning_is_asked_off_and_the_budget_is_not_the_local_one(monkeypatch, tmp_path: Path):
    from argus.eval.openrouter import NO_REASONING, SCHEMA_FORMAT, OpenRouterRunner

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-value")
    with FakeOpenAIServer(model=CLOUD_MODEL) as cloud:
        runner = OpenRouterRunner(name=CLOUD_MODEL, base_url=cloud.base_url)
        asyncio.run(runner.evaluate(make_case(write_frame(tmp_path / "f.jpg"))))
    sent = cloud.requests[0]
    assert sent["reasoning"] == NO_REASONING
    assert sent["response_format"] == SCHEMA_FORMAT
    # 192 is the appliance's budget for the served model. Out here it buys a
    # third of a chain of thought and no answer at all.
    assert sent["max_tokens"] >= 1024


def test_an_endpoint_that_insists_on_reasoning_is_obeyed_once_not_per_case(
    monkeypatch, tmp_path: Path,
):
    """google/gemini-3.7-flash and qwen/qwen3.8-max both answer 400 to `effort: none`."""
    from argus.eval.openrouter import BOUNDED_REASONING, OpenRouterRunner

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-value")
    verdict = {"triggered": False, "confidence": 0.5, "reason": "clear", "box_2d": None}
    with FakeOpenAIServer(replies=[verdict], model=CLOUD_MODEL,
                          demands_reasoning=True) as cloud:
        runner = OpenRouterRunner(name=CLOUD_MODEL, base_url=cloud.base_url, backoff_sec=0.0)
        case = make_case(write_frame(tmp_path / "f.jpg"))
        first = asyncio.run(runner.evaluate(case))
        after = cloud.calls
        second = asyncio.run(runner.evaluate(case))

    assert first.error is None and second.error is None
    assert runner.reasoning == BOUNDED_REASONING
    # The refusal cost one extra request, once — not once per case.
    assert after == 2 and cloud.calls == 3
    assert cloud.requests[-1]["reasoning"] == BOUNDED_REASONING
    assert any("bounded to effort=low" in note for note in runner.notes)


def test_the_case_that_loses_the_race_to_degrade_is_not_scored_as_a_failure(
    monkeypatch, tmp_path: Path,
):
    """Both in-flight cases were sent under options the endpoint rejects.

    Only one of them gets to be the one that steps the options down. Without
    the generation check the other is recorded as a hard 400 — a real error for
    a request the harness would never send again, which is exactly the kind of
    row that made the first cloud run unreadable.
    """
    from argus.eval.openrouter import OpenRouterRunner

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-value")
    verdict = {"triggered": False, "confidence": 0.5, "reason": "clear", "box_2d": None}
    with FakeOpenAIServer(replies=[verdict], model=CLOUD_MODEL,
                          demands_reasoning=True) as cloud:
        runner = OpenRouterRunner(name=CLOUD_MODEL, base_url=cloud.base_url, backoff_sec=0.0)
        path = write_frame(tmp_path / "f.jpg")
        cases = [make_case(path, case_id="a"), make_case(path, case_id="b")]
        results = asyncio.run(run_cases(runner, cases, concurrency=2, log=lambda *_: None))
    assert [r.error for r in results] == [None, None]


def test_a_price_or_a_not_found_error_is_never_retried(monkeypatch, tmp_path: Path):
    """Retrying a deterministic refusal spends money to learn nothing."""
    from argus.eval.openrouter import OpenRouterRunner

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-value")
    with FakeOpenAIServer(replies=[402], model=CLOUD_MODEL) as cloud:
        runner = OpenRouterRunner(name=CLOUD_MODEL, base_url=cloud.base_url, backoff_sec=0.0)
        asyncio.run(runner.evaluate(make_case(write_frame(tmp_path / "f.jpg"))))
        assert cloud.calls == 1


def test_a_cloud_http_error_becomes_a_result_without_leaking_the_key(monkeypatch, tmp_path: Path):
    from argus.eval.openrouter import OpenRouterRunner

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret-do-not-log")
    with FakeOpenAIServer(replies=[402], model=CLOUD_MODEL) as cloud:
        runner = OpenRouterRunner(name=CLOUD_MODEL, base_url=cloud.base_url)
        result = asyncio.run(runner.evaluate(make_case(write_frame(tmp_path / "f.jpg"))))
    assert result.triggered is None and "402" in (result.error or "")
    assert "sk-secret-do-not-log" not in json.dumps(result.as_dict())


def test_pricing_comes_from_the_catalog(monkeypatch):
    from argus.eval.openrouter import OpenRouterRunner, attach_pricing

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-value")
    with FakeOpenAIServer(
        model=CLOUD_MODEL, pricing={"prompt": "0.0000002", "completion": "0.0000008"},
    ) as cloud:
        runner = OpenRouterRunner(name=CLOUD_MODEL, base_url=cloud.base_url)
        notes = attach_pricing([runner])
    assert notes == []
    assert runner.pricing() == (2e-7, 8e-7)
    assert runner.vision_verified is True


def test_a_text_only_candidate_is_flagged_loudly(monkeypatch):
    """The check that would have caught z-ai/glm-5.2 before somebody ran it."""
    from argus.eval.openrouter import OpenRouterRunner, attach_pricing

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-value")
    with FakeOpenAIServer(model=CLOUD_MODEL, modalities=["text"],
                          pricing={"prompt": "0.0000002", "completion": "0.0000008"}) as cloud:
        runner = OpenRouterRunner(name=CLOUD_MODEL, base_url=cloud.base_url)
        notes = attach_pricing([runner])
    assert runner.vision_verified is False
    assert any("no image input" in note for note in notes)


def test_a_baseline_keeps_its_cost_column(config: ArgusConfig, tmp_path: Path):
    """A cloud model replayed through --baseline must not render as "local".

    vLLM serves one model per process and OpenRouter bills per token, so a
    multi-model cloud comparison is a chain of runs joined by --baseline. If the
    join drops the price, every model after the first is reported as costing
    nothing, which is the one number a procurement decision turns on.
    """
    from argus.eval.metrics import score

    cases = [make_case(write_frame(tmp_path / "f.jpg"), case_id="c1")]
    results = [RunnerResult(case_id="c1", model=CLOUD_MODEL, triggered=True,
                            prompt_tokens=1000, completion_tokens=100)]
    report = score(CLOUD_MODEL, f"{CLOUD_MODEL} (openrouter)", cases, results,
                   pricing=(2e-6, 6e-6))
    assert report.cost_per_1k is not None
    stored = report.as_dict()
    assert stored["prompt_usd"] == 2e-6 and stored["completion_usd"] == 6e-6

    run = EvalRun(started_at="2026-08-25T00:00:00+00:00")
    run.eval_set = EvalSet(cases=cases)
    run.results[f"{CLOUD_MODEL} (openrouter)"] = results
    run.reports.append(report)
    path = tmp_path / "prev.json"
    write_json(run, path)

    replayed = EvalRun(started_at="2026-08-26T00:00:00+00:00")
    replayed.eval_set = EvalSet(cases=cases)
    load_baseline(replayed, path, log=lambda *_: None)
    assert replayed.reports[0].cost_per_1k == pytest.approx(report.cost_per_1k)
    assert "local" not in render(replayed).split("## Latency")[0].split("| model | n |")[1][:400]


def test_a_measurement_caveat_travels_with_the_measurement(tmp_path: Path):
    """"Reasoning could not be disabled" explains a latency column three runs later."""
    cases = [make_case(write_frame(tmp_path / "f.jpg"), case_id="c1")]
    run = EvalRun(started_at="2026-08-25T00:00:00+00:00")
    run.eval_set = EvalSet(cases=cases)
    run.results["m (openrouter)"] = [RunnerResult(case_id="c1", model="m", triggered=True)]
    run.notes.append("cloud eval: m: reasoning cannot be disabled — bounded to effort=low")
    run.notes.append("eval set: something about this run only")
    first = tmp_path / "one.json"
    write_json(run, first)

    second = EvalRun(started_at="2026-08-26T00:00:00+00:00")
    second.eval_set = EvalSet(cases=cases)
    load_baseline(second, first, log=lambda *_: None)
    carried = [n for n in second.notes if n.startswith("cloud eval:")]
    assert len(carried) == 1 and "one.json" in carried[0]
    # Not everything travels: a note about that run's eval set is not a property
    # of the model and would be false here.
    assert not any("something about this run only" in n for n in second.notes)

    # And a third hop must not re-stamp the provenance.
    third_path = tmp_path / "two.json"
    write_json(second, third_path)
    third = EvalRun(started_at="2026-08-27T00:00:00+00:00")
    third.eval_set = EvalSet(cases=cases)
    load_baseline(third, third_path, log=lambda *_: None)
    hop = [n for n in third.notes if n.startswith("cloud eval:")][0]
    assert hop.count("(measured in ") == 1 and "one.json" in hop


@pytest.mark.parametrize("error, kind", [
    ('RuntimeError: HTTP 402: {"error":{"message":"more credits"}}', "HTTP 402"),
    ("RuntimeError: HTTP 429: slow down", "HTTP 429"),
    ("VlmResponseError: empty response [finish=length]", "VlmResponseError: empty response"),
    ("VlmResponseError: not JSON: Expecting value: line 1 column 1",
     "VlmResponseError: unreadable reply"),
])
def test_errors_are_counted_by_kind_not_by_wording(error: str, kind: str):
    from argus.eval.metrics import error_kind

    assert error_kind(error) == kind


def test_running_out_of_credit_is_not_reported_as_a_bad_model(tmp_path: Path):
    """The distinction the 2026-08-25 kimi-k3 row exists to make.

    334 refusals for an empty account and 334 refusals from a model that cannot
    read a frame produce the same number in the errors column and mean opposite
    things. The report has to say which it is next to the number, because a
    reader comparing F1 has no reason to scroll to the Errors section.
    """
    path = write_frame(tmp_path / "f.jpg")
    cases = [make_case(path, case_id=f"c{i}", expected=True) for i in range(4)]
    results = [RunnerResult(case_id="c0", model="m", triggered=True)] + [
        RunnerResult(case_id=f"c{i}", model="m", triggered=None, latency_sec=0.01,
                     error='RuntimeError: HTTP 402: {"error":{"message":"more credits"}}')
        for i in range(1, 4)
    ]
    run = EvalRun(started_at="2026-08-25T00:00:00+00:00")
    run.eval_set = EvalSet(cases=cases)
    run.results["m (openrouter)"] = results
    run.reports.append(score("m", "m (openrouter)", cases, results))
    document = render(run)

    assert run.reports[0].error_kinds == {"HTTP 402": 3}
    assert "3 × `HTTP 402`" in document
    assert "ran out of credit" in document
    assert "incomplete run rather than as a bad model" in document


def test_glm_is_documented_as_excluded_and_why():
    assert "z-ai/glm-5.2" in EXCLUDED_CANDIDATES
    assert "TEXT-ONLY" in EXCLUDED_CANDIDATES["z-ai/glm-5.2"]


# --------------------------------------------------------------------------
# the isolation boundary
# --------------------------------------------------------------------------


def test_the_runtime_tree_never_imports_the_cloud_runner():
    """The "no cloud calls in the product" claim, as an assertion.

    A fresh interpreter imports everything that runs during `argus run` and
    `argus api`, then reports any `argus.eval` module that came along. Nothing
    should have.
    """
    code = (
        "import argus.engine, argus.api.app, argus.vlm.scheduler, argus.index.runner, "
        "argus.retention, argus.notify, argus.health;"
        "import sys, json;"
        "print(json.dumps(sorted(m for m in sys.modules if m.startswith('argus.eval'))))"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True,
                               text=True, check=True)
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == []


def test_importing_the_harness_does_not_import_the_cloud_runner():
    """The gate is checked *before* the module that can make a request exists."""
    code = (
        "import argus.eval.harness, argus.eval.dataset, argus.eval.report, argus.cli;"
        "import sys, json;"
        "print(json.dumps('argus.eval.openrouter' in sys.modules))"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True,
                               text=True, check=True)
    assert json.loads(completed.stdout.strip().splitlines()[-1]) is False


def test_only_the_cli_reaches_into_the_eval_package_at_all():
    """`argus eval` has to be reachable from somewhere, and `argus/cli.py` is
    that somewhere. Every other module in the product tree must not know this
    package exists — a new import here is how an eval-only dependency ends up on
    a runtime path by accident."""
    from argus.config import ROOT

    offenders = []
    for path in sorted((ROOT / "argus").rglob("*.py")):
        if path.parent.name == "eval" or path.name == "cli.py":
            continue
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith(("import argus.eval", "from argus.eval")):
                offenders.append(f"{path.relative_to(ROOT)}: {stripped}")
    assert offenders == []


def test_nothing_outside_the_cloud_module_names_it_as_an_import():
    """Only `harness._cloud_runners` may import the cloud runner, and only
    inside a function body that runs after the gate."""
    from argus.config import ROOT

    offenders = []
    for path in sorted((ROOT / "argus").rglob("*.py")):
        if path.name == "openrouter.py":
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if not stripped.startswith(("import argus.eval.openrouter",
                                        "from argus.eval.openrouter")):
                continue
            if path.name == "harness.py" and line.startswith("    "):
                continue  # indented == inside the gated function
            offenders.append(f"{path.relative_to(ROOT)}:{number}: {stripped}")
    assert offenders == []


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def test_confusion_counts_the_way_a_safety_officer_would():
    table = Confusion()
    table.add(expected=True, predicted=True)     # violation caught
    table.add(expected=False, predicted=True)    # alert for nothing
    table.add(expected=True, predicted=False)    # violation missed
    table.add(expected=False, predicted=False)
    assert (table.tp, table.fp, table.fn, table.tn) == (1, 1, 1, 1)
    assert table.precision == 0.5 and table.recall == 0.5 and table.f1 == 0.5


def test_ratios_are_none_rather_than_zero_when_undefined():
    """A model with no positive predictions has undefined precision. Reporting
    0.0 would rank it below a model that at least tried."""
    table = Confusion(tn=5)
    assert table.precision is None and table.recall is None and table.f1 is None
    assert table.accuracy == 1.0


def test_errors_are_scored_separately_from_wrong_answers(tmp_path: Path):
    path = write_frame(tmp_path / "f.jpg")
    cases = [make_case(path, expected=True, case_id="a"),
             make_case(path, expected=True, case_id="b")]
    results = [
        RunnerResult(case_id="a", model="m", triggered=True, latency_sec=1.0),
        RunnerResult(case_id="b", model="m", triggered=None, latency_sec=9.0,
                     error="VlmUnavailable: timeout"),
    ]
    report = score("m", "m", cases, results)
    assert report.overall.tp == 1 and report.overall.fn == 0
    assert report.overall.errors == 1
    # The slow failure is still a latency measurement.
    assert report.latency.n == 2 and report.latency.mean == 5.0


def test_unlabelled_cases_are_timed_but_never_scored(tmp_path: Path):
    path = write_frame(tmp_path / "f.jpg")
    cases = [make_case(path, expected=None, case_id="a")]
    results = [RunnerResult(case_id="a", model="m", triggered=True, latency_sec=2.0)]
    report = score("m", "m", cases, results)
    assert report.overall.n == 0
    assert report.unlabelled == 1
    assert report.latency.mean == 2.0


def test_cost_per_1k_uses_measured_tokens_and_published_prices(tmp_path: Path):
    path = write_frame(tmp_path / "f.jpg")
    cases = [make_case(path, expected=True, case_id="a")]
    results = [RunnerResult(case_id="a", model="m", triggered=True, latency_sec=1.0,
                            prompt_tokens=1000, completion_tokens=100)]
    report = score("m", "m", cases, results, pricing=(2e-7, 8e-7))
    assert report.cost_per_1k == pytest.approx(1000 * (1000 * 2e-7 + 100 * 8e-7))


def test_p95_is_a_measured_sample_not_an_interpolation():
    latency = Latency(samples=[1.0, 2.0, 3.0, 100.0])
    assert latency.quantile(0.95) == 100.0
    assert latency.quantile(0.5) == 2.0


def test_disagreements_include_a_model_that_could_not_answer(tmp_path: Path):
    path = write_frame(tmp_path / "f.jpg")
    cases = [make_case(path, expected=True, case_id="a"),
             make_case(path, expected=True, case_id="b")]
    found = disagreements(cases, {
        "left": [RunnerResult("a", "left", True), RunnerResult("b", "left", True)],
        "right": [RunnerResult("a", "right", True),
                  RunnerResult("b", "right", None, error="boom")],
    })
    assert [item.case.id for item in found] == ["b"]
    assert found[0].matches_expected == ["left"]


def test_one_model_produces_no_disagreements(tmp_path: Path):
    path = write_frame(tmp_path / "f.jpg")
    cases = [make_case(path, case_id="a")]
    assert disagreements(cases, {"only": [RunnerResult("a", "only", True)]}) == []


# --------------------------------------------------------------------------
# baselines — two local candidates are two runs
# --------------------------------------------------------------------------


def _baseline_file(path: Path, *, verdicts: dict[str, bool], label: str = "old (local)",
                   latency: float = 4.0, started: str = "2026-08-15T18:32:24+00:00",
                   prompts: dict[str, str] | None = None) -> Path:
    payload: dict = {
        "started_at": started,
        "results": {label: [
            {"case_id": case_id, "model": "old", "triggered": value,
             "confidence": 0.9, "reason": "because", "latency_sec": latency,
             "prompt_tokens": 600, "completion_tokens": 60, "error": None}
            for case_id, value in verdicts.items()
        ]},
    }
    if prompts is not None:
        payload["eval_set"] = {
            "cases": [{"id": case_id, "prompt": text} for case_id, text in prompts.items()],
        }
    path.write_text(json.dumps(payload))
    return path


def test_a_baseline_is_rescored_against_todays_labels(tmp_path: Path):
    """The point of re-scoring rather than copying the old report: a label fixed
    since that run must move the old model's score too, or the comparison is
    between two different questions."""
    frame = write_frame(tmp_path / "f.jpg")
    run = EvalRun(started_at="now")
    # The stored answer was `True`. Under the label it was scored against then
    # that may have been a hit; here the label says False, so it is a miss.
    run.eval_set = EvalSet(cases=[make_case(frame, expected=False, case_id="a")])
    load_baseline(run, _baseline_file(tmp_path / "old.json", verdicts={"a": True}),
                  log=lambda *_: None)

    assert len(run.reports) == 1
    assert (run.reports[0].overall.fp, run.reports[0].overall.tp) == (1, 0)


def test_a_baseline_label_carries_its_date_because_its_latency_is_not_todays(tmp_path: Path):
    frame = write_frame(tmp_path / "f.jpg")
    run = EvalRun(started_at="now")
    run.eval_set = EvalSet(cases=[make_case(frame, expected=True, case_id="a")])
    load_baseline(run, _baseline_file(tmp_path / "old.json", verdicts={"a": True}),
                  log=lambda *_: None)

    assert run.reports[0].label == "old (local) · 2026-08-15 baseline"
    assert run.reports[0].latency.mean == 4.0
    assert any("its latency is from that run, not this one" in note for note in run.notes)
    # Recorded as data, so the ship/hold section does not have to parse the
    # label to find out which row is the incumbent.
    assert run.baseline_labels == ["old (local) · 2026-08-15 baseline"]


def test_a_baseline_of_a_baseline_keeps_the_date_it_was_measured_on(tmp_path: Path):
    """Three local candidates are three runs, and run 3 loads run 2's file — which
    already holds run 1's rows. Re-dating them would credit a model's latency to
    a run that never measured it."""
    frame = write_frame(tmp_path / "f.jpg")
    run = EvalRun(started_at="now")
    run.eval_set = EvalSet(cases=[make_case(frame, expected=True, case_id="a")])
    chained = tmp_path / "run2.json"
    # No `baseline_labels` key: written by a version that predates the field, so
    # the tag on the label is the only evidence the row is already a baseline.
    chained.write_text(json.dumps({
        "started_at": "2026-08-16T09:00:00+00:00",
        "results": {
            "first (local) · 2026-08-15 baseline": [
                {"case_id": "a", "model": "first", "triggered": True, "latency_sec": 4.0},
            ],
            "second (local)": [
                {"case_id": "a", "model": "second", "triggered": True, "latency_sec": 9.0},
            ],
        },
    }))
    load_baseline(run, chained, log=lambda *_: None)

    assert run.baseline_labels == [
        "first (local) · 2026-08-15 baseline",
        "second (local) · 2026-08-16 baseline",
    ]


def test_a_baseline_that_answered_a_different_prompt_says_so(tmp_path: Path):
    """A stored answer is only comparable to the question it was given, and a
    results file stores its own eval set — so this is checkable rather than
    assumed. It is how the count-clause change reads in one table instead of
    looking like a model that suddenly got better."""
    frame = write_frame(tmp_path / "f.jpg")
    run = EvalRun(started_at="now")
    run.eval_set = EvalSet(cases=[make_case(frame, expected=True, case_id="a")])
    old = _baseline_file(
        tmp_path / "old.json", verdicts={"a": True},
        prompts={"a": "a question nobody asks any more"},
    )

    load_baseline(run, old, log=lambda *_: None)

    assert run.prompt_drift == {"old (local) · 2026-08-15 baseline": 1}
    note = next(n for n in run.notes if n.startswith("baseline:"))
    assert "answered a DIFFERENT prompt on 1 of 1 case(s)" in note
    assert "read it as a before, not as a rival" in note


def test_a_baseline_asked_the_same_question_is_not_flagged(tmp_path: Path):
    frame = write_frame(tmp_path / "f.jpg")
    run = EvalRun(started_at="now")
    case = make_case(frame, expected=True, case_id="a")
    run.eval_set = EvalSet(cases=[case])

    load_baseline(
        run,
        _baseline_file(tmp_path / "old.json", verdicts={"a": True},
                       prompts={"a": case.prompt}),
        log=lambda *_: None,
    )

    assert run.prompt_drift == {"old (local) · 2026-08-15 baseline": 0}
    assert not any("DIFFERENT prompt" in note for note in run.notes)


def test_a_baseline_case_that_no_longer_exists_is_counted_and_named(tmp_path: Path):
    frame = write_frame(tmp_path / "f.jpg")
    run = EvalRun(started_at="now")
    run.eval_set = EvalSet(cases=[make_case(frame, expected=True, case_id="a")])
    load_baseline(
        run,
        _baseline_file(tmp_path / "old.json", verdicts={"a": True, "gone": False}),
        log=lambda *_: None,
    )
    note = next(n for n in run.notes if n.startswith("baseline:"))
    assert "1 of 2 stored case(s) are in this eval set" in note
    assert "1 stored case(s) no longer exist" in note
    # The vanished case is not scored against a case that is not there.
    assert run.reports[0].attempted == 1


def test_a_baseline_and_a_live_model_disagree_in_one_report(
    seeded: ArgusConfig, database: Database, tmp_path: Path,
):
    """The whole reason --baseline exists: one report, two local candidates, and
    a disagreements table that only appears with two."""
    ids = [case.id for case in build_eval_set(seeded, database, slices=("triage",)).cases]
    baseline = _baseline_file(tmp_path / "old.json", verdicts={cid: False for cid in ids})

    with FakeOpenAIServer(replies=[
        {"triggered": True, "confidence": 0.95, "reason": "over limit", "box_2d": None},
    ]) as server:
        seeded.vlm.base_url = server.base_url
        seeded.vlm.model = MODEL
        run = run_eval(seeded, database, slices=("triage",), baseline=baseline,
                       log=lambda *_: None)

    assert len(run.reports) == 2
    # Incumbent first: the model under test reads as the challenger.
    assert run.reports[0].label in run.baseline_labels
    assert len(run.disagreements) == len(ids)
    document = render(run)
    assert "2026-08-15 baseline" in document
    assert "nothing to disagree about" not in document


def _two_model_run(
    *, incumbent_f1_source: tuple[bool, bool], challenger_f1_source: tuple[bool, bool],
    incumbent_latency: float, challenger_latency: float, tmp_path: Path,
) -> EvalRun:
    """A run holding one baseline and one live model on two counting cases.

    Each tuple is (verdict on a case whose label is True, verdict on a case
    whose label is False) — so (True, False) is perfect and (True, True) is the
    Phase 2 failure, a model that upholds everything.
    """
    from argus.eval.harness import EvalRun
    from argus.eval.metrics import score

    frame = write_frame(tmp_path / "f.jpg")
    cases = [
        make_case(frame, expected=True, policy=COUNT_VERIFY_POLICY, case_id="hit"),
        make_case(frame, expected=False, policy=COUNT_VERIFY_POLICY, case_id="miss"),
    ]
    run = EvalRun(started_at="now")
    run.eval_set = EvalSet(cases=cases)
    run.context = {"vlm_interval_sec": 8.0, "vlm_max_concurrency": 2}
    for label, (a, b), latency in (
        ("old (local) · 2026-08-15 baseline", incumbent_f1_source, incumbent_latency),
        ("new (local)", challenger_f1_source, challenger_latency),
    ):
        results = [
            RunnerResult("hit", label, a, latency_sec=latency),
            RunnerResult("miss", label, b, latency_sec=latency),
        ]
        run.results[label] = results
        run.reports.append(score(label, label, cases, results))
    run.baseline_labels = ["old (local) · 2026-08-15 baseline"]
    return run


def test_a_challenger_that_is_better_and_fast_enough_ships(tmp_path: Path):
    run = _two_model_run(
        incumbent_f1_source=(True, True), challenger_f1_source=(True, False),
        incumbent_latency=3.0, challenger_latency=3.0, tmp_path=tmp_path,
    )
    section = render(run)
    assert "**SHIP**" in section
    assert "**HOLD**" not in section


def test_a_challenger_that_is_better_but_too_slow_is_held(tmp_path: Path):
    """Both conditions, not either: better and unusable changes nothing at a site."""
    run = _two_model_run(
        incumbent_f1_source=(True, True), challenger_f1_source=(True, False),
        incumbent_latency=5.0, challenger_latency=30.0, tmp_path=tmp_path,
    )
    section = render(run)
    assert "**HOLD**" in section
    assert "against a floor of 3" in section


def test_a_challenger_that_is_fast_but_no_better_is_held(tmp_path: Path):
    run = _two_model_run(
        incumbent_f1_source=(True, False), challenger_f1_source=(True, True),
        incumbent_latency=5.0, challenger_latency=2.0, tmp_path=tmp_path,
    )
    document = render(run)
    assert "not materially better on the counting slice" in document
    # The trade reads in the direction it actually goes: capacity bought with
    # accuracy, not the reverse.
    assert "More questions asked faster is a bigger product, bought with " \
           "answers that are measurably worse" in document


def test_the_trade_sentence_follows_the_direction_of_the_trade(tmp_path: Path):
    run = _two_model_run(
        incumbent_f1_source=(True, True), challenger_f1_source=(True, False),
        incumbent_latency=3.0, challenger_latency=9.0, tmp_path=tmp_path,
    )
    assert "the better answers are what pays for it" in render(run)


def test_the_same_model_after_a_prompt_change_is_a_before_after_not_a_control(tmp_path: Path):
    """The control text says a difference means the serving stack is broken.
    After the count clause landed, the same model *did* move — by 0.147 F1 — and
    calling that non-determinism would have been the wrong alarm entirely."""
    run = _two_model_run(
        incumbent_f1_source=(True, True), challenger_f1_source=(True, False),
        incumbent_latency=4.98, challenger_latency=5.35, tmp_path=tmp_path,
    )
    for report in run.reports:                      # one model, two runs
        report.model = "Qwen/Qwen3-VL-8B-Instruct"
    run.prompt_drift = {"old (local) · 2026-08-15 baseline": 71}

    document = render(run)

    assert "**BEFORE/AFTER**" in document
    assert "**CONTROL**" not in document
    assert "is the prompt's doing and not the model's" in document
    assert "the counting F1 move of +0.333" in document
    assert "4.98 s -> 5.35 s per evaluation" in document


def test_two_rows_from_one_baseline_are_still_a_control(tmp_path: Path):
    """Both answered whatever that run asked, so they answered each other's
    question — a rerun stays a determinism check even when the *production*
    prompt has moved on since."""
    run = _two_model_run(
        incumbent_f1_source=(True, False), challenger_f1_source=(True, False),
        incumbent_latency=4.98, challenger_latency=4.93, tmp_path=tmp_path,
    )
    for report in run.reports:
        report.model = "Qwen/Qwen3-VL-8B-Instruct"
    run.baseline_labels = [report.label for report in run.reports]
    run.prompt_drift = {report.label: 71 for report in run.reports}

    document = render(run)

    assert "**CONTROL**" in document
    assert "**BEFORE/AFTER**" not in document


def test_a_hairs_breadth_lead_is_not_a_reason_to_swap_the_default(tmp_path: Path):
    """`better` has to mean better than the noise floor: on 68 counting cases one
    flipped verdict moves F1 by more than a point, and a shipping default should
    not turn on that."""
    from argus.eval.harness import EvalRun
    from argus.eval.metrics import score
    from argus.eval.report import MATERIAL_F1_GAIN

    frame = write_frame(tmp_path / "f.jpg")
    # 40 positives, 40 negatives: enough resolution for a sub-threshold lead.
    cases, hits, misses = [], [], []
    for index in range(40):
        cases.append(make_case(frame, expected=True, policy=COUNT_VERIFY_POLICY,
                               case_id=f"pos{index}"))
        cases.append(make_case(frame, expected=False, policy=COUNT_VERIFY_POLICY,
                               case_id=f"neg{index}"))
    run = EvalRun(started_at="now")
    run.eval_set = EvalSet(cases=cases)
    run.context = {"vlm_interval_sec": 8.0, "vlm_max_concurrency": 2}
    # The incumbent fires on every negative; the challenger spares two of them.
    for label, spared in (("old (local) · 2026-08-15 baseline", 0), ("new (local)", 2)):
        results = [RunnerResult(f"pos{i}", label, True, latency_sec=3.0) for i in range(40)]
        results += [RunnerResult(f"neg{i}", label, i >= spared, latency_sec=3.0)
                    for i in range(40)]
        run.results[label] = results
        run.reports.append(score(label, label, cases, results))
    run.baseline_labels = ["old (local) · 2026-08-15 baseline"]

    incumbent, challenger = (r.by_policy[COUNT_VERIFY_POLICY].f1 for r in run.reports)
    assert 0 < challenger - incumbent < MATERIAL_F1_GAIN
    assert "**HOLD**" in render(run)


def test_the_same_model_run_twice_is_the_noise_floor_not_a_candidate(tmp_path: Path):
    """Two runs of one model bound how big a gap between two *different* models
    has to be. Scoring that pair as a ship/hold decision would be nonsense."""
    from argus.eval.harness import EvalRun
    from argus.eval.metrics import score

    frame = write_frame(tmp_path / "f.jpg")
    cases = [make_case(frame, expected=True, policy=COUNT_VERIFY_POLICY, case_id="c")]
    run = EvalRun(started_at="now")
    run.eval_set = EvalSet(cases=cases)
    run.context = {"vlm_interval_sec": 8.0, "vlm_max_concurrency": 2}
    for label in ("m (local) · 2026-08-15 baseline", "m (local)"):
        results = [RunnerResult("c", "m", True, latency_sec=3.0)]
        run.results[label] = results
        run.reports.append(score("m", label, cases, results))
    run.baseline_labels = ["m (local) · 2026-08-15 baseline"]

    document = render(run)
    assert "**CONTROL**" in document
    assert "**SHIP**" not in document and "**HOLD**" not in document
    assert "measures run-to-run determinism" in document


def test_two_runs_of_one_model_on_one_day_do_not_overwrite_each_other(tmp_path: Path):
    frame = write_frame(tmp_path / "f.jpg")
    run = EvalRun(started_at="now")
    run.eval_set = EvalSet(cases=[make_case(frame, expected=True, case_id="a")])
    chained = tmp_path / "run.json"
    chained.write_text(json.dumps({
        "started_at": "2026-08-15T09:00:00+00:00",
        "baseline_labels": ["m (local) · 2026-08-15 baseline"],
        "results": {
            "m (local) · 2026-08-15 baseline": [
                {"case_id": "a", "model": "m", "triggered": True, "latency_sec": 4.0}],
            "m (local)": [
                {"case_id": "a", "model": "m", "triggered": False, "latency_sec": 5.0}],
        },
    }))
    load_baseline(run, chained, log=lambda *_: None)

    assert run.baseline_labels == [
        "m (local) · 2026-08-15 baseline",
        "m (local) · 2026-08-15 baseline #2",
    ]
    assert [r.latency.mean for r in run.reports] == [4.0, 5.0]


def test_new_false_alarms_are_named_where_f1_cannot_see_them(tmp_path: Path):
    """`phone_usage_on_line` on this footage is all-negative: precision 0 and
    recall undefined score a model that invents two violations exactly like one
    that invents none."""
    from argus.eval.harness import EvalRun
    from argus.eval.metrics import score

    frame = write_frame(tmp_path / "f.jpg")
    cases = [
        make_case(frame, expected=True, policy=COUNT_VERIFY_POLICY, case_id="c"),
        make_case(frame, expected=False, policy="phone_usage_on_line", case_id="p"),
    ]
    run = EvalRun(started_at="now")
    run.eval_set = EvalSet(cases=cases)
    run.context = {"vlm_interval_sec": 8.0, "vlm_max_concurrency": 2}
    for label, phone in (("old (local) · 2026-08-15 baseline", False), ("new (local)", True)):
        results = [RunnerResult("c", label, True, latency_sec=3.0),
                   RunnerResult("p", label, phone, latency_sec=3.0)]
        run.results[label] = results
        run.reports.append(score(label, label, cases, results))
    run.baseline_labels = ["old (local) · 2026-08-15 baseline"]

    document = render(run)
    # Both models score "—" on that policy, so only the count says what changed.
    assert "`phone_usage_on_line` | `new (local)` | 1 | 0/1 | 0.000 | — | — |" in document
    assert "`phone_usage_on_line`: 0 -> 1 false alarm(s) in 1 case(s) — 1 more" in document


def test_the_policy_budget_is_derated_for_the_detector_sharing_the_gpu(tmp_path: Path):
    """An idle-box latency implies a budget the appliance does not have. Phase 2
    measured the gap; the report must apply it, not mention it."""
    from argus.eval.report import IN_SERVICE_DERATE, _policy_pairs

    run = _two_model_run(
        incumbent_f1_source=(True, False), challenger_f1_source=(True, False),
        incumbent_latency=4.0, challenger_latency=4.0, tmp_path=tmp_path,
    )
    pairs = _policy_pairs(run.reports[0], run)
    assert pairs == pytest.approx(2 * 8.0 / (4.0 * IN_SERVICE_DERATE))
    assert pairs < 2 * 8.0 / 4.0


def test_without_a_baseline_the_report_says_there_is_no_decision(
    seeded: ArgusConfig, database: Database,
):
    with FakeOpenAIServer(replies=[
        {"triggered": True, "confidence": 0.9, "reason": "y", "box_2d": None},
    ]) as server:
        seeded.vlm.base_url = server.base_url
        seeded.vlm.model = MODEL
        run = run_eval(seeded, database, slices=("triage",), log=lambda *_: None)
    document = render(run)
    assert "## Ship or hold" in document
    assert "no ship/hold decision to take from it" in document


def test_the_cli_refuses_to_overwrite_the_baseline_it_is_comparing_against(
    seeded: ArgusConfig, tmp_path: Path, capsys,
):
    """Two runs on one day resolve to the same default results path. Writing this
    run over the baseline is silent, and only discovered the next time somebody
    wants that baseline."""
    from argus.cli import main

    baseline = _baseline_file(tmp_path / "results.json", verdicts={"a": True})
    config_path = tmp_path / "argus.yaml"
    config_path.write_text(f"data_dir: {seeded.data_dir}\n")
    code = main([
        "-c", str(config_path), "eval",
        "--baseline", str(baseline), "--json", str(baseline),
    ])
    assert code == 2
    assert "same file" in capsys.readouterr().err
    # Untouched: the guard fires before anything is written.
    assert json.loads(baseline.read_text())["results"]


def test_a_baseline_with_no_live_model_still_reports(
    seeded: ArgusConfig, database: Database, tmp_path: Path,
):
    """A dead endpoint plus a baseline is a re-score of stored answers, which is
    a legitimate thing to want and must not fall out the early return."""
    ids = [case.id for case in build_eval_set(seeded, database, slices=("triage",)).cases]
    baseline = _baseline_file(tmp_path / "old.json", verdicts={cid: True for cid in ids})
    seeded.vlm.base_url = "http://127.0.0.1:9/v1"
    run = run_eval(seeded, database, slices=("triage",), baseline=baseline,
                   log=lambda *_: None)
    assert len(run.reports) == 1
    assert "No model was evaluated" not in render(run)


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------


def test_a_local_only_run_scores_reports_and_writes_both_files(
    seeded: ArgusConfig, database: Database, tmp_path: Path, capsys,
):
    with FakeOpenAIServer(replies=[
        {"triggered": True, "confidence": 0.95, "reason": "over limit", "box_2d": None},
    ]) as server:
        seeded.vlm.base_url = server.base_url
        seeded.vlm.model = MODEL
        run = run_eval(seeded, database, slices=("triage",), log=lambda *_: None)

    assert len(run.reports) == 1
    report = run.reports[0]
    # One confirmed, one false_positive; the model said "triggered" to both.
    assert (report.overall.tp, report.overall.fp) == (1, 1)
    assert any("cloud eval skipped" in note for note in run.notes)

    path = write_json(run, tmp_path / "results.json")
    payload = json.loads(path.read_text())
    assert payload["models"][0]["overall"]["tp"] == 1
    assert len(payload["eval_set"]["cases"]) == 2

    document = render(run)
    assert "# VLM evaluation" in document
    assert "z-ai/glm-5.2" in document
    assert "the appliance hands its verifier the *annotated* snapshot" in document.lower()


def test_a_run_with_no_model_available_still_produces_a_report(
    seeded: ArgusConfig, database: Database,
):
    """A dead endpoint must yield a report that says so, not a traceback and not
    a table of zeroes that looks like a bad model."""
    seeded.vlm.base_url = "http://127.0.0.1:9/v1"
    run = run_eval(seeded, database, slices=("triage",), log=lambda *_: None)
    assert run.reports == []
    assert any("local eval skipped" in note and "argus vlm start" in note
               for note in run.notes)
    assert "No model was evaluated" in render(run)


def test_the_report_names_the_counting_slice_even_when_it_is_empty(
    seeded: ArgusConfig, database: Database,
):
    with FakeOpenAIServer() as server:
        seeded.vlm.base_url = server.base_url
        seeded.vlm.model = MODEL
        run = run_eval(seeded, database, slices=("triage",), log=lambda *_: None)
    document = render(run)
    assert "## The counting slice" in document
    assert "**Empty.**" in document


def test_the_cli_runs_a_local_eval_end_to_end(
    seeded: ArgusConfig, database: Database, tmp_path: Path, monkeypatch, capsys,
):
    from argus.cli import main

    config_path = tmp_path / "argus.yaml"
    with FakeOpenAIServer(replies=[
        {"triggered": False, "confidence": 0.9, "reason": "no", "box_2d": None},
    ]) as server:
        config_path.write_text(
            f"data_dir: {seeded.data_dir}\n"
            "vlm:\n"
            f"  base_url: {server.base_url}\n"
            f"  model: {MODEL}\n"
            "eval:\n"
            f"  labels_path: {seeded.eval.labels_path}\n"
            f"  frames_dir: {seeded.eval.frames_dir}\n"
        )
        code = main([
            "-c", str(config_path), "eval", "--slice", "triage",
            "--report", str(tmp_path / "report.md"), "--json", str(tmp_path / "out.json"),
        ])
    assert code == 0
    assert "# VLM evaluation" in (tmp_path / "report.md").read_text()
    assert json.loads((tmp_path / "out.json").read_text())["models"]
    assert "report:" in capsys.readouterr().out


# --------------------------------------------------------------------------
# scoping a policy to the frames it is actually about
# --------------------------------------------------------------------------


def test_a_policy_with_no_applies_to_is_asked_about_every_frame():
    """The behaviour every label file had before the key existed, kept exactly."""
    assert policy_applies({"prompt": "x"}, "anything-at-all")
    assert policy_applies({"prompt": "x", "applies_to": None}, "anything-at-all")


def test_applies_to_matches_frame_ids_as_globs():
    spec = {"applies_to": ["Loading-Point-*-bay"]}
    assert policy_applies(spec, "Loading-Point-0000300-bay")
    assert not policy_applies(spec, "Loading-Point-0000300-dock")
    assert not policy_applies(spec, "Line-No-3-Packing3-Bas-0000300-bay")
    # One pattern may be written without the list, because a policy about one
    # scene is the common case and a list of one reads like a mistake.
    assert policy_applies({"applies_to": "*-bay"}, "Loading-Point-0000300-bay")


def test_a_malformed_applies_to_widens_rather_than_silences():
    """One bad row must not cost a whole eval run — and the failure has to be in
    the direction of asking too much, never of a policy quietly not being
    measured at all."""
    for broken in ({"applies_to": []}, {"applies_to": 7}, {"applies_to": {}}):
        assert policy_applies(broken, "any-frame")
    assert policy_applies("not a mapping", "any-frame")


def test_a_scoped_policy_skips_the_frames_it_is_not_about(
    seeded: ArgusConfig, database: Database, tmp_path: Path,
):
    """The whole point: seven prompts over eighty frames is 560 evaluations, and
    the ones asking about a dispatch bay over a packing hall are unlabelled by
    construction — GPU spent producing rows no report can score."""
    for name in ("bay-0000300", "hall-0000300"):
        write_frame(tmp_path / "frames" / f"{name}.jpg")
    save(seeded.eval_labels_path, {
        "version": 1,
        "policies": {
            "cartons": {"label": "Cartons", "prompt": "how many cartons?",
                        "applies_to": ["bay-*"]},
            "anywhere": {"label": "Anywhere", "prompt": "anything wrong?"},
        },
        "frames": [
            {"id": "bay-0000300", "source": str(tmp_path / "frames" / "bay-0000300.jpg"),
             "ts_sec": 3.0, "labels": {"cartons": True, "anywhere": False}},
            {"id": "hall-0000300", "source": str(tmp_path / "frames" / "hall-0000300.jpg"),
             "ts_sec": 3.0, "labels": {"cartons": None, "anywhere": False}},
        ],
    })
    eval_set = build_eval_set(seeded, database, slices=("policy",))
    pairs = {(case.policy, case.id) for case in eval_set.cases}
    assert ("cartons", "policy-bay-0000300-cartons") in pairs
    assert ("cartons", "policy-hall-0000300-cartons") not in pairs
    # The unscoped policy is untouched.
    assert sum(1 for policy, _ in pairs if policy == "anywhere") == 2


def test_build_labels_scopes_a_policy_to_its_own_cameras_scene(
    seeded: ArgusConfig, database: Database,
):
    """A policy the appliance already scoped to one camera is scoped in the
    label file too, so the human labelling is asked about that camera's frames
    and the report is not padded with cases nobody can score."""
    database.upsert_camera(Camera(id="bay", name="Bay",
                                  source="/plant/Loading_Point.MOV", kind="file"))
    database.upsert_policy(Policy(
        id="cartons", kind="vlm", detector_key="vlm_prompt", label="Cartons",
        prompt="how many cartons?", lifecycle="pilot", scope=["bay"],
    ))
    database.upsert_policy(Policy(
        id="everywhere", kind="vlm", detector_key="vlm_prompt", label="Everywhere",
        prompt="anything wrong?", lifecycle="pilot", scope=[],
    ))
    build_labels(seeded, database, log=lambda *_: None)
    policies = load(seeded.eval_labels_path)["policies"]
    assert policies["cartons"]["applies_to"] == ["Loading-Point-*"]
    # Unscoped means every camera in the appliance and every frame in the file.
    assert "applies_to" not in policies["everywhere"]


def test_a_policy_scoped_to_a_camera_with_no_video_is_not_narrowed(
    seeded: ArgusConfig, database: Database,
):
    """An RTSP camera has no file in footage/ to have been sampled from, so a
    pattern derived from it would match nothing — which would turn a policy the
    appliance runs into a policy the harness never asks."""
    database.upsert_camera(Camera(id="ip", name="IP", source="rtsp://cam/live", kind="rtsp"))
    database.upsert_policy(Policy(
        id="exit", kind="vlm", detector_key="vlm_prompt", label="Exit",
        prompt="blocked?", lifecycle="pilot", scope=["ip"],
    ))
    build_labels(seeded, database, log=lambda *_: None)
    assert "applies_to" not in load(seeded.eval_labels_path)["policies"]["exit"]


def test_a_hand_narrowed_applies_to_survives_rebuilding_the_labels(
    seeded: ArgusConfig, database: Database,
):
    """`--build-labels` proposes the scene; a human narrows it to the crop rows
    inside that scene. Merging must not undo the narrowing, for the same reason
    it must not undo a label."""
    database.upsert_camera(Camera(id="bay", name="Bay",
                                  source="/plant/Loading_Point.MOV", kind="file"))
    database.upsert_policy(Policy(
        id="cartons", kind="vlm", detector_key="vlm_prompt", label="Cartons",
        prompt="how many cartons?", lifecycle="pilot", scope=["bay"],
    ))
    build_labels(seeded, database, log=lambda *_: None)
    data = load(seeded.eval_labels_path)
    data["policies"]["cartons"]["applies_to"] = ["Loading-Point-*-bay"]
    save(seeded.eval_labels_path, data)

    build_labels(seeded, database, log=lambda *_: None)
    assert load(seeded.eval_labels_path)["policies"]["cartons"]["applies_to"] == [
        "Loading-Point-*-bay"
    ]


# --------------------------------------------------------------------------
# finding a customer's footage
# --------------------------------------------------------------------------


def test_footage_is_scanned_one_directory_down(tmp_path: Path):
    """Anmol's corpus arrived as footage/anmol-samples/ — six videos and a
    labelled image dataset beside them. A non-recursive scan found none of it,
    so --build-labels quietly produced a label file about the demo clip while
    the customer's plant sat one directory below it."""
    (tmp_path / "footage" / "anmol-samples").mkdir(parents=True)
    (tmp_path / "footage" / "loose.mp4").touch()
    (tmp_path / "footage" / "anmol-samples" / "Loading_Point.MOV").touch()
    (tmp_path / "footage" / "anmol-samples" / "notes.txt").touch()
    (tmp_path / "packing_section_recording.mp4").touch()

    found = [path.name for path in find_videos(tmp_path)]
    assert found[0] == "packing_section_recording.mp4"
    assert set(found[1:]) == {"loose.mp4", "Loading_Point.MOV"}


def test_the_scan_is_depth_capped(tmp_path: Path):
    """footage/ is also where an operator drops an NVR export, and an unbounded
    walk of a mounted share is a command that appears to hang."""
    deep = tmp_path / "footage"
    for level in range(MAX_SCAN_DEPTH + 2):
        deep = deep / f"level{level}"
    deep.mkdir(parents=True)
    (deep / "too-deep.mp4").touch()
    assert find_videos(tmp_path) == []


def test_a_symlinked_directory_is_not_followed(tmp_path: Path):
    """A footage directory is exactly where somebody symlinks a mount, and a
    link back up the tree would walk the same share twice before the depth cap
    ended it."""
    (tmp_path / "footage" / "real").mkdir(parents=True)
    (tmp_path / "footage" / "real" / "clip.mp4").touch()
    (tmp_path / "footage" / "link").symlink_to(tmp_path / "footage" / "real")
    assert [path.name for path in find_videos(tmp_path)] == ["clip.mp4"]


# --------------------------------------------------------------------------
# the plant policies, as committed
# --------------------------------------------------------------------------


def test_the_committed_labels_scope_every_plant_policy():
    """Four new prompt policies over 72 frames would be 288 cases if any of them
    were left unscoped; the ones that can be labelled are 46. This asserts the
    file that ships, not a fixture."""
    data = load(Path("evals/labels.yaml"))
    plant = [key for key in data["policies"] if key.startswith("anmol_")]
    assert len(plant) == 4
    for key in plant:
        patterns = data["policies"][key]["applies_to"]
        assert patterns and all(pattern.startswith("*-") for pattern in patterns), key
    # And the three that predate the plant fleet are scoped to the demo footage,
    # or they would be asked about 46 frames of somebody else's factory.
    for key in ("emergency_exit_blocked", "phone_usage_on_line", "person_near_the_conveyor"):
        assert data["policies"][key]["applies_to"], key


def test_every_plant_label_says_it_wants_auditing():
    """These verdicts were written by an agent looking at frames, not by a plant
    supervisor. `labelled_by` is what lets a reviewer tell those apart, and the
    ones here say so out loud."""
    data = load(Path("evals/labels.yaml"))
    plant = [
        entry for entry in data["frames"]
        if any(key.startswith("anmol_") for key in (entry.get("labels") or {}))
    ]
    assert len(plant) == 46
    for entry in plant:
        assert "AUDIT ME" in str(entry.get("labelled_by")), entry["id"]
        # A zone-scoped prompt policy is shown its zone, so the frame it is
        # scored on has to be that zone and not the wide shot it came from.
        assert entry.get("crop"), entry["id"]


def test_a_lopsided_triage_slice_names_the_reviewer(
    seeded: ArgusConfig, database: Database,
):
    """The slice's appeal is that a site's own verdicts are free ground truth.
    Its failure mode is that a soak test files verdicts too — on 2026-08-25 this
    repository's database held 161 reviewed events, 158 of them from one name in
    an alternating pattern, which would have been 96% of the slice."""
    write_frame(seeded.snapshots_dir / "soak.jpg")
    for index in range(20):
        event_id = database.insert_event(Event(
            camera_id="cam", policy_id="manpower_headcount",
            detector_name="manpower_headcount", detector_type="detector",
            severity="medium", stage="reviewed", ts=2000.0 + index,
            description="Zone 'Packing floor' over sanctioned limit: 5 people (limit 3)",
            snapshot_path="soak.jpg",
        ))
        database.record_review(
            event_id, "confirmed" if index % 2 else "false_positive", reviewer="rc",
        )
    notes = build_eval_set(seeded, database, slices=("triage",)).notes
    assert any("filed by 'rc'" in note for note in notes)
    # Reported, never filtered: no rule here can tell a script from a busy
    # supervisor, and silently dropping a site's labels is the worse failure.
    assert len(build_eval_set(seeded, database, slices=("triage",))) == 22


def test_a_balanced_triage_slice_says_nothing_extra(
    seeded: ArgusConfig, database: Database,
):
    write_frame(seeded.snapshots_dir / "many.jpg")
    for index in range(20):
        event_id = database.insert_event(Event(
            camera_id="cam", policy_id="manpower_headcount",
            detector_name="manpower_headcount", detector_type="detector",
            severity="medium", stage="reviewed", ts=3000.0 + index,
            description="Zone 'Packing floor' over sanctioned limit: 5 people (limit 3)",
            snapshot_path="many.jpg",
        ))
        database.record_review(event_id, "confirmed", reviewer=f"operator-{index % 5}")
    notes = build_eval_set(seeded, database, slices=("triage",)).notes
    assert not any("worth a look" in note for note in notes)


# --------------------------------------------------------------------------
# the resolution probe
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def probe_script():
    """`scripts/vlm_resolution_probe.py` as a module — loaded by path, like the
    trainer, because every heavy import in it is inside a function."""
    import importlib.util

    from argus.config import ROOT

    path = ROOT / "scripts" / "vlm_resolution_probe.py"
    spec = importlib.util.spec_from_file_location("argus_vlm_resolution_probe", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_a_crop_rectangle_becomes_the_polygon_zone_crop_wants(probe_script):
    assert probe_script.rectangle("0.28,0.0,0.6,0.5") == [
        [0.28, 0.0], [0.6, 0.0], [0.6, 0.5], [0.28, 0.5],
    ]


def test_a_malformed_crop_rectangle_exits_with_a_sentence(probe_script):
    with pytest.raises(SystemExit) as raised:
        probe_script.rectangle("0.28,0.0,0.6")
    assert "x1,y1,x2,y2" in str(raised.value)


def test_the_probe_reports_agreement_only_where_it_was_told_the_answer(probe_script):
    unknown = probe_script.Probe(
        frame="f", width=960, crop_px=(1, 1, 1, 1), cropped=True,
        triggered=True, confidence=0.9, reason="", latency_sec=1.0,
    )
    assert unknown.agrees is None
    told = probe_script.Probe(
        frame="f", width=960, crop_px=(1, 1, 1, 1), cropped=True,
        triggered=True, confidence=0.9, reason="", latency_sec=1.0, expected=False,
    )
    assert told.agrees is False


def test_the_policy_table_shows_the_label_balance_and_names_a_one_sided_slice():
    """A policy whose labelled frames are all violations scores 1.0 precision by
    doing nothing, and the number is a fact about the footage. The Anmol
    broken-biscuit clip is 70 seconds in which a reject pile is on the floor the
    whole time — a real condition, and an unmeasurable one."""
    cases = [
        EvalCase(id=f"c{i}", slice="policy", policy="overflow", prompt="?",
                 frame=FrameRef(path=Path("x.jpg")), expected=True)
        for i in range(6)
    ] + [
        EvalCase(id=f"m{i}", slice="policy", policy="mixed", prompt="?",
                 frame=FrameRef(path=Path("x.jpg")), expected=i % 2 == 0)
        for i in range(6)
    ]
    run = EvalRun(started_at="now", eval_set=EvalSet(cases=cases))
    results = [RunnerResult(case.id, MODEL, True, latency_sec=1.0) for case in cases]
    run.results[MODEL] = results
    run.reports.append(score(MODEL, MODEL, cases, results))
    body = render(run)
    assert "| should fire |" in body
    assert "`overflow` has no counter-example" in body
    assert "6 of 6 labelled frame(s) should fire" in body
    assert "`mixed` has no counter-example" not in body
