"""The optional fine-tuned detector: staging, provenance, and the refusal to guess.

Two halves, and neither of them trains anything — the GPU work lives in
``scripts/train_detector.py`` and is a lab activity, not a test.

**Staging** is arithmetic on text files: the corpus is YOLO ``class cx cy w h``
and RF-DETR wants a different directory layout and a different class column, so
the conversion is exactly the kind of thing that is silently wrong for a week.
The tests below use a three-frame fake corpus with known boxes.

**Provenance** is the gate in front of the weights. Every test in that half is a
way a checkpoint could be wrong that no tensor would reveal — the sidecar
missing, describing a different file, or claiming a person slot the head does
not have — and each asserts that the appliance refuses rather than starts.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from argus.config import ROOT, ArgusConfig, load_config
from argus.vision.checkpoint import CheckpointError, load_provenance

TRAIN_SCRIPT = ROOT / "scripts" / "train_detector.py"


@pytest.fixture(scope="module")
def trainer():
    """`scripts/train_detector.py` as a module.

    It is a script, not a package, and it deliberately imports nothing from
    `argus` — so it is loaded by path. Importing it is safe in this venv: every
    training import is inside a function, and the re-exec into `.venv-train`
    only runs under `__main__`.
    """
    spec = importlib.util.spec_from_file_location("argus_train_detector", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# staging the corpus
# --------------------------------------------------------------------------


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A miniature stand-in for the Anmol export: two train frames, one val.

    Images are 1x1 PNGs — nothing here reads a pixel, and a real 2560x1440 JPEG
    would make the test a disk benchmark.
    """
    from PIL import Image

    root = tmp_path / "corpus"
    boxes = {
        ("train", "Loading_Point_frame_01"): ["0 0.5 0.5 0.2 0.4"],
        ("train", "Broken_Biscuits_frame_02"): ["0 0.25 0.75 0.1 0.2", "0 0.9 0.1 0.05 0.1"],
        ("val", "Loading_Point_frame_09"): ["0 0.4 0.4 0.3 0.3"],
    }
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
    for (split, stem), lines in boxes.items():
        Image.new("RGB", (1, 1)).save(root / "images" / split / f"{stem}.jpg")
        (root / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    return root


def test_staging_produces_the_layout_rfdetr_validates(trainer, corpus, tmp_path):
    """`images/<split>` -> `<split>/images`, and `val` -> `valid`.

    Asserted against RF-DETR's own gate rather than against our idea of it: the
    layout is a contract with somebody else's loader.
    """
    pytest.importorskip("rfdetr")
    from rfdetr.datasets.yolo import is_valid_yolo_dataset

    staged = tmp_path / "staged"
    trainer.stage_dataset(corpus, staged)

    assert is_valid_yolo_dataset(str(staged))
    assert (staged / "train" / "images").is_dir()
    assert (staged / "valid" / "labels").is_dir()
    assert not (staged / "val").exists()


def test_staging_shifts_person_onto_cocos_class_slot(trainer, corpus, tmp_path):
    """The whole reason the appliance's inference path needs no change.

    Class `0` in the export becomes class `1` in the staged copy, because
    RF-DETR's COCO head is indexed by raw category id and person is id 1. Get
    this backwards and the fine-tune trains a head the engine never reads.
    """
    staged = tmp_path / "staged"
    trainer.stage_dataset(corpus, staged)

    written = (staged / "train" / "labels" / "Loading_Point_frame_01.txt").read_text().split()
    assert written[0] == str(trainer.PERSON_CLASS_INDEX) == "1"
    # The geometry is untouched — only the class column moves.
    assert [float(value) for value in written[1:]] == pytest.approx([0.5, 0.5, 0.2, 0.4])

    data = (staged / "data.yaml").read_text()
    assert '"__background__", "person"' in data or "'__background__', 'person'" in data
    assert "nc: 2" in data


def test_staging_symlinks_images_rather_than_copying_customer_footage(
    trainer, corpus, tmp_path
):
    staged = tmp_path / "staged"
    trainer.stage_dataset(corpus, staged)
    link = staged / "train" / "images" / "Loading_Point_frame_01.jpg"
    assert link.is_symlink()
    assert link.resolve() == (corpus / "images" / "train" / "Loading_Point_frame_01.jpg")


def test_manifest_counts_boxes_and_names_scenes(trainer, corpus, tmp_path):
    manifest = trainer.stage_dataset(corpus, tmp_path / "staged")
    assert manifest["splits"]["train"] == {
        "images": 2,
        "boxes": 3,
        "source_classes": {"0": 3},
        "digest": manifest["splits"]["train"]["digest"],
        "scenes": ["Broken_Biscuits", "Loading_Point"],
    }
    assert manifest["splits"]["val"]["images"] == 1
    assert manifest["label_class_shift"] == 1


def test_digest_identifies_the_corpus_not_the_staging_run(trainer, corpus, tmp_path):
    """Staging twice must digest the same; changing one label must not.

    The digest is what `provenance.json` offers as "these weights came from that
    data". A value that moved every run would prove nothing.
    """
    first = trainer.stage_dataset(corpus, tmp_path / "a")["digest"]
    second = trainer.stage_dataset(corpus, tmp_path / "b")["digest"]
    assert first == second

    label = corpus / "labels" / "val" / "Loading_Point_frame_09.txt"
    label.write_text("0 0.4 0.4 0.3 0.35\n")
    assert trainer.stage_dataset(corpus, tmp_path / "c")["digest"] != first


def test_an_unpaired_image_stops_the_run(trainer, corpus, tmp_path):
    """A label-less frame is a broken copy, not an empty room.

    Staged silently, it would teach the detector that a scene full of people
    contains none — the one training error that makes the model worse at exactly
    the job it has.
    """
    (corpus / "labels" / "train" / "Loading_Point_frame_01.txt").unlink()
    with pytest.raises(SystemExit) as exit_info:
        trainer.stage_dataset(corpus, tmp_path / "staged")
    assert "no label file" in str(exit_info.value)


def test_boxes_outside_the_unit_square_stop_the_run(trainer, tmp_path):
    """Un-normalized coordinates are a coordinate-system bug in the export, and
    they are silent: pixel values look like perfectly good floats."""
    label = tmp_path / "bad.txt"
    label.write_text("0 640.0 360.0 120.0 240.0\n")
    with pytest.raises(SystemExit) as exit_info:
        trainer.read_boxes(label)
    assert "not normalized" in str(exit_info.value)


def test_a_short_label_line_stops_the_run(trainer, tmp_path):
    label = tmp_path / "short.txt"
    label.write_text("0 0.5 0.5\n")
    with pytest.raises(SystemExit) as exit_info:
        trainer.read_boxes(label)
    assert "class cx cy w h" in str(exit_info.value)


def test_metric_history_keeps_only_the_rows_that_carry_a_score(trainer, tmp_path):
    """PTL logs `epoch` on every step and the val metrics on some.

    A train-only row read as an epoch with no score is how a run reports a best
    epoch it never had.
    """
    (tmp_path / "metrics.csv").write_text(
        "epoch,step,train/loss,val/mAP_50_95,val/mAP_50\n"
        "0,4,,0.31,0.64\n"
        "0,4,6.19,,\n"
        "1,9,,0.28,0.63\n"
    )
    rows = trainer.read_csv_metrics(tmp_path)
    assert [row["epoch"] for row in rows] == [0.0, 1.0]
    assert [row["val/mAP_50_95"] for row in rows] == [0.31, 0.28]


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """A training run directory, shaped like the real one but weighing 9 bytes."""
    directory = tmp_path / "finetuned" / "2026-08-24"
    directory.mkdir(parents=True)
    weights = directory / "checkpoint_best_total.pth"
    weights.write_bytes(b"not-a-pth")
    (directory / "provenance.json").write_text(json.dumps({
        "created_at": "2026-08-24T23:23:00+00:00",
        "variant": "rf-detr-nano",
        "resolution": 384,
        "person_class_index": 1,
        "class_names": ["__background__", "person"],
        "dataset": {
            "digest": "d" * 64,
            "splits": {
                "train": {"images": 63, "boxes": 840},
                "val": {"images": 16, "boxes": 175},
            },
        },
        "training": {
            "epochs_run": 60,
            "best": {"epoch": 59, "val/mAP_50": 0.7916, "val/mAP_50_95": 0.4075},
            "best_ema": {"epoch": 51, "val/ema_mAP_50": 0.7712,
                         "val/ema_mAP_50_95": 0.3990},
        },
        "checkpoints": {
            "checkpoint_best_total.pth": {
                "sha256": hashlib.sha256(b"not-a-pth").hexdigest(),
                "num_classes": 2,
                "head_width": 3,
                "best_total_source": "regular",
            },
        },
    }))
    return directory


def test_provenance_reads_what_an_operator_needs_to_know(run_dir):
    provenance = load_provenance(run_dir / "checkpoint_best_total.pth")
    assert provenance.person_class_index == 1
    assert provenance.num_classes == 2
    assert (provenance.dataset_images, provenance.dataset_boxes) == (79, 1015)
    assert provenance.epochs_run == 60
    assert provenance.best_epoch == 59
    assert provenance.val_map_50 == pytest.approx(0.7916)
    assert provenance.variant == "rf-detr-nano"


def test_the_scores_follow_the_track_the_weights_came_from(run_dir):
    """A run keeps two tracks — the live weights and their EMA — that peak at
    different epochs and score differently. `checkpoint_best_total.pth` is a copy
    of whichever won, so reading the regular track's numbers off an EMA winner
    quotes one model's result on another model's weights."""
    sidecar = run_dir / "provenance.json"
    document = json.loads(sidecar.read_text())

    assert load_provenance(run_dir / "checkpoint_best_total.pth").best_epoch == 59
    assert load_provenance(
        run_dir / "checkpoint_best_total.pth"
    ).val_map_50 == pytest.approx(0.7916)

    document["checkpoints"]["checkpoint_best_total.pth"]["best_total_source"] = "ema"
    sidecar.write_text(json.dumps(document))
    provenance = load_provenance(run_dir / "checkpoint_best_total.pth")
    assert provenance.best_epoch == 51
    assert provenance.val_map_50 == pytest.approx(0.7712)
    assert provenance.val_map_50_95 == pytest.approx(0.3990)


def test_a_checkpoints_own_epoch_wins_over_the_tracks(run_dir):
    """`checkpoint_best_regular.pth` records the epoch it was written at; only
    the copied `_total` file has to be told where it came from."""
    sidecar = run_dir / "provenance.json"
    document = json.loads(sidecar.read_text())
    weights = run_dir / "checkpoint_best_regular.pth"
    weights.write_bytes(b"regular")
    document["checkpoints"]["checkpoint_best_regular.pth"] = {
        "sha256": hashlib.sha256(b"regular").hexdigest(),
        "num_classes": 2, "head_width": 3, "epoch": 42,
    }
    sidecar.write_text(json.dumps(document))
    assert load_provenance(weights).best_epoch == 42


def test_summary_names_the_corpus_and_the_person_slot(run_dir):
    """The startup log line. It exists so that "which weights is this box
    running" is answered by reading the log, not by hashing a file."""
    summary = load_provenance(run_dir / "checkpoint_best_total.pth").summary()
    assert "checkpoint_best_total.pth" in summary
    assert "79 labelled frames/1015 boxes" in summary
    assert "person=logit 1" in summary
    assert "2026-08-24" in summary


def test_weights_without_a_sidecar_are_refused(tmp_path):
    orphan = tmp_path / "checkpoint_best_total.pth"
    orphan.write_bytes(b"weights")
    with pytest.raises(CheckpointError) as error:
        load_provenance(orphan)
    assert "provenance.json" in str(error.value)


def test_a_missing_file_says_so_rather_than_blaming_the_sidecar(tmp_path):
    with pytest.raises(CheckpointError, match="not found"):
        load_provenance(tmp_path / "nope.pth")


def test_a_sidecar_from_another_run_is_refused(run_dir):
    """The failure this catches: weights replaced in place, sidecar left behind.

    The name is checked before the digest because the message differs — "this
    describes a different file" is a different mistake from "this file changed".
    """
    other = run_dir / "checkpoint_best_ema.pth"
    other.write_bytes(b"different")
    with pytest.raises(CheckpointError) as error:
        load_provenance(other)
    assert "does not describe" in str(error.value)


def test_weights_that_no_longer_hash_to_the_record_are_refused(run_dir):
    weights = run_dir / "checkpoint_best_total.pth"
    weights.write_bytes(b"tampered")
    with pytest.raises(CheckpointError) as error:
        load_provenance(weights)
    assert "not it" in str(error.value)


def test_the_eval_may_skip_the_digest_check(run_dir):
    """`argus detector-eval --checkpoint` runs *before* anything is promoted, so
    it accepts a run directory the engine would not."""
    (run_dir / "checkpoint_best_total.pth").write_bytes(b"tampered")
    assert load_provenance(
        run_dir / "checkpoint_best_total.pth", verify_digest=False
    ).person_class_index == 1


@pytest.mark.parametrize(
    "mutation, expected",
    [
        ({"person_class_index": None}, "person_class_index"),
        ({"person_class_index": -1}, "person_class_index"),
        ({"person_class_index": "1"}, "person_class_index"),
        # 3 is the no-object slot of a 3-wide head, so a person there is never read.
        ({"person_class_index": 3}, "outside a head"),
    ],
)
def test_a_sidecar_that_cannot_say_which_logit_is_person_is_refused(
    run_dir, mutation, expected
):
    sidecar = run_dir / "provenance.json"
    document = json.loads(sidecar.read_text())
    document.update(mutation)
    sidecar.write_text(json.dumps(document))
    with pytest.raises(CheckpointError, match=expected):
        load_provenance(run_dir / "checkpoint_best_total.pth")


def test_a_sidecar_that_is_not_json_is_refused(run_dir):
    (run_dir / "provenance.json").write_text("{ not json")
    with pytest.raises(CheckpointError):
        load_provenance(run_dir / "checkpoint_best_total.pth")


# --------------------------------------------------------------------------
# config plumbing
# --------------------------------------------------------------------------


def test_absent_means_stock():
    """The default every appliance ships with, asserted rather than assumed."""
    assert ArgusConfig().detection.model.checkpoint == ""
    assert ArgusConfig().detector_checkpoint_path is None


def test_a_relative_checkpoint_resolves_against_the_repo_root():
    """So `models/finetuned/…` works from the systemd unit's working directory,
    which is not the one anybody typed the path in."""
    config = ArgusConfig()
    config.detection.model.checkpoint = "models/finetuned/2026-08-24/best.pth"
    assert config.detector_checkpoint_path == ROOT / "models/finetuned/2026-08-24/best.pth"


def test_an_absolute_checkpoint_is_left_alone(tmp_path):
    config = ArgusConfig()
    config.detection.model.checkpoint = str(tmp_path / "best.pth")
    assert config.detector_checkpoint_path == tmp_path / "best.pth"


def test_the_checkpoint_is_settable_from_yaml_and_from_the_environment(
    tmp_path, monkeypatch
):
    path = tmp_path / "argus.yaml"
    path.write_text("detection:\n  model:\n    checkpoint: from/yaml.pth\n")
    assert load_config(path).detection.model.checkpoint == "from/yaml.pth"

    monkeypatch.setenv("ARGUS_DETECTION_MODEL_CHECKPOINT", "from/env.pth")
    assert load_config(path).detection.model.checkpoint == "from/env.pth"


def test_detector_from_config_carries_the_checkpoint_without_touching_the_disk(
    tmp_path, monkeypatch
):
    """Construction must stay free of I/O — `argus status` builds a detector.

    The sidecar is read in `load()`, so pointing at a path that does not exist
    is fine until something asks for weights.
    """
    from argus.vision.detector import detector_from_config

    config = ArgusConfig(data_dir=tmp_path / "data")
    config.detection.model.checkpoint = str(tmp_path / "nowhere" / "best.pth")
    detector = detector_from_config(config)
    assert detector.checkpoint == tmp_path / "nowhere" / "best.pth"
    assert detector.person_class_index == 1     # the COCO default, until told otherwise
    assert detector.provenance is None


def test_loading_reads_the_sidecar_and_asks_rfdetr_for_those_weights(run_dir):
    """The two things `load()` does with a checkpoint, without a GPU.

    `num_classes` is deliberately absent from the kwargs: leaving it unset is
    what makes RF-DETR treat the checkpoint's own head width as authoritative
    instead of reinitialising the rows the fine-tune was spent on.
    """
    from argus.vision.detector import PersonDetector

    weights = run_dir / "checkpoint_best_total.pth"
    detector = PersonDetector(checkpoint=weights)
    assert detector._checkpoint_kwargs() == {"pretrain_weights": str(weights.resolve())}
    assert detector.person_class_index == 1
    assert detector.provenance is not None
    assert "num_classes" not in detector._checkpoint_kwargs()


def test_stock_weights_ask_rfdetr_for_nothing_at_all():
    from argus.vision.detector import PersonDetector

    detector = PersonDetector()
    assert detector._checkpoint_kwargs() == {}
    assert detector.provenance is None
    assert detector.weights_summary() == "rf-detr-nano stock COCO weights"


def test_a_head_narrower_than_the_person_slot_is_refused(run_dir):
    """The expensive failure, caught at load.

    Uncaught it is an IndexError inside inference, thousands of frames later, on
    a box nobody is watching.
    """
    from argus.vision.detector import PersonDetector

    class NarrowHead:
        class class_embed:
            out_features = 1

    detector = PersonDetector(checkpoint=run_dir / "checkpoint_best_total.pth")
    detector._checkpoint_kwargs()
    with pytest.raises(CheckpointError, match="1-wide class head"):
        detector._check_person_slot(NarrowHead)


def test_describe_says_which_weights_are_running(run_dir):
    """`describe()` feeds the engine banner, `argus status` and the API's system
    route. "stock" is the absence of the key, so none of them need changing."""
    from argus.vision.detector import PersonDetector

    stock = PersonDetector().describe()
    assert "checkpoint" not in stock
    assert stock["model"] == "rf-detr-nano"

    detector = PersonDetector(checkpoint=run_dir / "checkpoint_best_total.pth")
    detector._checkpoint_kwargs()
    described = detector.describe()
    assert described["model"] == "rf-detr-nano+finetune"
    assert described["checkpoint"]["person_class_index"] == 1
    assert described["checkpoint"]["dataset_digest"] == "d" * 64
    assert "fine-tune" in detector.weights_summary()


# --------------------------------------------------------------------------
# the decision rule
# --------------------------------------------------------------------------
#
# `_finetune_verdict` is the only place in this stage where a number becomes a
# recommendation, so it is the place worth pinning: two clauses, and the second
# one — no scene may go materially backwards — exists precisely because it is
# the clause a mean would let you skip.


def _condition(truth_per_scene: dict[str, int], found_per_scene: dict[str, int]):
    """A `ConditionResult` whose recall is exactly what the caller asked for.

    Predictions are synthesised as boxes sitting on the first `found` truth
    boxes of each scene, so the real matching code produces the intended recall
    rather than the test asserting against a stub.
    """
    import numpy as np

    from argus.eval.detector import ConditionResult, FrameResult

    result = ConditionResult(condition="engine", split="val", input_size="960x544")
    for scene, total in truth_per_scene.items():
        truth = np.array(
            [[index * 20.0, 0.0, index * 20.0 + 10.0, 40.0] for index in range(total)],
            dtype=np.float32,
        ).reshape(-1, 4)
        found = found_per_scene.get(scene, 0)
        result.frames.append(FrameResult(
            stem=f"{scene}_frame_00",
            scene=scene,
            truth=truth,
            boxes=truth[:found].copy(),
            scores=np.full(found, 0.9, dtype=np.float32),
            truth_heights=np.full(total, 100.0, dtype=np.float32),
        ))
    return result


def _paired_runs(before: dict[str, int], after: dict[str, int], truth: dict[str, int]):
    from argus.eval.detector import DetectorEvalRun

    baseline = DetectorEvalRun(dataset=Path("corpus"), detector={"model": "rf-detr-nano"})
    baseline.results.append(_condition(truth, before))
    tuned = DetectorEvalRun(dataset=Path("corpus"), detector={"model": "rf-detr-nano+finetune"})
    tuned.results.append(_condition(truth, after))
    baseline.finetuned = tuned
    return baseline


def test_a_material_gain_with_no_regression_ships_as_an_option():
    from argus.eval.detector_report import _finetune

    run = _paired_runs(
        truth={"Loading_Point": 100, "Packing3": 100},
        before={"Loading_Point": 40, "Packing3": 40},
        after={"Loading_Point": 60, "Packing3": 55},
    )
    section = _finetune(run)
    assert "ship as an option" in section
    assert "detection.model.checkpoint" in section
    assert "no scene of 10+ boxes lost 10 points or more" in section
    # The cost is stated in the same breath as the gain, not left in a table.
    assert "presence precision" in section
    assert "false positives" in section


def test_a_room_traded_for_the_mean_does_not_ship():
    """The mean rises by 12 points while one scene loses 25. That is a zone that
    stopped counting people it used to count, and it is not a trade this product
    makes."""
    from argus.eval.detector_report import _finetune

    run = _paired_runs(
        truth={"Loading_Point": 100, "Packing3": 100},
        before={"Loading_Point": 40, "Packing3": 40},
        after={"Loading_Point": 15, "Packing3": 90},
    )
    section = _finetune(run)
    assert "do not ship" in section
    assert "Loading_Point" in section.split("### The decision")[1]


def test_a_scene_too_small_to_resolve_the_bar_neither_passes_nor_fails():
    """The amendment, and the thing it must not become.

    An 8-box scene moves in 12.5-point steps, so *any* single missed person
    clears a 10-point bar there. Such a scene must not veto the run — and must
    not be quietly counted as a pass either, or the guard becomes a way to
    launder regressions by splitting scenes small.
    """
    from argus.eval.detector_report import _finetune

    run = _paired_runs(
        truth={"Loading_Point": 8, "Packing7": 50},
        before={"Loading_Point": 7, "Packing7": 16},
        after={"Loading_Point": 6, "Packing7": 40},
    )
    section = _finetune(run)
    assert "ship as an option" in section
    assert "below the resolution of the bar" in section
    assert "`Loading_Point` 87.5% -> 75.0% (8 boxes)" in section


def test_a_big_enough_scene_still_vetoes():
    """The guard is a size test, not an escape hatch: the same 12.5-point drop
    on a scene four times larger still stops the run."""
    from argus.eval.detector_report import _finetune

    run = _paired_runs(
        truth={"Loading_Point": 32, "Packing7": 50},
        before={"Loading_Point": 28, "Packing7": 16},
        after={"Loading_Point": 24, "Packing7": 40},
    )
    section = _finetune(run)
    assert "do not ship" in section
    assert "below the resolution of the bar" not in section


def test_a_gain_below_the_bar_does_not_ship():
    from argus.eval.detector_report import _finetune

    run = _paired_runs(
        truth={"Loading_Point": 100, "Packing3": 100},
        before={"Loading_Point": 40, "Packing3": 40},
        after={"Loading_Point": 42, "Packing3": 42},
    )
    section = _finetune(run)
    assert "not material" in section


def test_the_bar_is_stated_in_the_report_not_only_in_the_code():
    """A threshold a reader cannot see is a threshold chosen after the fact."""
    from argus.eval.detector_report import (
        MATERIAL_RECALL_GAIN, SCENE_REGRESSION_DELTA, _finetune,
    )

    section = _finetune(_paired_runs(
        truth={"Loading_Point": 100},
        before={"Loading_Point": 40},
        after={"Loading_Point": 60},
    ))
    assert f"at least {MATERIAL_RECALL_GAIN * 100:.0f} points" in section
    assert f"more\nthan {SCENE_REGRESSION_DELTA * 100:.0f}" in section


def test_a_run_without_a_checkpoint_adds_no_section():
    """The report is one document whether or not anybody fine-tuned anything."""
    from argus.eval.detector_report import _finetune
    from argus.eval.detector import DetectorEvalRun

    assert _finetune(DetectorEvalRun(dataset=Path("corpus"), detector={})) == ""


# --------------------------------------------------------------------------
# variant and resolution
# --------------------------------------------------------------------------
#
# Added when the trainer grew `--variant` / `--resolution` for the RF-DETR
# Medium @ 576 student. Two things could go silently wrong and neither shows up
# in a loss curve: a licence-encumbered variant becoming reachable, and a
# checkpoint trained at one resolution being served at another.


def test_only_the_apache_variants_are_reachable(trainer):
    """XL and 2XL are Roboflow PML, not Apache-2.0 (docs/DETECTION_V2.md §6).

    They are absent from the choice list, so `--variant xl` is an argparse
    error rather than a download. A licence is not something anybody should be
    able to violate by passing a flag.
    """
    assert set(trainer.VARIANTS) == {"nano", "small", "medium", "large"}
    assert set(trainer.NATIVE_RESOLUTION) == set(trainer.VARIANTS)

    with pytest.raises(SystemExit):
        trainer.parse_args(["--variant", "xl"])
    with pytest.raises(SystemExit):
        trainer.parse_args(["--variant", "2xl"])


def test_resolution_defaults_to_the_variants_own_input(trainer):
    assert trainer.parse_args(["--variant", "medium"]).resolution == 576
    assert trainer.parse_args(["--variant", "nano"]).resolution == 384
    assert trainer.parse_args(["--variant", "medium", "--resolution", "512"]).resolution == 512


def test_the_run_directory_names_the_model_it_holds(trainer, tmp_path):
    """`models/finetuned/2026-09-20` says nothing about what is inside it.

    Two runs on one day at two resolutions would land in the same directory and
    the second would overwrite the first's weights while keeping a provenance
    that still described them.
    """
    args = trainer.parse_args(["--variant", "medium", "--resolution", "576"])
    assert f"{args.variant}{args.resolution}" == "medium576"


def test_a_medium_576_sidecar_is_something_the_appliance_would_load(tmp_path):
    """The schema, not the weights: `argus/vision/checkpoint.py` must accept it.

    The gate reads `variant` and `resolution` straight out of the sidecar and
    quotes them in the startup banner, so a trainer that stopped writing them —
    or wrote the nano's — would put a model into production under another
    model's name.
    """
    directory = tmp_path / "2026-09-20-medium576-harmonized"
    directory.mkdir(parents=True)
    weights = directory / "checkpoint_best_total.pth"
    weights.write_bytes(b"not-a-pth")
    (directory / "provenance.json").write_text(json.dumps({
        "created_at": "2026-09-20T23:00:00+00:00",
        "variant": "rf-detr-medium",
        "resolution": 576,
        "person_class_index": 1,
        "class_names": ["__background__", "person"],
        "dataset": {"digest": "e" * 64,
                    "splits": {"train": {"images": 1841, "boxes": 9589},
                               "val": {"images": 112, "boxes": 427}}},
        "training": {"epochs_run": 12,
                     "best": {"epoch": 9, "val/mAP_50": 0.9, "val/mAP_50_95": 0.5}},
        "checkpoints": {"checkpoint_best_total.pth": {
            "sha256": hashlib.sha256(b"not-a-pth").hexdigest(),
            "num_classes": 2, "head_width": 3, "best_total_source": "regular",
        }},
    }))

    provenance = load_provenance(weights)
    assert provenance.variant == "rf-detr-medium"
    assert provenance.resolution == 576
    assert provenance.person_class_index == 1
    assert (provenance.dataset_images, provenance.dataset_boxes) == (1953, 10016)
