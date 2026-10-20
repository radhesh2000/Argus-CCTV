"""A training run must not hand back the weights it started from.

Found 2026-09-20, after `docs/STUDENT_DETECTOR.md` had already published a
verdict measured on one. rfdetr's `BestModelCheckpoint` gives the *regular*
track to PyTorch Lightning's `ModelCheckpoint`, which knows not to record a
best score during the pre-training sanity-check validation. Its own EMA
high-water mark has no such guard, so the sanity check — two batches, whichever
views sort first — sets it, writes `checkpoint_best_ema.pth` from the untrained
EMA copy, and if those two batches were easy no real epoch ever beats it.
`on_fit_end` then prefers EMA on a strict `>` and copies that file over
`checkpoint_best_total.pth`.

Nothing downstream can notice. The file is the right size, loads, has the right
head width, and detects people — it is stock COCO, and stock COCO detects
people. `scripts/train_detector.py:audit_best_total` is the check, and these
are the cases it has to get right, including the one where flagging would be
wrong.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch", reason="the audit compares weight tensors")

from scripts.train_detector import audit_best_total  # noqa: E402


def write(path, tensors: dict[str, torch.Tensor], *, epoch: int | None = None, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": tensors, "epoch": epoch, **extra}, path)
    return path


def weights(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "backbone.0.encoder.layer.0.attention.weight": torch.rand(8, 8, generator=generator),
        "backbone.0.encoder.layer.1.attention.weight": torch.rand(8, 8, generator=generator),
        "transformer.decoder.class_embed.bias": torch.rand(3, generator=generator),
    }


HISTORY = [
    {"epoch": 0.0, "val/ema_mAP_50_95": 0.45, "val/mAP_50_95": 0.44},
    {"epoch": 7.0, "val/ema_mAP_50_95": 0.50, "val/mAP_50_95": 0.49},
]


def test_the_pre_training_snapshot_is_caught_and_replaced(tmp_path):
    base = write(tmp_path / "base.pth", weights(1))
    pretrained, trained = weights(1), weights(2)
    write(tmp_path / "checkpoint_best_total.pth", pretrained, epoch=0)
    write(tmp_path / "checkpoint_best_regular.pth", trained, epoch=7)
    write(tmp_path / "checkpoint_best_ema.pth", pretrained, epoch=0)

    audit = audit_best_total(tmp_path, base, HISTORY)

    assert audit["repaired"] is True
    assert audit["replaced_with"] == "checkpoint_best_regular.pth"
    # The epoch the history says had the best EMA is 7, and the file claimed 0.
    assert audit["best_ema_epoch_in_history"] == 7.0
    restored = torch.load(tmp_path / "checkpoint_best_total.pth", weights_only=False)["model"]
    assert torch.equal(restored["backbone.0.encoder.layer.0.attention.weight"],
                       trained["backbone.0.encoder.layer.0.attention.weight"])


def test_the_untrained_ema_file_stops_looking_like_a_checkpoint(tmp_path):
    """It is evidence, so it is kept — but a file called `best_ema` sitting in
    a run directory is a thing somebody will eventually point a config line at."""
    base = write(tmp_path / "base.pth", weights(1))
    write(tmp_path / "checkpoint_best_total.pth", weights(1), epoch=0)
    write(tmp_path / "checkpoint_best_regular.pth", weights(2), epoch=7)
    write(tmp_path / "checkpoint_best_ema.pth", weights(1), epoch=0)

    audit_best_total(tmp_path, base, HISTORY)

    assert not (tmp_path / "checkpoint_best_ema.pth").exists()
    assert (tmp_path / "checkpoint_untrained_sanity_ema.pth.rejected").exists()


def test_a_genuinely_trained_best_total_is_left_alone(tmp_path):
    base = write(tmp_path / "base.pth", weights(1))
    trained = weights(2)
    write(tmp_path / "checkpoint_best_total.pth", trained, epoch=7)
    write(tmp_path / "checkpoint_best_regular.pth", trained, epoch=7)

    audit = audit_best_total(tmp_path, base, HISTORY)

    assert audit["repaired"] is False
    assert "differs from the pretrained" in audit["reason"]
    kept = torch.load(tmp_path / "checkpoint_best_total.pth", weights_only=False)["model"]
    assert torch.equal(kept["backbone.0.encoder.layer.0.attention.weight"],
                       trained["backbone.0.encoder.layer.0.attention.weight"])


def test_a_frozen_backbone_is_not_mistaken_for_an_unselected_run(tmp_path):
    """The false positive worth avoiding. If the backbone were frozen on
    purpose, every checkpoint would match the pretrained file — and swapping
    best_total for best_regular would change nothing while reporting a repair
    that did not happen. Requiring that the *regular* track moved is what makes
    the audit a statement about file selection."""
    base = write(tmp_path / "base.pth", weights(1))
    write(tmp_path / "checkpoint_best_total.pth", weights(1), epoch=7)
    write(tmp_path / "checkpoint_best_regular.pth", weights(1), epoch=7)

    audit = audit_best_total(tmp_path, base, HISTORY)

    assert audit["repaired"] is False
    assert "did not train in this run at all" in audit["reason"]


def test_a_run_with_no_base_checkpoint_recorded_says_it_could_not_check(tmp_path):
    """Silence and 'checked, fine' must not look the same in provenance.json."""
    write(tmp_path / "checkpoint_best_total.pth", weights(1), epoch=7)

    audit = audit_best_total(tmp_path, None, HISTORY)

    assert audit["checked"] is False
    assert audit["repaired"] is False


def test_the_audit_is_recorded_in_provenance_by_audit_only(tmp_path):
    from scripts.train_detector import audit_only

    base = write(tmp_path / "base.pth", weights(1))
    write(tmp_path / "checkpoint_best_total.pth", weights(1), epoch=0)
    write(tmp_path / "checkpoint_best_regular.pth", weights(2), epoch=7)
    (tmp_path / "provenance.json").write_text(json.dumps({
        "training": {"base_checkpoint": {"path": str(base)}, "history": HISTORY},
    }))

    assert audit_only(tmp_path) == 0

    provenance = json.loads((tmp_path / "provenance.json").read_text())
    assert provenance["best_total_audit"]["repaired"] is True
    # And the checkpoint block is re-derived, so the digest in provenance is
    # the digest of the file that is now on disk.
    assert provenance["checkpoints"]["checkpoint_best_total.pth"]["epoch"] == 7
