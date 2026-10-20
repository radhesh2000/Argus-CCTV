"""``argus`` console entry point.

Subcommands are thin: they parse arguments, load config, open the database and
hand off. Anything with logic lives in a module so it stays testable without a
subprocess.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from argus import __version__
from argus.config import ROOT, ConfigError, EvalConfig, load_config
from argus.db import Database
# The slice names, for `--slice`'s choices. `argus.eval.cases` is a dataclass
# module — it pulls in nothing the CLI does not already have, and crucially not
# `argus.eval.openrouter`, which is imported only behind the cloud gate.
from argus.eval.cases import SLICES
# Likewise a constants-only module: no torch, no CUDA, no corpus on disk.
from argus.eval.detector import CONDITIONS as DETECTOR_CONDITIONS

_EPILOG = """\
examples:
  argus doctor                   is this box a working appliance? (exits non-zero if not)
  argus install-services         write + enable the systemd user units
  argus user add ops --role ADMIN   create the first account (do this on install)
  argus seed                     import config/seed.yaml into the database
  argus run --duration 30        bring up gateway + decoders for 30s
  argus api                      serve the JSON API (docs at /api/docs)
  argus retention --dry-run      show what storage is being used, delete nothing
  argus backup now               snapshot both SQLite files, prune old ones
  argus backup restore latest    put the newest snapshot back (engine stopped)
  argus processing               is the GPU plane running, and each service's state
  argus processing pause         stop the engine + VLM (the console does this too)
  argus vlm start                bring up the local VLM server for prompt policies
  argus vlm health               is the VLM up, and which policies run against it
  argus index backfill           embed existing event snapshots for search
  argus index status             how much footage is searchable, and by which model
  argus eval --build-labels      sample frames into evals/labels.yaml for a human
  argus eval                     score the local VLM on this box's own labels
  argus detector-eval            score the shipped detector against labelled boxes
  argus status                   show cameras, policies and gateway state
"""


def _load(args) -> tuple:
    config = load_config(args.config)
    config.ensure_dirs()
    database = Database(config.db_path)
    applied = database.migrate()
    if applied:
        print(f"database: applied migration(s) {applied} -> {config.db_path}")
    return config, database


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------


def cmd_seed(args) -> int:
    from argus.seed import SeedError, seed

    config, database = _load(args)
    try:
        result = seed(database, args.file)
    except SeedError as exc:
        print(f"seed failed: {exc}", file=sys.stderr)
        return 1
    print(f"seeded {result} into {config.db_path}")
    # Printed, always, and before the counts: a seed file that quietly imported
    # two of eight cameras looks exactly like one that imported everything, and
    # the difference is a plant nobody is watching.
    for note in result.notes:
        print(f"  skipped: {note}")
    counts = database.counts()
    print(f"database now holds: {counts['cameras']} cameras, {counts['zones']} zones, "
          f"{counts['policies']} policies, {counts['events']} events")
    return 0


def cmd_run(args) -> int:
    from argus.engine import Engine

    config, database = _load(args)
    engine = Engine(
        config, database,
        use_gateway=not args.no_gateway,
        direct=args.direct,
        record=not args.no_record,
    )
    try:
        return engine.run(duration=args.duration, report_every=args.report_every)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"engine: {exc}", file=sys.stderr)
        return 1


def cmd_api(args) -> int:
    from argus.accounts import bootstrap_hint
    from argus.api.app import run

    config, database = _load(args)
    host = args.host or config.api.host
    port = args.port or config.api.port
    print(f"argus api: http://{host}:{port}  (docs at /api/docs, health at /api/health)")
    # Starting is not refused, because the box being reachable is how somebody
    # discovers it needs an account — but it is said out loud, at the top of the
    # log, rather than left for an operator to deduce from a console that
    # answers 401 to everything.
    hint = bootstrap_hint(database)
    if hint:
        print(f"argus api: WARNING — {hint}", file=sys.stderr)
        print("argus api: every route except /api/health will answer 401 until then.",
              file=sys.stderr)
    run(config, database, host=host, port=port)
    return 0


def _amount(value: float) -> str:
    """A number with as much precision as it has, and no more.

    The retention plan prints quantities spanning five orders of magnitude in
    one table — 300 seconds of ring expressed in days (0.0035) next to 100 GB of
    clips — and a fixed format either rounds the ring window to ``0d`` or pads
    every clip figure with dead zeros. Neither is a readable plan.
    """
    if value >= 10 or value == int(value):
        return f"{value:.0f}"
    return f"{value:.4f}".rstrip("0").rstrip(".") if value < 0.01 else f"{value:.2f}"


def cmd_retention(args) -> int:
    """One-shot housekeeping — the same tick `argus run` performs on a timer,
    exposed on its own so an operator can reclaim disk without the engine.

    "The same tick" is the point: it sweeps, rotates the supervised children's
    logs, and takes a backup if one is owed. A command that did only the
    deleting would quietly be a different job from the one running in
    production, which is how the two drift apart.
    """
    from argus.health import disk_info
    from argus.retention import Retention

    config, database = _load(args)
    retention = Retention(config, database)
    if args.dry_run:
        info = disk_info(config)
        print(f"disk: {info['percent']}% full ({info['free_gb']} GB free), "
              f"Argus holds {info['argus_mb']} MB")
        for name, megabytes in sorted(info["by_directory_mb"].items()):
            print(f"  {name:<12} {megabytes:>10.1f} MB")

        # The plan, not the windows. Printing `clips_days: 14` was true and
        # useless on a box where the actual answer was "1 day for shadow, 30 for
        # confirmed, and 480 GB of it goes to the budget regardless" — an
        # operator about to reclaim half a terabyte deserves the arithmetic.
        print()
        plan = retention.plan()
        for media in plan["media"]:
            budget = (f", budget {_amount(media['budget_gb'])} GB"
                      if media["budget_gb"] else ", no budget")
            print(f"{media['kind']}: {_amount(media['used_gb'])} GB{budget}"
                  + ("  OVER BUDGET" if media["over_budget"] else ""))
            if media["tiered"]:
                for tier, bucket in media["by_tier"].items():
                    print(f"  {tier:<16} {_amount(bucket['days']):>7}d  "
                          f"{bucket['files']:>7} file(s)  {_amount(bucket['gb']):>8} GB  "
                          f"{bucket['expiring']:>7} past its window")
            elif media["by_tier"]:
                total = sum(b["files"] for b in media["by_tier"].values())
                expiring = sum(b["expiring"] for b in media["by_tier"].values())
                print(f"  flat {_amount(media['flat_days'])}d  {total:>7} file(s)  "
                      f"{expiring:>7} past its window")
            if media["evict_files"]:
                print(f"  budget eviction: {media['evict_files']} file(s), "
                      f"{_amount(media['evict_gb'])} GB, lowest-stage-first")
        print(f"\nrows: events {plan['rows']['events_days']}d, "
              f"occupancy {plan['rows']['occupancy_days']}d, "
              f"index samples {config.retention.index_days}d")
        print(f"guards: disk {plan['max_disk_percent']}%, "
              f"nothing younger than {plan['min_keep_hours']}h is ever deleted")
        from argus import backup as backup_module

        latest = backup_module.newest(config)
        print(f"backup:  every {config.backup.interval_hours}h, keeping "
              f"{config.backup.keep}, in {config.backups_dir}"
              + (f" — newest {latest.stamp} ({latest.mb} MB)" if latest else " — none yet"))
        return 0
    retention.housekeeping()
    return 0


def cmd_backup(args) -> int:
    """Snapshots of the two SQLite files, and the way back from a bad day."""
    from argus import backup

    # Deliberately not `_load`: that migrates, and migrating the database you
    # are about to restore over is at best noise and at worst the thing you were
    # trying to undo. The restore path migrates afterwards, on the file that
    # ends up in place.
    config = load_config(args.config)
    config.ensure_dirs()

    if args.action == "list":
        entries = backup.list_backups(config)
        if not entries:
            print(f"no backups in {config.backups_dir}")
            return 0
        print(f"{'STAMP':<22} {'AGE':>9} {'SIZE':>9}  SCHEMA  FILES")
        for entry in entries:
            age = entry.age_sec()
            age_text = f"{age / 3600:.1f} h" if age is not None else "?"
            if not entry.complete:
                print(f"{entry.stamp:<22} {age_text:>9} {'':>9}  INCOMPLETE — {entry.error}")
                continue
            names = ", ".join(f.name for f in entry.files)
            print(f"{entry.stamp:<22} {age_text:>9} {entry.mb:>7.1f} MB  "
                  f"v{entry.schema_version:<5}  {names}")
        print(f"\nin {config.backups_dir}, keeping {config.backup.keep}, "
              f"every {config.backup.interval_hours}h")
        return 0

    if args.action == "verify":
        try:
            entry = backup.find(config, args.stamp or "latest")
        except backup.BackupError as exc:
            print(f"verify: {exc}", file=sys.stderr)
            return 1
        problems = backup.verify(entry)
        if problems:
            print(f"{entry.stamp}: DAMAGED", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            return 1
        print(f"{entry.stamp}: intact — {len(entry.files)} file(s), {entry.mb} MB, "
              f"sha256 verified")
        return 0

    if args.action == "now":
        try:
            entry = backup.run(config, reason=args.reason)
        except backup.BackupError as exc:
            print(f"backup: {exc}", file=sys.stderr)
            return 1
        for record in entry.files:
            print(f"  {record.name:<20} {record.mb:>8.1f} MB  {record.sha256[:16]}")
        removed = backup.prune(config)
        print(f"backup {entry.stamp} -> {entry.path} in {entry.duration_sec:.1f}s"
              + (f" (pruned {len(removed)})" if removed else ""))
        return 0

    # restore
    try:
        result = backup.restore(config, args.stamp or "latest", force=args.force)
    except backup.BackupError as exc:
        print(f"restore: {exc}", file=sys.stderr)
        return 1
    print(f"restored {result['stamp']} (taken {result['created_at']}, "
          f"schema v{result['from_schema']})")
    for item in result["restored"]:
        print(f"  {item['name']:<20} {item['mb']:>8.1f} MB -> {item['path']}")
        for kept in item["kept_aside"]:
            print(f"{'':<22} previous kept as {kept}")
    if result["migrations_applied"]:
        print(f"  applied migration(s) {result['migrations_applied']} "
              f"-> schema v{result['schema_version']}")
    print("nothing was deleted; remove the .pre-restore-* files once you are satisfied.")
    return 0


def cmd_doctor(args) -> int:
    """Everything that has to be true for this box to be an appliance."""
    import json as json_module

    from argus import doctor

    config = load_config(args.config)
    config.ensure_dirs()
    database = Database(config.db_path)
    checks, elapsed = doctor.report(config, database)
    if args.json:
        print(json_module.dumps({
            "checks": [c.as_dict() for c in checks],
            "summary": doctor.summarise(checks),
            "elapsed_sec": round(elapsed, 2),
        }, indent=2))
    else:
        print(doctor.format_report(checks, config, elapsed=elapsed))
    return doctor.exit_code(checks, strict=args.strict)


def cmd_install_services(args) -> int:
    """Write and enable the systemd **user** units for this checkout."""
    from argus import services

    config = load_config(args.config)
    config.ensure_dirs()

    if args.uninstall:
        result = services.uninstall(dry_run=args.dry_run)
        for path in result["removed"]:
            print(f"removed {path}")
        for path in result["kept"]:
            print(f"kept {path} (not generated by argus)")
        if not result["removed"] and not result["kept"]:
            print(f"nothing to remove in {result['directory']}")
        return 0

    try:
        result = services.install(
            config, with_vlm=args.vlm, enable=not args.no_enable,
            start=args.start, dry_run=args.dry_run,
        )
    except services.ServiceError as exc:
        print(f"install-services: {exc}", file=sys.stderr)
        return 1

    for unit in result["units"]:
        print(f"{unit['action']:<9} {unit['path']}")
    for command in result["commands"]:
        print(f"{'ran' if not args.dry_run else 'would run':<9} {command}")
    if result["enabled"]:
        print(f"enabled:  {', '.join(result['enabled'])}")
    if not args.vlm:
        print(f"note:     {services.VLM_UNIT} was written but NOT enabled — the VLM is "
              f"optional equipment.\n"
              f"          systemctl --user enable --now {services.VLM_UNIT}")
    if result.get("linger") is False:
        print("\nWARNING: linger is off — every service here stops when this user logs out.")
        print(f"         {result['linger_command']}")
    print("\n  systemctl --user status argus-engine argus-api")
    print("  journalctl --user -u argus-engine -f")
    return 0


def cmd_vlm(args) -> int:
    """Manage the local VLM inference server (Phase 2).

    Deliberately a separate subcommand rather than something `argus run` starts:
    the model takes minutes to load and the engine must be restartable in
    seconds, so their lifecycles are not the same lifecycle.
    """
    from argus.vlm import server as vlm_server

    config = load_config(args.config)
    config.ensure_dirs()

    if args.action == "start":
        try:
            result = vlm_server.start(config, wait=not args.no_wait)
        except vlm_server.VlmServerError as exc:
            print(f"vlm: {exc}", file=sys.stderr)
            return 1
        if result.get("already_running"):
            print(f"vlm: already serving {result.get('model')} at {result['endpoint']}")
            return 0
        print(f"vlm: started pid {result['pid']}, logging to {result['log']}")
        print(f"     {result['command']}")
        if args.no_wait:
            return 0
        if result.get("available"):
            print(f"vlm: ready — {result.get('model')} at {result['endpoint']} "
                  f"(probe {result.get('probe_sec')}s)")
            return 0
        print(f"vlm: NOT ready — {result.get('reason')}", file=sys.stderr)
        print(f"     see {result['log']}", file=sys.stderr)
        return 1

    if args.action == "stop":
        result = vlm_server.stop(config)
        if result.get("stopped"):
            forced = " (SIGKILL)" if result.get("forced") else ""
            print(f"vlm: stopped pid {result['pid']}{forced}")
            return 0
        print(f"vlm: not stopped — {result.get('reason')}")
        return 1

    # health
    state = vlm_server.status(config)
    print(f"endpoint:  {state['endpoint']}")
    print(f"model:     {state['configured_model']}")
    print(f"weights:   {state['model_path']} "
          f"({'present' if state['model_ready'] else 'MISSING — run scripts/fetch_vlm.sh'})")
    print(f"venv:      {state['venv']} "
          f"({'present' if state['venv_ready'] else 'MISSING — run scripts/setup_vlm_server.sh'})")
    print(f"pid:       {state['pid'] or '(not started by argus)'}")
    if state.get("available"):
        print(f"serving:   {', '.join(state.get('models') or [])} "
              f"(probe {state.get('probe_sec')}s)")
    else:
        print(f"serving:   UNAVAILABLE — {state.get('reason')}")

    verify = config.vlm.verify
    print(f"verify:    {'on' if verify.enabled else 'off'}"
          + (f" for stages {', '.join(verify.stage_list)}" if verify.enabled else ""))

    database = Database(config.db_path)
    database.migrate()
    prompts = [p for p in database.list_policies(kind="vlm", enabled_only=True, active_only=True)
               if p.get("prompt")]
    print("policies:")
    for policy in prompts:
        print(f"  {policy['id']:<28} every {policy['interval_sec'] or '?'}s, "
              f"cooldown {policy['cooldown_sec']}s, stage={policy['stage']}")
    if not prompts:
        print("  (no active vlm policies — run `argus seed`)")
    return 0 if state.get("available") else 1


def cmd_processing(args) -> int:
    """The appliance's pause switch — the same durable flag the console flips.

    ``gate`` is the one the systemd units call as their ``ExecCondition``: it
    exits 0 when processing is enabled and 1 when it is paused, and a 1 is what
    makes a rebooted-but-paused appliance come up with the GPU idle. It is silent
    on the enabled path and leaves a one-line breadcrumb in the journal on the
    paused one, so ``journalctl -u argus-engine`` explains why the engine did not
    start rather than looking broken.
    """
    from argus import processing

    if args.action == "gate":
        # Fully self-contained and defensive: only a cleanly-read "disabled"
        # skips the start (exit 1). Anything else — a bad config, an unreadable
        # flag, an unexpected error — proceeds (exit 0), so the real unit starts
        # and surfaces the problem loudly instead of this masking it as a pause.
        try:
            config = load_config(args.config)
            enabled = processing.is_enabled(config)
        except Exception:  # noqa: BLE001 - a gate that raises hangs boot
            return 0
        if enabled:
            return 0
        print("argus processing: paused — engine/VLM start skipped "
              f"({processing.state_path(config)}); resume from the console or "
              "`argus processing resume`", file=sys.stderr)
        return 1

    config = load_config(args.config)
    config.ensure_dirs()

    if args.action in ("resume", "pause"):
        enabled = args.action == "resume"
        result = processing.set_processing(
            config, enabled, actor="cli",
            note=f"argus processing {args.action}",
        )
        print(f"processing: {'ENABLED' if enabled else 'PAUSED'} "
              f"-> {processing.state_path(config)}")
        if result["command"]:
            print(f"  {'ran' if result['actuated'] else 'attempted'}: {result['command']}")
        if result["detail"]:
            print(f"  note: {result['detail']}", file=sys.stderr)
        if not result["actuated"] and not result["command"]:
            print(f"  {result['detail']}", file=sys.stderr)
        return 0

    # status (default)
    from argus.health import read_engine_stats

    stats = read_engine_stats(config)
    vlm = stats.get("vlm") or {}
    report = processing.processing_report(
        config,
        engine_running=bool(stats.get("running")),
        engine_uptime_sec=stats.get("uptime_sec"),
        vlm_available=bool(vlm.get("available")) if stats.get("running") else False,
    )
    flag = "ENABLED" if report[processing.FLAG_KEY] else "PAUSED"
    print(f"processing: {flag}  (status: {report['status']})")
    print(f"flag file:  {processing.state_path(config)}")
    if report["updated_at"]:
        print(f"changed:    {report['updated_at']} by {report['updated_by'] or 'unknown'}")
    print(f"systemctl:  {'available' if report['systemctl'] else 'NOT available on this box'}")
    for name, svc in report["services"].items():
        detail = f"{svc['active_state'] or '?'}/{svc['sub_state'] or ''}".rstrip("/")
        print(f"  {name:<7} {svc['state']:<9} ({detail})")
    return 0


def cmd_index(args) -> int:
    """Manage the semantic index (Phase 3).

    ``backfill`` is the one command that matters here: it makes the evidence
    already on this box searchable, which is what turning the feature on has to
    mean if it is not going to look like it lost the plant's history.
    """
    from argus.index.store import IndexError_

    config, database = _load(args)

    if args.action == "backfill":
        from argus.index.backfill import backfill_events

        try:
            result = backfill_events(
                config, database, reset=args.reset, limit=args.limit,
            )
        except (FileNotFoundError, IndexError_) as exc:
            print(f"index: {exc}", file=sys.stderr)
            return 1
        print(result)
        for error in result.errors:
            print(f"  {error}", file=sys.stderr)
        return 1 if result.failed and not result.indexed else 0

    # status
    from argus.index.store import VectorStore

    model_path = config.index_model_path
    ready = config.index.model == "fake" or (model_path / "config.json").is_file()
    print(f"enabled:   {config.index.enabled}")
    print(f"model:     {config.index.model}")
    print(f"weights:   {model_path} "
          f"({'present' if ready else 'MISSING — run scripts/fetch_siglip.sh'})")
    print(f"store:     {config.index_db_path}")
    print(f"sampling:  every {config.index.sample_interval_sec}s per camera, "
          f"batch {config.index.batch_size}, queue {config.index.queue_depth}")
    print(f"queries:   text tower on {config.index.query_device}, "
          f"floor {config.index.min_score}")
    print(f"retention: index samples {config.retention.index_days}d")

    if not config.index_db_path.exists():
        print("index:     empty — nothing indexed yet "
              "(run `argus run`, or `argus index backfill` for existing evidence)")
        return 0 if ready else 1
    try:
        meta = VectorStore(config.index_db_path, dim=0, model="").meta()
        store = VectorStore(config.index_db_path, dim=int(meta.get("dim", 0) or 0),
                            model=str(meta.get("model", "")))
        stats = store.stats()
    except IndexError_ as exc:
        print(f"index:     UNREADABLE — {exc}", file=sys.stderr)
        return 1
    print(f"index:     {stats['frames']} frames ({stats['events']} event, "
          f"{stats['samples']} sample) across {stats['cameras']} camera(s), "
          f"{stats['span_hours']}h span, {stats['size_mb']} MB")
    if stats["model"] != config.index.model and config.index.model != "fake":
        print(f"           WARNING: written by {stats['model']}, configured for "
              f"{config.index.model} — run `argus index backfill --reset`", file=sys.stderr)
    for row in stats.get("by_camera") or []:
        print(f"  {row['camera_id']:<28} {row['frames']:>7} frames")
    return 0


def cmd_detector_eval(args) -> int:
    """Score the shipped person detector against a labelled corpus.

    The sibling of `argus eval`, one layer down: that one asks whether a VLM can
    be trusted to verify a claim, this asks whether the detector underneath it
    finds the people at all. Both exist as shipped commands for the same reason
    — the question is asked *at* a site, about that site's cameras, and a number
    from somebody else's benchmark is not an answer.
    """
    import json

    from argus.eval.detector import (
        CONDITIONS, ENGINE_THRESHOLD, DetectorEvalError, run_detector_eval,
    )
    from argus.eval.detector_report import render
    from argus.vision.checkpoint import CheckpointError

    config = load_config(args.config)
    config.ensure_dirs()

    conditions = tuple(args.condition) if args.condition else CONDITIONS
    splits = tuple(args.split) if args.split else ("val", "train")

    try:
        run = run_detector_eval(
            config,
            dataset=args.dataset,
            splits=splits,
            conditions=conditions,
            checkpoint=args.checkpoint,
            progress=(lambda line: print(line, flush=True)) if not args.quiet else None,
        )
    except DetectorEvalError as exc:
        # Exit 2, not 1: "the corpus is not on this box" is a different thing
        # from "the detector scored badly", and a CI step should be able to tell
        # them apart without parsing the message.
        print(f"detector-eval: {exc}", file=sys.stderr)
        return 2
    except CheckpointError as exc:
        # 2 as well, and for the same reason: an unusable checkpoint is a fact
        # about this box's filesystem, not a verdict on any detector.
        print(f"detector-eval: {exc}", file=sys.stderr)
        return 2

    report_path = _eval_path(args.report) or ROOT / "docs" / "DETECTOR_EVAL.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render(run))

    json_path = _eval_path(args.json)
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(run.as_dict(), indent=2))

    for label, scored in (("", run), ("finetuned ", run.finetuned)):
        if scored is None:
            continue
        for result in scored.results:
            strict = result.point(ENGINE_THRESHOLD)
            centre = result.point(ENGINE_THRESHOLD, criterion="centre")
            print(f"{label}{result.split:>5} / {result.condition:<19} AP@50 "
                  f"{(result.ap() or 0.0):.3f}  P {(strict.precision or 0.0):.1%} "
                  f"R {(strict.recall or 0.0):.1%}  |  presence R "
                  f"{(centre.recall or 0.0):.1%}  ({result.truth_count()} boxes)")
    print(f"report: {report_path}")
    if json_path is not None:
        print(f"json:   {json_path}")
    return 0


def _eval_path(value: str | None) -> Path | None:
    """An eval output path, resolved against the repo root when relative."""
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def cmd_eval(args) -> int:
    """Measure candidate vision models on this box's own policies and footage.

    A development command that ships with the product on purpose: the question
    "should this site's appliance run a different model?" is asked *at* a site,
    against that site's triage verdicts, and answering it from a laptop with
    somebody else's benchmark is how the Phase 2 verifier got shipped disabled
    on the strength of five hand-run prompts.
    """
    from argus.eval.harness import build_labels, default_json_path, run_eval, write_json
    from argus.eval.report import render

    config, database = _load(args)

    if args.build_labels:
        build_labels(config, database)
        if not args.run:
            return 0

    slices = tuple(args.slice) if args.slice else None
    crop = None if args.crop_banner is None else bool(args.crop_banner)

    # Two runs on the same day resolve to the same default results path, so the
    # obvious invocation — `--baseline evals/results-<today>.json` with no
    # `--json` — would overwrite the file it is comparing against, and the
    # damage is only visible the next time somebody wants that baseline.
    baseline_path = _eval_path(args.baseline)
    json_path = _eval_path(args.json) or default_json_path(config)
    if baseline_path is not None and baseline_path.resolve() == json_path.resolve():
        print(f"eval: --baseline and the results path are the same file "
              f"({json_path}). Pass --json to write this run somewhere else.",
              file=sys.stderr)
        return 2

    run = run_eval(
        config, database,
        cloud=args.cloud,
        local=not args.no_local,
        local_model=args.local_model,
        cloud_models=[m.strip() for m in args.models.split(",") if m.strip()] if args.models else None,
        slices=slices,
        crop_banner=crop,
        limit=args.limit,
        concurrency=args.concurrency,
        baseline=baseline_path,
    )

    # `--report docs/VLM_EVAL.md` is the documented invocation, and it must mean
    # the same file whatever directory it is typed in — so a relative path
    # resolves against the repo root rather than the shell's cwd.
    write_json(run, json_path)

    report_path = _eval_path(args.report) or config.eval_report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render(run))

    print(f"eval: {len(run.eval_set)} case(s), {len(run.eval_set.scored)} labelled, "
          f"{len(run.reports)} model(s) in {run.duration_sec:.1f}s")
    print(f"report: {report_path}")
    print(f"json:   {json_path}")
    # A run that measured nothing is not a successful run — it is a
    # misconfiguration, and a zero exit code would hide it from a script.
    return 0 if run.reports else 1


def _read_password(prompt: str = "password: ") -> str:
    """Read a password without echoing it, or from stdin when piped.

    The pipe path exists for the installer: an offline provisioning script
    should be able to create the first admin without a TTY, and the alternative
    — a ``--password`` flag — writes the plant's administrator password into
    shell history and into the process table where every other user on the box
    can read it with ``ps``.
    """
    import getpass

    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        if not line:
            raise ValueError("no password on stdin")
        return line.rstrip("\n")
    first = getpass.getpass(prompt)
    second = getpass.getpass("confirm: ")
    if first != second:
        raise ValueError("passwords do not match")
    return first


def cmd_user(args) -> int:
    """Local accounts — the bootstrap, and the recovery path.

    This exists on the box for two reasons the API cannot cover. A freshly
    imaged appliance has no accounts and therefore no way to authenticate the
    request that would create one; and an appliance whose last administrator
    locked themselves out has to be recoverable by somebody with physical
    access, which is the only credential left at that point.
    """
    from argus import accounts
    from argus.auth import ROLE_DESCRIPTIONS, ROLES

    config, database = _load(args)

    if args.action == "list":
        rows = database.list_users()
        if not rows:
            print(accounts.bootstrap_hint(database))
            return 1
        print(f"{'USERNAME':<20} {'ROLE':<9} {'STATE':<9} {'LAST LOGIN':<26} NAME")
        for row in rows:
            state = "disabled" if row["disabled"] else "active"
            print(f"{row['username']:<20} {row['role']:<9} {state:<9} "
                  f"{row['last_login_at'] or '(never)':<26} {row['display_name']}")
        print()
        for role in ROLES:
            print(f"  {role:<9} {ROLE_DESCRIPTIONS[role]}")
        return 0

    try:
        if args.action == "add":
            password = _read_password()
            user = accounts.create_user(
                database, args.username, password,
                role=args.role, display_name=args.name,
            )
            print(f"created {user['username']} ({user['role']}) — id {user['id']}")
            return 0

        if args.action == "passwd":
            user = accounts.user_by_name(database, args.username)
            password = _read_password(f"new password for {user['username']}: ")
            accounts.set_password(database, int(user["id"]), password)
            print(f"password changed for {user['username']}; every session for it was ended")
            return 0

        if args.action in ("disable", "enable"):
            user = accounts.user_by_name(database, args.username)
            updated = accounts.set_disabled(database, int(user["id"]),
                                            args.action == "disable")
            state = "disabled" if updated["disabled"] else "enabled"
            print(f"{updated['username']} is now {state}")
            return 0

        if args.action == "role":
            user = accounts.user_by_name(database, args.username)
            updated = accounts.set_role(database, int(user["id"]), args.role)
            print(f"{updated['username']} is now {updated['role']}; its sessions were ended")
            return 0
    except accounts.AccountError as exc:
        print(f"user: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:  # password entry, not account rules
        print(f"user: {exc}", file=sys.stderr)
        return 1

    return 2  # unreachable: argparse constrains `action`


def cmd_status(args) -> int:
    from argus.ingest.gateway import Gateway

    config, database = _load(args)
    counts = database.counts()
    print(f"config:   {config.source_path or '(defaults)'}")
    print(f"data dir: {config.data_dir}")
    print(f"database: {config.db_path} (schema v{database.schema_version()})")
    print("counts:   " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    gateway = Gateway(config, database.list_cameras(enabled_only=True))
    print(f"gateway:  {gateway.version() or 'NOT INSTALLED (run scripts/fetch_gateway.sh)'}")
    model = config.detection.model
    print(f"detector: rf-detr-{model.variant} {model.precision}"
          f"{'-traced' if model.trace else '-eager'} on {model.device}, "
          f"threshold={model.threshold}, batch<={model.max_batch}")
    # Which weights, said out loud on the one command whose job is "what is this
    # box configured to do". `status` must not load a model, so the sidecar is
    # read and the refusal reported rather than raised — a checkpoint the engine
    # would reject is exactly what somebody runs `argus status` to find out.
    checkpoint = config.detector_checkpoint_path
    if checkpoint is not None:
        from argus.vision.checkpoint import CheckpointError, load_provenance

        try:
            print(f"weights:  {load_provenance(checkpoint).summary()}")
        except CheckpointError as exc:
            print(f"weights:  UNUSABLE — {exc}", file=sys.stderr)
    else:
        print("weights:  stock COCO (detection.model.checkpoint is unset)")

    cameras = database.list_cameras()
    if cameras:
        from argus.redact import redact_source

        print("\ncameras:")
        for camera in cameras:
            flag = "on " if camera["enabled"] else "off"
            # Masked: this output gets pasted into tickets and chat.
            print(f"  [{flag}] {camera['id']:<28} {camera['status']:<10} {camera['kind']:<5} "
                  f"{redact_source(camera['source'])}")
    policies = database.list_policies()
    if policies:
        print("\npolicies:")
        for policy in policies:
            flag = "on " if policy["enabled"] else "off"
            print(f"  [{flag}] {policy['id']:<28} {policy['kind']:<8} "
                  f"{policy['lifecycle']:<10} stage={policy['stage']}")
    return 0


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="argus",
        description="Argus — fully-local industrial safety video analytics.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"argus {__version__}")
    parser.add_argument(
        "-c", "--config", type=Path, default=None,
        help="path to argus.yaml (default: $ARGUS_CONFIG or config/argus.yaml)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed_parser = subparsers.add_parser("seed", help="import config/seed.yaml into the database")
    seed_parser.add_argument("-f", "--file", type=Path, default=None, help="seed file to import")
    seed_parser.set_defaults(func=cmd_seed)

    run_parser = subparsers.add_parser("run", help="run the engine (gateway + decoders)")
    run_parser.add_argument("--duration", type=float, default=None,
                            help="stop after N seconds (default: run until interrupted)")
    run_parser.add_argument("--report-every", type=float, default=10.0,
                            help="seconds between decoder stat reports (default: 10)")
    run_parser.add_argument("--no-gateway", action="store_true",
                            help="do not start go2rtc; decode straight from camera sources")
    run_parser.add_argument("--direct", action="store_true",
                            help="start the gateway but decode from the original sources")
    run_parser.add_argument("--no-record", action="store_true",
                            help="skip the ring buffer; events get snapshots but no clips")
    run_parser.set_defaults(func=cmd_run)

    api_parser = subparsers.add_parser("api", help="run the JSON HTTP API")
    api_parser.add_argument("--host", default=None, help="bind address (default: config api.host)")
    api_parser.add_argument("--port", type=int, default=None, help="port (default: config api.port)")
    api_parser.set_defaults(func=cmd_api)

    retention_parser = subparsers.add_parser(
        "retention", help="run one housekeeping sweep now (`argus run` does this on a timer)"
    )
    retention_parser.add_argument("-n", "--dry-run", action="store_true",
                                  help="report disk usage and the configured windows; delete nothing")
    retention_parser.set_defaults(func=cmd_retention)

    vlm_parser = subparsers.add_parser(
        "vlm", help="manage the local VLM inference server (start | stop | health)"
    )
    vlm_parser.add_argument("action", choices=("start", "stop", "health"), nargs="?",
                            default="health", help="default: health")
    vlm_parser.add_argument("--no-wait", action="store_true",
                            help="start: return as soon as the process is launched, "
                                 "without waiting for the model to load")
    vlm_parser.set_defaults(func=cmd_vlm)

    processing_parser = subparsers.add_parser(
        "processing",
        help="the appliance pause switch (status | pause | resume); `gate` is the "
             "systemd ExecCondition",
    )
    processing_parser.add_argument(
        "action", choices=("status", "pause", "resume", "gate"), nargs="?",
        default="status",
        help="status: show the flag and each service; pause/resume: flip it and "
             "start/stop the engine+VLM; gate: exit 0 if enabled, 1 if paused "
             "(used by the units, not by hand)",
    )
    processing_parser.set_defaults(func=cmd_processing)

    index_parser = subparsers.add_parser(
        "index", help="manage the semantic search index (backfill | status)"
    )
    index_parser.add_argument("action", choices=("backfill", "status"), nargs="?",
                              default="status", help="default: status")
    index_parser.add_argument("--reset", action="store_true",
                              help="backfill: drop the existing index first (required after "
                                   "changing index.model — vectors from two models are not "
                                   "comparable)")
    index_parser.add_argument("--limit", type=int, default=None,
                              help="backfill: stop after N events, newest first")
    index_parser.set_defaults(func=cmd_index)

    eval_parser = subparsers.add_parser(
        "eval", help="measure candidate VLMs on this appliance's policies and footage"
    )
    eval_parser.add_argument("--report", default=None,
                             help=f"markdown report path (default: {EvalConfig().report_path})")
    eval_parser.add_argument("--json", default=None,
                             help="per-case results path (default: evals/results-<date>.json)")
    eval_parser.add_argument(
        "--cloud", action="store_true",
        help="also evaluate the OpenRouter candidates. Requires OPENROUTER_API_KEY; "
             "frames are sent to third-party APIs. Development only — no runtime "
             "path in this product can do this.",
    )
    eval_parser.add_argument("--no-local", action="store_true",
                             help="skip the local vLLM service")
    eval_parser.add_argument("--local-model", default=None,
                             help="override the model id sent to the local endpoint")
    eval_parser.add_argument("--models", default=None,
                             help="comma-separated cloud candidates (default: eval.cloud_models)")
    eval_parser.add_argument("--slice", action="append", choices=SLICES, default=None,
                             help="restrict to one slice; repeatable")
    eval_parser.add_argument("--limit", type=int, default=None,
                             help="stop after N cases, labelled ones first")
    eval_parser.add_argument("--concurrency", type=int, default=None,
                             help="in-flight evaluations per model (default: 1 — latency "
                                  "measured under load is not latency)")
    eval_parser.add_argument("--baseline", default=None,
                             help="a previous results JSON whose answers join this report, "
                                  "re-scored against the current labels. vLLM serves one "
                                  "model at a time, so comparing two local candidates is "
                                  "two runs; this is how they land in one table")
    banner = eval_parser.add_mutually_exclusive_group()
    banner.add_argument("--crop-banner", dest="crop_banner", action="store_true", default=None,
                        help="crop Argus's burned-in headline off every frame before "
                             "sending it, including the triage slice")
    banner.add_argument("--keep-banner", dest="crop_banner", action="store_false",
                        help="keep the headline on every frame, including generated ones")
    eval_parser.add_argument("--build-labels", action="store_true",
                             help="sample frames from the test footage and merge them into "
                                  "evals/labels.yaml, then exit unless --run is given")
    eval_parser.add_argument("--run", action="store_true",
                             help="with --build-labels: evaluate after rebuilding the labels")
    eval_parser.set_defaults(func=cmd_eval)

    detector_eval_parser = subparsers.add_parser(
        "detector-eval",
        help="score the shipped person detector against a labelled corpus",
    )
    detector_eval_parser.add_argument(
        "--dataset", type=Path, default=None,
        help="YOLO-format corpus root, holding images/<split> and labels/<split> "
             "(default: footage/anmol-samples/cctv_person_tracking_dataset)",
    )
    detector_eval_parser.add_argument(
        "--split", action="append", default=None,
        help="restrict to one split; repeatable (default: val then train)",
    )
    detector_eval_parser.add_argument(
        "--condition", action="append", default=None, choices=list(DETECTOR_CONDITIONS),
        help="restrict to one input condition; repeatable. 'engine' is the shipped "
             "path; the others isolate what the decode downscale and the aspect "
             "squash cost (default: all three)",
    )
    detector_eval_parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="also score fine-tuned weights (scripts/train_detector.py) over the same "
             "frames and add the before/after to the report. One command, because the "
             "answer wanted is a difference",
    )
    detector_eval_parser.add_argument(
        "--report", default=None, help="markdown report path (default: docs/DETECTOR_EVAL.md)")
    detector_eval_parser.add_argument(
        "--json", default=None, help="per-frame results dump (default: not written)")
    detector_eval_parser.add_argument("-q", "--quiet", action="store_true",
                                      help="no per-frame progress")
    detector_eval_parser.set_defaults(func=cmd_detector_eval)

    user_parser = subparsers.add_parser(
        "user", help="local accounts (add | passwd | list | disable | enable | role)"
    )
    user_parser.add_argument(
        "action", choices=("add", "passwd", "list", "disable", "enable", "role"),
        help="add: create an account (prompts for the password, or reads it from stdin)",
    )
    user_parser.add_argument("username", nargs="?", default=None,
                             help="the account to act on (not needed for `list`)")
    user_parser.add_argument("--role", default="REVIEW", choices=("ADMIN", "RESPOND",
                                                                  "REVIEW", "SYSTEM"),
                             help="add/role: the permission set (default: REVIEW)")
    user_parser.add_argument("--name", default=None,
                             help="add: display name, e.g. 'M. Alvarez' (default: the username)")
    user_parser.set_defaults(func=cmd_user)

    backup_parser = subparsers.add_parser(
        "backup", help="database snapshots (now | list | verify | restore)"
    )
    backup_parser.add_argument("action", choices=("now", "list", "verify", "restore"),
                               nargs="?", default="list", help="default: list")
    backup_parser.add_argument("stamp", nargs="?", default=None,
                               help="restore/verify: the stamp from `argus backup list`, "
                                    "an unambiguous prefix of one, or `latest`")
    backup_parser.add_argument("--force", action="store_true",
                               help="restore: proceed even though the engine looks alive "
                                    "(it holds the database open — be sure it is gone)")
    backup_parser.add_argument("--reason", default="manual",
                               help="now: recorded in the manifest (e.g. 'pre-upgrade')")
    backup_parser.set_defaults(func=cmd_backup)

    doctor_parser = subparsers.add_parser(
        "doctor", help="check everything an appliance needs; non-zero exit means broken"
    )
    doctor_parser.add_argument("--json", action="store_true",
                               help="machine-readable output for a monitoring probe")
    doctor_parser.add_argument("--strict", action="store_true",
                               help="exit non-zero on warnings too")
    doctor_parser.set_defaults(func=cmd_doctor)

    services_parser = subparsers.add_parser(
        "install-services",
        help="write and enable systemd --user units for this checkout (no root)",
    )
    services_parser.add_argument("--vlm", action="store_true",
                                 help="also enable argus-vlm.service (optional equipment; "
                                      "it claims GPU memory for the model)")
    services_parser.add_argument("--start", action="store_true",
                                 help="start argus.target once the units are enabled")
    services_parser.add_argument("--no-enable", action="store_true",
                                 help="write the units but do not enable them")
    services_parser.add_argument("-n", "--dry-run", action="store_true",
                                 help="print what would be written and run; change nothing")
    services_parser.add_argument("--uninstall", action="store_true",
                                 help="stop, disable and remove the generated units")
    services_parser.set_defaults(func=cmd_install_services)

    status_parser = subparsers.add_parser("status", help="show configuration and database state")
    status_parser.set_defaults(func=cmd_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interactive path
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
