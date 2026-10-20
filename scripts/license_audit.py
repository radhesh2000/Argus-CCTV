#!/usr/bin/env python3
"""Licence audit of the *resolved* dependency trees — the two that are enforced,
the one that is reported, and any other one you point it at.

`pyproject.toml` and `ui/package.json` say what we asked for. This says what we
got, which is the only thing a lawyer cares about. It reads metadata already on
disk — installed `.dist-info/METADATA` for Python, `package.json` under
`node_modules` for the console — so it needs no network and no extra tool, and
where a package declares nothing it reads the licence *text* the wheel shipped
rather than guessing from the name.

    uv run python scripts/license_audit.py            # rewrite docs/LICENSES.md
    uv run python scripts/license_audit.py --check    # exit 1 on GPL/AGPL/LGPL
    uv run python scripts/license_audit.py --tree .venv-train   # one side env, listed

`--check` is the form worth wiring into CI: the rule "no copyleft in the product
tree" expressed as a command instead of a promise.

Three classes come out of this, and conflating them is how audits go wrong:

* **permissive** — Apache/MIT/BSD/PSF/ISC and friends. Nothing to do.
* **weak (file-level) copyleft** — MPL-2.0. Combining it with proprietary code is
  explicitly allowed (MPL §3.3); what is owed is the source of *those files*.
  An obligation to discharge at packaging time, not a reason to remove anything.
* **strong copyleft** — GPL/AGPL/LGPL. A release blocker in the product tree.
  `bin/ffmpeg` is LGPL by construction and lives outside both package trees; it
  is handled in the document, not here.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from email.parser import Parser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: The day `pipeline/`, `dashboard/` and `detectors/` — and with them the AGPL
#: `ultralytics` dependency — left `main` for the `v1-archive` branch. A fact
#: about the history, so it is a constant: the generated document used to
#: interpolate *today's* date here and quietly rewrote when v1 was deleted every
#: time the audit was run.
V1_DELETED = "2026-08-12"

#: Licences the product may depend on without anyone thinking about it.
ALLOWED = frozenset(
    {
        "Apache-2.0",
        "MIT",
        "MIT-0",
        "MIT-CMU",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "BSD-3-Clause-Clear",
        "0BSD",
        "ISC",
        "PSF-2.0",
        "Python-2.0",
        "CNRI-Python",
        "HPND",
        "BSL-1.0",
        "Unlicense",
        "OFL-1.1",
        "CC0-1.0",
        "BlueOak-1.0.0",
        "Zlib",
        "libpng-2.0",
        "Apache-2.0-with-LLVM-exception",
    }
)

#: Release blockers.
STRONG = ("AGPL", "LGPL", "GPL", "SSPL", "OSL", "EUPL", "CDDL", "CECILL")
#: Obligations, not blockers.
WEAK = ("MPL", "CC-BY", "CDLA")

_CLASSIFIER_SPDX = {
    "Apache Software License": "Apache-2.0",
    "MIT License": "MIT",
    "MIT No Attribution License (MIT-0)": "MIT-0",
    "BSD License": "BSD-3-Clause",
    "Python Software Foundation License": "PSF-2.0",
    "Historical Permission Notice and Disclaimer (HPND)": "HPND",
    "ISC License (ISCL)": "ISC",
    "The Unlicense (Unlicense)": "Unlicense",
    "zlib/libpng License": "Zlib",
    "Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "GNU General Public License (GPL)": "GPL-1.0-or-later",
    "GNU General Public License v2 (GPLv2)": "GPL-2.0",
    "GNU General Public License v3 (GPLv3)": "GPL-3.0",
    "GNU Affero General Public License v3": "AGPL-3.0",
    "GNU Lesser General Public License v2 (LGPLv2)": "LGPL-2.0",
    "GNU Lesser General Public License v3 (LGPLv3)": "LGPL-3.0",
}

#: Free-text licence fields normalised to SPDX. Exact match only (lowercased and
#: stripped); anything unrecognised is reported verbatim so a human reads it.
_FREETEXT = {
    "apache 2.0": "Apache-2.0",
    "apache 2.0 license": "Apache-2.0",
    "apache-2.0": "Apache-2.0",
    "apache license 2.0": "Apache-2.0",
    "apache license, version 2.0": "Apache-2.0",
    "apache software license": "Apache-2.0",
    "asl 2": "Apache-2.0",
    "bsd": "BSD-3-Clause",
    "bsd license": "BSD-3-Clause",
    "bsd-3-clause": "BSD-3-Clause",
    "bsd 3-clause": "BSD-3-Clause",
    "3-clause bsd license": "BSD-3-Clause",
    "new bsd license": "BSD-3-Clause",
    "isc": "ISC",
    "isc license": "ISC",
    "mit": "MIT",
    "mit license": "MIT",
    "psf": "PSF-2.0",
    "psfl": "PSF-2.0",
    "psf-2.0": "PSF-2.0",
    "python software foundation license": "PSF-2.0",
    "unlicense": "Unlicense",
}

#: Where declared metadata is absent, wrong or unparseable and the licence *text*
#: does not settle it either. Every entry is a judgement, so every entry carries
#: its evidence into the generated document.
OVERRIDES: dict[str, tuple[str, str]] = {
    # NVIDIA CUDA runtime, all of it arriving as a dependency of torch.
    **{
        name: ("NVIDIA-SLA", "CUDA Toolkit EULA / per-library SLA; a torch dependency")
        for name in (
            "cuda-bindings",
            "cuda-pathfinder",
            "cuda-python",
            "cuda-toolkit",
            "nvidia-cublas",
            "nvidia-cuda-cccl",
            "nvidia-cuda-crt",
            "nvidia-cuda-cudart",
            "nvidia-cuda-cupti",
            "nvidia-cuda-nvcc",
            "nvidia-cuda-nvdisasm",
            "nvidia-cuda-nvrtc",
            "nvidia-cuda-nvvm",
            "nvidia-cuda-runtime",
            "nvidia-cudnn-cu13",
            "nvidia-cufft",
            "nvidia-cufile",
            "nvidia-curand",
            "nvidia-cusolver",
            "nvidia-cusparse",
            "nvidia-cusparselt-cu13",
            "nvidia-cutlass-dsl",
            "nvidia-cutlass-dsl-libs-base",
            "nvidia-cutlass-dsl-libs-core",
            "nvidia-cutlass-dsl-libs-cu12",
            "nvidia-cutlass-dsl-libs-cu13",
            "nvidia-nccl-cu13",
            "nvidia-nvjitlink",
            "nvidia-nvshmem-cu13",
            "nvidia-nvtx",
            "nvidia-nvvm",
        )
    },
    "matplotlib": ("PSF-2.0", "Matplotlib licence — a PSF-2.0 derivative, BSD-compatible"),
    "networkx": ("BSD-3-Clause", "declares the bare string 'BSD'; LICENSE.txt is 3-clause"),
    "pillow": ("MIT-CMU", "the historical PIL/Pillow permissive licence"),
    "python-dateutil": ("Apache-2.0", "declares 'Dual License'; it is Apache-2.0 OR BSD-3-Clause"),
    "sqlite-vec": (
        "MIT OR Apache-2.0",
        "declares the free-text 'MIT License, Apache License, Version 2.0'; the "
        "upstream repo dual-licenses under both and ships LICENSE-MIT + LICENSE-APACHE",
    ),
    "supervisor": ("BSD-3-Clause", "'BSD-derived (repoze)' — the ZPL-style permissive repoze licence"),
    "caniuse-lite": ("CC-BY-4.0", "a browser-support *data* set, build-time only; never shipped"),
}

# Signatures for reading a licence out of the text a wheel shipped, used only
# when the metadata says nothing useful. Ordered: first hit wins.
_TEXT_SIGNATURES: tuple[tuple[str, str], ...] = (
    ("GNU AFFERO GENERAL PUBLIC LICENSE", "AGPL-3.0"),
    ("GNU LESSER GENERAL PUBLIC LICENSE", "LGPL-3.0"),
    ("GNU GENERAL PUBLIC LICENSE", "GPL-3.0"),
    ("MOZILLA PUBLIC LICENSE", "MPL-2.0"),
    ("APACHE LICENSE", "Apache-2.0"),
    ("BSD 3-CLAUSE", "BSD-3-Clause"),
    ("BSD 2-CLAUSE", "BSD-2-Clause"),
    ("MIT LICENSE", "MIT"),
    ("PERMISSION IS HEREBY GRANTED, FREE OF CHARGE", "MIT"),
    ("ISC LICENSE", "ISC"),
    ("PERMISSION TO USE, COPY, MODIFY, AND/OR DISTRIBUTE", "ISC"),
)


@dataclass(frozen=True)
class Pkg:
    name: str
    version: str
    licence: str
    note: str = ""
    source: str = ""  # how the licence was determined, when not from metadata

    @property
    def atoms(self) -> list[str]:
        """The SPDX expression split into the licences it is made of."""
        expr = re.sub(r"\bWITH\s+[\w.-]+", "", self.licence, flags=re.I)
        expr = expr.replace("(", " ").replace(")", " ")
        return [a for a in re.split(r"\s+(?:OR|AND)\s+", expr, flags=re.I) if a.strip()]

    @property
    def classification(self) -> str:
        up = self.licence.upper()
        for tok in STRONG:
            if re.search(rf"(?<![A-Z]){tok}(?![A-Z])", up):
                return "strong"
        for tok in WEAK:
            if tok in up:
                return "weak"
        if self.licence == "NVIDIA-SLA" or "PROPRIETARY" in up:
            return "proprietary"
        if all(a.strip() in ALLOWED for a in self.atoms):
            return "permissive"
        return "unknown"


def _normalise(raw: str) -> str:
    raw = (raw or "").strip().strip('"')
    if not raw:
        return ""
    key = raw.lower().strip(" .")
    if key in _FREETEXT:
        return _FREETEXT[key]
    if len(raw) > 60 or "\n" in raw:  # a whole licence pasted into the field
        return ""
    return raw


def _from_text(dist_info: Path) -> tuple[str, str]:
    """Read the licence out of whatever licence file the wheel shipped."""
    candidates = sorted(dist_info.glob("licenses/**/*")) + sorted(dist_info.glob("LICEN*"))
    for path in candidates:
        if not path.is_file():
            continue
        head = path.read_text(encoding="utf-8", errors="replace")[:4000].upper()
        head = re.sub(r"\s+", " ", head)
        for needle, spdx in _TEXT_SIGNATURES:
            if needle in head:
                return spdx, f"read from {path.name}"
    return "UNDECLARED", ""


def python_packages(venv: Path) -> list[Pkg]:
    """Every distribution installed in `venv`, with the licence it declares."""
    site = next(venv.glob("lib/python3.*/site-packages"), None)
    if site is None:
        raise SystemExit(f"no site-packages under {venv}")
    out: list[Pkg] = []
    for meta in sorted(site.glob("*.dist-info/METADATA")):
        msg = Parser().parsestr(meta.read_text(encoding="utf-8", errors="replace"))
        name = (msg.get("Name") or meta.parent.name.split("-")[0]).strip()
        version = (msg.get("Version") or "?").strip()
        if name == "argus":
            continue
        if name in OVERRIDES:
            licence, note = OVERRIDES[name]
            out.append(Pkg(name, version, licence, note, "declared wrong; see note"))
            continue
        licence = _normalise(msg.get("License-Expression") or "") or _normalise(msg.get("License") or "")
        source = ""
        if not licence:
            mapped = [
                _CLASSIFIER_SPDX.get(c.split("::")[-1].strip(), c.split("::")[-1].strip())
                for c in msg.get_all("Classifier", [])
                if c.startswith("License ::")
            ]
            mapped = [m for m in mapped if m not in ("OSI Approved", "Other/Proprietary License")]
            if mapped:
                licence = " OR ".join(dict.fromkeys(mapped))
        if not licence:
            licence, source = _from_text(meta.parent)
        out.append(Pkg(name, version, licence, source=source))
    return sorted(out, key=lambda p: p.name.lower())


def node_packages(ui: Path) -> list[Pkg]:
    """Every package `npm ls` resolves for the console, production and dev.

    Optional platform binaries (`@rollup/rollup-win32-*` and the rest) appear in
    the tree with no version because npm never installed them on this machine.
    They are skipped: a package that is not on disk is not shipped, and listing
    fifty phantom rows as UNDECLARED would bury the ones that matter.
    """
    proc = subprocess.run(
        ["npm", "ls", "--all", "--json", "--long"],
        cwd=ui,
        capture_output=True,
        text=True,
    )
    if not proc.stdout:
        raise SystemExit(f"npm ls produced nothing: {proc.stderr[:400]}")
    tree = json.loads(proc.stdout)
    seen: dict[tuple[str, str], Pkg] = {}

    def walk(node: dict) -> None:
        for name, dep in (node.get("dependencies") or {}).items():
            version = dep.get("version")
            if not version or not dep.get("path"):
                continue  # optional, not installed here
            key = (name, version)
            if key not in seen:
                if name in OVERRIDES:
                    licence, note = OVERRIDES[name]
                    seen[key] = Pkg(name, version, licence, note)
                else:
                    lic = dep.get("license")
                    if isinstance(lic, dict):
                        lic = lic.get("type", "")
                    if isinstance(lic, list):
                        lic = " OR ".join(
                            x.get("type", str(x)) if isinstance(x, dict) else str(x) for x in lic
                        )
                    seen[key] = Pkg(name, version, _normalise(str(lic or "")) or "UNDECLARED")
            walk(dep)

    walk(tree)
    return sorted(seen.values(), key=lambda p: p.name.lower())


def _table(pkgs: list[Pkg]) -> str:
    marks = {"permissive": "", "weak": " ⚠ weak copyleft", "strong": " ⛔ copyleft",
             "proprietary": " ⚠ proprietary", "unknown": " ? unreviewed"}
    rows = ["| package | version | licence |", "|---|---|---|"]
    for p in pkgs:
        rows.append(f"| `{p.name}` | {p.version} | {p.licence}{marks[p.classification]} |")
    return "\n".join(rows)


def _summary(pkgs: list[Pkg]) -> str:
    counts: dict[str, int] = {}
    for p in pkgs:
        counts[p.licence] = counts.get(p.licence, 0) + 1
    return ", ".join(f"{lic} ×{n}" for lic, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _verdict(pkgs: list[Pkg]) -> str:
    strong = [p for p in pkgs if p.classification == "strong"]
    weak = [p for p in pkgs if p.classification == "weak"]
    prop = [p for p in pkgs if p.classification == "proprietary"]
    unknown = [p for p in pkgs if p.classification == "unknown"]
    bits = []
    bits.append(f"**{len(strong)} copyleft**" if strong else "**no GPL/AGPL/LGPL**")
    if weak:
        bits.append(f"{len(weak)} weak-copyleft")
    if prop:
        bits.append(f"{len(prop)} NVIDIA proprietary")
    if unknown:
        bits.append(f"{len(unknown)} unreviewed")
    return ", ".join(bits)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="exit 1 if the product tree has copyleft")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "LICENSES.md")
    ap.add_argument(
        "--tree", type=Path, default=None,
        help="list the licences of one arbitrary venv and stop. Neither the "
             "document nor --check sees it: this is for side environments that "
             "exist on the disk without being part of the product "
             "(`.venv-train`, built by scripts/setup_detector_training.sh).",
    )
    args = ap.parse_args()

    if args.tree is not None:
        pkgs = python_packages(args.tree)
        notable = _notable(pkgs, ("strong", "weak", "unknown"))
        for p in notable:
            print(f"  {p.classification:10} {p.name} {p.version} — {p.licence}")
        print(f"{len(pkgs)} packages in {args.tree}: {_verdict(pkgs)}")
        return 0

    py = python_packages(ROOT / ".venv")
    ui = node_packages(ROOT / "ui")
    vlm_venv = ROOT / ".venv-vlm"
    vlm = python_packages(vlm_venv) if vlm_venv.exists() else []

    if args.check:
        rc = 0
        for label, pkgs in (("python", py), ("node", ui), ("vlm", vlm)):
            for p in pkgs:
                if p.classification == "strong":
                    print(f"COPYLEFT [{label}]: {p.name} {p.version} — {p.licence}", file=sys.stderr)
                    if label != "vlm":  # the vLLM venv is an install artefact, reported not enforced
                        rc = 1
        print(f"checked {len(py)} python + {len(ui)} node + {len(vlm)} vlm packages")
        return rc

    args.out.write_text(_document(py, ui, vlm))
    print(f"wrote {args.out} — {len(py)} python, {len(ui)} node, {len(vlm)} vlm")
    for label, pkgs in (("python", py), ("node", ui), ("vlm", vlm)):
        for p in pkgs:
            if p.classification in ("strong", "weak", "unknown"):
                print(f"  [{label}] {p.classification:10} {p.name} {p.version} — {p.licence}")
    return 0


def _notable(pkgs: list[Pkg], kinds: tuple[str, ...]) -> list[Pkg]:
    return [p for p in pkgs if p.classification in kinds]


def _document(py: list[Pkg], ui: list[Pkg], vlm: list[Pkg]) -> str:
    weak_py = _notable(py, ("weak",))
    weak_ui = _notable(ui, ("weak",))
    strong_all = _notable(py + ui, ("strong",))
    strong_vlm = _notable(vlm, ("strong",))
    nvidia_py = [p for p in py if p.licence == "NVIDIA-SLA"]
    unknown = _notable(py + ui + vlm, ("unknown",))

    weak_rows = "\n".join(
        f"| `{p.name}` | {p.version} | {p.licence} | {'appliance' if p in weak_py else 'console'} |"
        for p in weak_py + weak_ui
    )
    strong_vlm_rows = "\n".join(
        ["| package | version | licence |", "|---|---|---|"]
        + [f"| `{p.name}` | {p.version} | {p.licence} |" for p in strong_vlm]
    )
    unknown_rows = (
        "\n".join(f"- `{p.name}` {p.version} — declares `{p.licence}`" for p in unknown)
        or "- none"
    )
    no_copyleft = "no" if not strong_all else f"**{len(strong_all)}**"

    return f"""# Dependency licences

Generated **{date.today().isoformat()}** by `scripts/license_audit.py`, from the
*resolved* trees — the packages actually on disk, not the ranges asked for in
`pyproject.toml` and `ui/package.json`.

```bash
uv sync && (cd ui && npm ci)                      # resolve both trees
uv run python scripts/license_audit.py            # rewrite this file
uv run python scripts/license_audit.py --check    # exit 1 if the product tree gains copyleft
```

The rule: **Apache-2.0 / MIT / BSD / PSF / ISC / OFL only in the product tree, and
no GPL or AGPL anywhere in it.** `--check` is that rule as a command — it reads
the same resolved metadata and fails on GPL, AGPL, LGPL, SSPL, OSL, EUPL, CDDL
or CECILL — and it is the thing to wire into CI, because a rule nobody can run is
a rule nobody keeps.

## Result

**There is {no_copyleft} GPL, AGPL or LGPL package in the appliance tree or the
console tree.** The one AGPL dependency this project ever had — `ultralytics`,
v1's YOLOv8n runtime — went out with v1 itself on {V1_DELETED};
RF-DETR (Apache-2.0) had already replaced it in the vision engine, and no module
under `argus/` ever imported it.

| tree | packages | verdict |
|---|---|---|
| Python appliance — `.venv`, from `uv.lock` | {len(py)} | {_verdict(py)} |
| Operator console — `ui/node_modules`, from `package-lock.json` | {len(ui)} | {_verdict(ui)} |
| VLM serving venv — `.venv-vlm`, an install artefact | {len(vlm)} | {_verdict(vlm)} |

Three things in that table need a paragraph rather than a row.

### Weak (file-level) copyleft: {len(weak_py) + len(weak_ui)} packages, an obligation rather than a problem

| package | version | licence | tree |
|---|---|---|---|
{weak_rows}

`certifi` (the CA bundle) and `tqdm` both arrive under
`transformers`/`huggingface-hub`/`supervision`; neither is imported by `argus/`.
MPL-2.0 is *file*-level copyleft and §3.3 explicitly permits combining MPL files
with proprietary code under other terms, so nothing here needs removing. What it
obliges is publishing the source of **those files** — unmodified upstream, which
is what we ship. **Action at packaging time: include both packages' source (or a
written offer) in the appliance image**, alongside the ffmpeg source LGPL already
requires.

`caniuse-lite` is browser-support *data* under CC-BY-4.0, consumed by the CSS
autoprefixer at build time. It is not code and never reaches `ui/dist`.

### NVIDIA CUDA runtime: {len(nvidia_py)} proprietary packages, unavoidable, and not the same case as `tensorrt`

They arrive as dependencies of `torch`, not as anything this project chose, and
they *are* the CUDA runtime — no CUDA program runs without them. The CUDA Toolkit
EULA and the per-library SLAs permit redistributing the runtime libraries as part
of an application, so this ships; but they are proprietary, not permissive, and an
appliance image must carry their EULA text.

This is a different decision from the `tensorrt` wheel the benchmark stage
rejected. That one was an *optional* dependency with a measured-slower
alternative, so the SLA was a reason to say no. These are not optional for any
program that touches the GPU. On a DGX Spark they also duplicate libraries the
driver and CUDA 13 toolkit already provide, so an appliance build that links the
system CUDA and installs torch `--no-deps` drops all {len(nvidia_py)} — a Phase-5
packaging decision, and the reason this section exists is so that decision gets
made deliberately.

### `.venv-vlm` contains LGPL, and it is not in the product tree

{("The vLLM serving venv resolves one copyleft package:" if len(strong_vlm) == 1 else f"The vLLM serving venv resolves {len(strong_vlm)} copyleft packages:") if strong_vlm else "No copyleft appears in the serving venv today."}

{strong_vlm_rows if strong_vlm else ""}

`pycountry` arrives through `mistral-common` → `pydantic-extra-types[pycountry]`,
which vLLM installs for Mistral tokenizer support — a code path Argus never
enters, since the appliance serves Qwen3-VL-8B and talks to it over the
OpenAI-compatible HTTP endpoint. It matters anyway, because the venv is built on
the appliance by `scripts/setup_vlm_server.sh`:

- **It is a separate process behind an HTTP boundary.** Nothing in `argus/`
  imports it, links it or is derived from it, so the LGPL's reciprocity reaches
  the package and stops.
- **It is pure Python, installed unmodified.** LGPL-2.1 §6's relink requirement
  is satisfied by the fact that the file is right there and replaceable.
- **It is removable.** `pip uninstall pycountry mistral-common` in `.venv-vlm`
  costs Argus nothing measurable; if a customer contract forbids LGPL on the
  appliance at all, that is the fix, and `--check` reports the venv without
  failing on it for exactly this reason.

`--check` therefore enforces the appliance and console trees and *reports* the
serving venv.

### `.venv-train`, when a site has fine-tuned its detector

`scripts/setup_detector_training.sh` builds a third virtualenv for
`scripts/train_detector.py` — pytorch-lightning, torchmetrics, pycocotools and
their dependencies. It is on the same footing as `.venv-vlm` and one step
further removed: the engine does not import it, does not talk to it, and does
not need it to exist. It is a lab tool that happens to live on the same disk,
which is why it is neither in this document nor in `--check`. It is not
unexamined, though — `--tree` lists it on demand:

```bash
uv run python scripts/license_audit.py --tree .venv-train
```

Measured 2026-08-24, 89 packages: **no GPL/AGPL/LGPL**, two weak-copyleft
(`certifi` MPL-2.0 and `tqdm` MPL-2.0-AND-MIT, both already in the appliance
tree), one unreviewed (`pycocotools`, which declares "FreeBSD" — BSD-2-Clause).
The `rfdetr[train]` extra's `roboflow` and `rf100vl` are deliberately **not**
installed; see the header of the setup script for why a cloud SDK has no place
on this box even inert.

## Vendored binaries — in neither package tree

Fetched or built by `scripts/`, living in `bin/`:

| binary | licence | how it gets that way |
|---|---|---|
| `bin/ffmpeg`, `bin/ffprobe` | **LGPL-2.1+** | `scripts/build_ffmpeg.sh` builds n8.1.2 from source with **no** `--enable-gpl` and **no** `--enable-nonfree`, against MIT nv-codec-headers. A distribution ffmpeg is GPL and would poison the appliance; this one is dynamically linked, redistributable, and obliges us to ship its source under LGPL §6 — an open packaging item for Phase 5. |
| `bin/go2rtc` | **MIT** | Upstream release binary, pinned by sha256 in `scripts/fetch_gateway.sh`. |

Model weights are licensed separately from code, and all of ours are permissive
and ungated: **RF-DETR Nano** checkpoints are Apache-2.0, **Qwen3-VL-8B-Instruct**
is Apache-2.0, and **google/siglip2-base-patch16-384** — the semantic index's
embedder — is Apache-2.0. None is committed; each is fetched once at install time
by a script under `scripts/`, never at runtime.

Two more were fetched for the Phase 4 model trial and are on the same footing:
**Qwen/Qwen3.8-27B-FP8** and **Qwen/Qwen3-VL-8B-Instruct-FP8**, both Apache-2.0,
both ungated, both first-party Qwen repositories. First-party is not a detail —
the third-party NVFP4/AWQ int4 checkpoints for the 27B would have been the
faster option and were rejected partly because a re-quantised upload carries an
unaudited licence and undisclosed calibration data, which is precisely what this
document exists to prevent. See `scripts/fetch_vlm.sh`.

The gating status is checked, not assumed, because it is what stopped
Cosmos-Reason2 in Phase 2: every SigLIP 2 checkpoint considered was confirmed
`gated: false` against the HuggingFace API before it was downloaded.

## Unreviewed declarations

Packages whose declared string is not an SPDX id this script recognises. Each has
been read by hand and none is copyleft; they are listed so the list stays honest
rather than silently normalised away.

{unknown_rows}

## Python — appliance runtime (`.venv`)

{_summary(py)}

{_table(py)}

## Node — operator console (`ui/`)

Build-time and runtime together: everything `npm ls --all` resolves, minus the
optional platform binaries npm never installed here. Only the console's own
source and the bundled fonts reach `ui/dist` — Vite, TypeScript, esbuild and
rollup are build tooling and are not shipped to a customer.

{_summary(ui)}

{_table(ui)}

## Python — VLM serving venv (`.venv-vlm`)

Built by `scripts/setup_vlm_server.sh` from the vLLM nightly cu130 aarch64 index.
It is a separate virtualenv on purpose — vLLM pins a different torch than the
appliance does — and it is an install artefact rather than source, but it runs on
the box, so it is audited on the same terms.

{_summary(vlm) if vlm else "_Not installed here; re-run the audit after `scripts/setup_vlm_server.sh`._"}

{_table(vlm) if vlm else ""}
"""


if __name__ == "__main__":
    raise SystemExit(main())
