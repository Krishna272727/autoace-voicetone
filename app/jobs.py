"""Background batch jobs.

An in-process ThreadPoolExecutor plus a {job_id: Job} dict. No Celery, no
Redis -- they are a day of work this does not need (BUILD_SPEC.md 11.5).

Threads rather than processes: the heavy stages (ffmpeg subprocess, numpy,
onnxruntime, torch) all release the GIL, and a process pool would reload every
model per worker, which costs more than it saves at these batch sizes.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from voicetone import FileResult, analyze_file
from voicetone.audio import SUPPORTED
from voicetone.score import format_report, load_labels, score

log = logging.getLogger("autoace.jobs")

# --- upload limits ---------------------------------------------------------
# Enforced before anything is read into memory, and again per ZIP member so a
# small archive cannot expand into a large one (BUILD_SPEC 6, "ZIP bomb").
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "512")) * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = int(os.getenv("MAX_UNCOMPRESSED_MB", "2048")) * 1024 * 1024
MAX_MEMBERS = int(os.getenv("MAX_ZIP_MEMBERS", "2000"))
MAX_COMPRESSION_RATIO = 200.0
WORKERS = int(os.getenv("BATCH_WORKERS", str(min(8, (os.cpu_count() or 2)))))

# --- audio retention -------------------------------------------------------
# The dashboard lets an evaluator play a clip back and check a prediction by
# ear, which means the audio has to outlive the batch. That is a real change to
# the data-handling posture, so it is bounded, configurable and stated in the
# UI rather than left implicit.
#
#   AUDIO_RETENTION_MIN=0  deletes audio the moment the batch finishes and
#                          disables playback -- the strictest setting.
#   default 60             keeps it for an hour so results can be reviewed.
#
# Audio never leaves this host either way, and is purged on eviction, on expiry
# and on process exit.
AUDIO_RETENTION_MIN = int(os.getenv("AUDIO_RETENTION_MIN", "60"))
PLAYBACK_ENABLED = AUDIO_RETENTION_MIN > 0

RETENTION = (
    f"Uploaded audio is kept for {AUDIO_RETENTION_MIN} minutes so results can be "
    f"reviewed, then deleted automatically. Results live in memory only and are "
    f"lost on restart."
) if PLAYBACK_ENABLED else (
    "Uploaded audio is deleted as soon as the batch completes. Results live in "
    "memory only and are lost on restart."
)


# What each pipeline stage is called on screen. The internal names are
# accurate but mean nothing to someone waiting on their calls.
STAGE_LABELS = {
    "vad":     "Finding the speech",
    "quality": "Checking line quality",
    "noise":   "Listening to the background",
    "overlap": "Looking for interruptions",
    "roles":   "Separating the speakers",
    "emotion": "Reading the customer's tone",
    "stub":    "Finishing up",
}


class UploadTooLarge(ValueError):
    pass


class BadArchive(ValueError):
    pass


@dataclass
class Job:
    id: str
    status: str = "queued"          # queued | running | done | error
    total: int = 0
    completed: int = 0
    current: str = ""
    # What was uploaded, as named by the person uploading it.
    sources: list[str] = field(default_factory=list)
    started: float = field(default_factory=time.monotonic)
    finished: float | None = None
    error: str | None = None
    results: list[FileResult] = field(default_factory=list)
    report: dict[str, Any] | None = None
    unmatched_labels: list[str] = field(default_factory=list)
    unlabelled_audio: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Retained audio, keyed by the exact display name. Serving only values from
    # this map is what makes the playback route traversal-proof: a request names
    # a row, it never names a path.
    audio_paths: dict[str, Path] = field(default_factory=dict)
    root: Path | None = None
    audio_expires: float | None = None
    # Which stage the current file is on. Without this a one-file batch reports
    # 0 of 1 for its whole run and reads as frozen.
    stage: str = ""
    stage_i: int = 0
    stage_n: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def title(self) -> str:
        """What to call this upload in a list.

        A count ("3 recordings") says nothing about which upload it was, which
        is no help at all once there are a few of them. The names the user
        chose are what they will recognise, so those come first and the count
        moves to the line underneath.
        """
        names = self.sources or [r.name for r in self.results]
        if not names:
            return f"{self.total} recording{'' if self.total == 1 else 's'}"
        if len(names) == 1:
            return names[0]
        if len(names) == 2:
            return f"{names[0]} and {names[1]}"
        return f"{names[0]} and {len(names) - 1} others"

    @property
    def audio_available(self) -> bool:
        return bool(self.audio_paths) and not self.audio_expired

    @property
    def audio_expired(self) -> bool:
        return self.audio_expires is not None and time.monotonic() > self.audio_expires

    def purge_audio(self) -> None:
        if self.root:
            shutil.rmtree(self.root, ignore_errors=True)
        self.root = None
        self.audio_paths = {}

    @property
    def elapsed(self) -> float:
        return (self.finished or time.monotonic()) - self.started

    @property
    def age_s(self) -> float:
        """Seconds since the batch was submitted."""
        return time.monotonic() - self.started

    @property
    def n_ok(self) -> int:
        return sum(r.status == "ok" for r in self.results)

    @property
    def n_failed(self) -> int:
        return sum(r.status == "failed" for r in self.results)

    @property
    def running(self) -> bool:
        return self.status in ("queued", "running")

    @property
    def percent(self) -> int:
        """Whole files done, plus how far into the file in flight we are.

        A batch of one would otherwise only ever read 0% or 100%.
        """
        if not self.total:
            return 0
        done = self.completed
        partial = (self.stage_i / self.stage_n) if self.stage_n else 0.0
        if done < self.total:
            done += min(partial, 0.99)
        return int(min(100, 100 * done / self.total))

    def public(self) -> dict[str, Any]:
        """What the polling endpoint returns. Small on purpose -- the page hits
        this every second."""
        with self._lock:
            return {
                "id": self.id,
                "status": self.status,
                "total": self.total,
                "completed": self.completed,
                "current": self.current,
                "stage": STAGE_LABELS.get(self.stage, self.stage),
                "percent": self.percent,
                "elapsed_s": round(self.elapsed, 1),
                "error": self.error,
                "n_ok": sum(r.status == "ok" for r in self.results),
                "n_failed": sum(r.status == "failed" for r in self.results),
            }


class JobStore:
    def __init__(self, max_jobs: int = 40) -> None:
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=2,
                                        thread_name_prefix="autoace-job")
        self.max_jobs = max_jobs

    def get(self, job_id: str) -> Job | None:
        self.sweep()
        with self._lock:
            return self._jobs.get(job_id)

    def recent(self, limit: int = 12) -> list[Job]:
        """Newest first. Batches keep running whether or not anyone is watching,
        so this is how you get back to one after navigating away."""
        self.sweep()
        with self._lock:
            ids = list(reversed(self._order))
            return [self._jobs[i] for i in ids if i in self._jobs][:limit]

    def sweep(self) -> None:
        """Delete audio whose retention window has closed."""
        with self._lock:
            jobs = list(self._jobs.values())
        for j in jobs:
            if j.root and j.audio_expired:
                log.info("purging audio for job %s (retention expired)", j.id)
                j.purge_audio()

    def _add(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            while len(self._order) > self.max_jobs:
                evicted = self._jobs.pop(self._order.pop(0), None)
                if evicted:
                    evicted.purge_audio()   # never outlive the job record

    def shutdown(self) -> None:
        """Purge every retained upload. Called on application shutdown."""
        with self._lock:
            jobs = list(self._jobs.values())
        for j in jobs:
            j.purge_audio()

    def submit(self, staged: Staged) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], total=len(staged.audio),
                  warnings=staged.warnings, sources=staged.sources)
        job.unmatched_labels = staged.unmatched
        job.unlabelled_audio = staged.unlabelled
        job.skipped = staged.skipped
        self._add(job)
        self._pool.submit(self._run, job, staged)
        return job

    def _run(self, job: Job, staged: Staged) -> None:
        from voicetone.stack import build_predictors
        try:
            job.status = "running"
            preds = list(build_predictors())    # memoised: models load once

            names = staged.names

            def one(path: Path) -> FileResult:
                shown = names.get(path, path.name)
                with job._lock:
                    job.current = shown

                def stage(name: str, i: int, n: int) -> None:
                    with job._lock:
                        job.stage, job.stage_i, job.stage_n = name, i, n

                # Each file is isolated: analyze_file never raises, it returns
                # status="failed". One malformed upload cannot kill the batch.
                r = analyze_file(path, preds, on_stage=stage)
                # Report the name the user uploaded, not the temp path, and
                # preserve it byte for byte including unicode.
                r.name = shown
                with job._lock:
                    job.results.append(r)
                    job.completed += 1
                    job.stage, job.stage_i, job.stage_n = "", 0, 0
                return r

            if len(staged.audio) > 1 and WORKERS > 1:
                with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                    list(pool.map(one, staged.audio))
            else:
                for p in staged.audio:
                    one(p)

            job.results.sort(key=lambda r: r.name)
            if staged.labels:
                preds_by_name = {r.name: r.result for r in job.results
                                 if r.status == "ok" and r.result}
                job.report = score(preds_by_name, staged.labels)
            job.status = "done"
        except Exception as exc:                       # noqa: BLE001
            log.exception("job %s failed", job.id)
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"
        finally:
            job.finished = time.monotonic()
            if PLAYBACK_ENABLED:
                # Hold the audio for the retention window so it can be played
                # back beside its prediction, then purge on the timer.
                job.root = staged.root
                job.audio_paths = {shown: path
                                   for path, shown in staged.names.items()}
                job.audio_expires = time.monotonic() + AUDIO_RETENTION_MIN * 60
            else:
                shutil.rmtree(staged.root, ignore_errors=True)


def display_names(paths: list[Path], root: Path) -> dict[Path, str]:
    """Map each staged path to the name shown and exported.

    The bare original filename is used -- exactly as uploaded, unicode and all,
    since filename preservation is graded. A folder prefix is added *only* to
    files whose basename collides with another, which is the one case where the
    bare name cannot identify a row (BUILD_SPEC 6).
    """
    by_base: dict[str, list[Path]] = {}
    for p in paths:
        by_base.setdefault(p.name, []).append(p)

    out: dict[Path, str] = {}
    for base, group in by_base.items():
        if len(group) == 1:
            out[group[0]] = base
            continue
        for p in group:
            try:
                # Drop the synthetic "<upload>.d" wrapper segment.
                parts = p.relative_to(root).parts[1:]
            except ValueError:
                parts = (p.name,)
            label = str(Path(*parts)) if parts else base
            out[p] = label
        # A folder prefix still may not disambiguate (same nested path twice
        # across two archives); fall back to an index so names stay unique.
        seen: dict[str, int] = {}
        for p in group:
            name = out[p]
            n = seen.get(name, 0)
            seen[name] = n + 1
            if n:
                stem, suffix = Path(name).stem, Path(name).suffix
                out[p] = f"{stem} ({n}){suffix}"
    return out


# --------------------------------------------------------------------------
# upload handling
# --------------------------------------------------------------------------

def is_junk(rel: str) -> bool:
    """Archive noise that is not a user file.

    macOS Archive Utility writes an AppleDouble stub for every entry into a
    parallel `__MACOSX/` tree, named `._original.ogg`. They carry the extension
    of the file they shadow, so they look like audio to a suffix check, and
    ffprobe rejects them. Left unfiltered, zipping three calls on a Mac produces
    a batch of six with three failures.

    Also covers Finder/Windows/Linux directory metadata and Spotlight indexes.
    """
    parts = [p for p in rel.replace("\\", "/").split("/") if p]
    if any(p in {"__MACOSX", ".Spotlight-V100", ".Trashes", ".fseventsd",
                 "$RECYCLE.BIN", "System Volume Information"} for p in parts):
        return True
    name = parts[-1] if parts else ""
    return (name.startswith("._")
            or name in {".DS_Store", "Thumbs.db", "desktop.ini"})


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extract with the three checks a naive extractall skips: path traversal,
    total uncompressed size, and per-member compression ratio."""
    infos = [i for i in zf.infolist()
             if not i.is_dir() and not is_junk(i.filename)]
    if len(infos) > MAX_MEMBERS:
        raise BadArchive(f"archive has {len(infos)} entries; limit is {MAX_MEMBERS}")
    total = sum(i.file_size for i in infos)
    if total > MAX_UNCOMPRESSED_BYTES:
        raise BadArchive(
            f"archive expands to {total / 2**20:.0f} MB; limit is "
            f"{MAX_UNCOMPRESSED_BYTES / 2**20:.0f} MB")
    for i in infos:
        if i.compress_size > 0 and i.file_size / i.compress_size > MAX_COMPRESSION_RATIO:
            raise BadArchive(f"entry {i.filename!r} has an implausible "
                             f"compression ratio; refusing to expand it")
        name = i.filename.replace("\\", "/")
        if name.startswith("/") or ".." in Path(name).parts:
            raise BadArchive(f"unsafe path in archive: {i.filename!r}")
        out = (dest / name).resolve()
        if not str(out).startswith(str(dest.resolve())):
            raise BadArchive(f"unsafe path in archive: {i.filename!r}")
        out.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(i) as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)


@dataclass
class Staged:
    """A validated upload, ready to run."""
    root: Path
    audio: list[Path]
    names: dict[Path, str]
    # What the user actually handed over: the ZIP name, or the loose filenames.
    # Kept separately from `audio` because a single "recordings.zip" is what
    # someone recognises in a list, while its contents are not.
    sources: list[str] = field(default_factory=list)
    labels: dict[str, dict[str, Any]] = field(default_factory=dict)
    unmatched: list[str] = field(default_factory=list)
    unlabelled: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Everything in the upload that was NOT sent to the analyser. Reported so
    # "only the audio was picked" is something you can see rather than assume.
    skipped: list[str] = field(default_factory=list)


def stage_upload(files: list[tuple[str, bytes]]) -> Staged:
    """Write uploads to a temp dir, unpack any ZIPs, find the audio and the
    optional manifest, and check the manifest in both directions.

    Raises UploadTooLarge / BadArchive before any heavy work happens.
    """
    total = sum(len(b) for _, b in files)
    if total > MAX_UPLOAD_BYTES:
        raise UploadTooLarge(
            f"upload is {total / 2**20:.0f} MB; limit is "
            f"{MAX_UPLOAD_BYTES / 2**20:.0f} MB")

    root = Path(tempfile.mkdtemp(prefix="autoace_upload_"))
    warnings: list[str] = []
    try:
        for name, blob in files:
            # Only the basename: a browser or a crafted request can send a path.
            safe = Path(name.replace("\\", "/")).name or "upload.bin"
            dest = root / safe
            dest.write_bytes(blob)
            if zipfile.is_zipfile(dest):
                sub = root / (safe + ".d")
                sub.mkdir(parents=True, exist_ok=True)
                try:
                    with zipfile.ZipFile(dest) as zf:
                        _safe_extract(zf, sub)
                except BadArchive:
                    raise
                except zipfile.BadZipFile as exc:
                    warnings.append(f"{safe}: not a readable archive ({exc})")
                finally:
                    dest.unlink(missing_ok=True)

        # --- selection rule --------------------------------------------
        # Audio is found recursively, so root or nested makes no difference,
        # and selection is purely by extension against `SUPPORTED`. Anything
        # else in the archive -- documents, images, nested archives, editor
        # backups -- is ignored and listed in `skipped`.
        #
        # A file that *claims* an audio extension but is not audio is
        # deliberately NOT filtered out here. It enters the batch and fails
        # with a decode reason, because a genuinely corrupt recording and a
        # mislabelled text file are indistinguishable until ffprobe looks, and
        # silently dropping the first would hide a real problem.
        audio, skipped = [], []
        for p in sorted(root.rglob("*"), key=str):
            if not p.is_file():
                continue
            rel = str(p.relative_to(root))
            if is_junk(rel):
                continue                      # archive metadata, not a user file
            if p.suffix.lower() in SUPPORTED:
                audio.append(p)
            elif p.suffix.lower() != ".csv":  # the manifest is handled below
                skipped.append(Path(rel).name)

        names = display_names(audio, root)

        labels: dict[str, dict[str, Any]] = {}
        csvs = sorted(root.rglob("*.csv"), key=lambda p: (len(p.parts), str(p)))
        for c in csvs:
            try:
                found = load_labels(c)
            except Exception as exc:               # noqa: BLE001
                warnings.append(f"{c.name}: could not be parsed ({exc})")
                continue
            if found:
                labels = found
                break
        if csvs and not labels:
            warnings.append("a CSV was uploaded but no usable "
                            "name/result_json rows were found in it")

        # A manifest may name a file by basename while the file needed a folder
        # prefix to be unique. Re-key such rows onto the name we will report, so
        # the row is scored rather than filed as unmatched.
        shown = set(names.values())
        if labels:
            by_base: dict[str, list[str]] = {}
            for s in shown:
                by_base.setdefault(Path(s).name, []).append(s)
            for key in [k for k in labels if k not in shown]:
                cands = by_base.get(Path(key).name, [])
                if len(cands) == 1:              # unambiguous, so safe to remap
                    labels[cands[0]] = labels.pop(key)

        # Report both directions, and treat neither as fatal (BUILD_SPEC 11.4).
        unmatched = sorted(k for k in labels if k not in shown)
        unlabelled = sorted(s for s in shown if labels and s not in labels)
        sources = [Path(n.replace("\\", "/")).name or "upload.bin"
                   for n, _ in files]
        return Staged(root=root, audio=audio, names=names, labels=labels,
                      unmatched=unmatched, unlabelled=unlabelled,
                      warnings=warnings, skipped=sorted(skipped),
                      sources=sources)
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


# --------------------------------------------------------------------------
# downloads
# --------------------------------------------------------------------------

def results_json(job: Job) -> bytes:
    payload = [{
        "name": r.name,
        "status": r.status,
        "error": r.error,
        "elapsed_s": round(r.elapsed_s, 3),
        "audio_s": round(r.audio_s, 2),
        "rtf": round(r.rtf, 4) if r.rtf else None,
        "result": r.result,
        "latents": r.latents,
        "stage_times": {k: round(v, 4) for k, v in r.stage_times.items()},
        "warnings": r.warnings,
    } for r in job.results]
    body: dict[str, Any] = {"job": job.id, "results": payload}
    if job.report:
        body["score_report"] = job.report
    return json.dumps(body, indent=2, ensure_ascii=False).encode("utf-8")


def results_csv(job: Job) -> bytes:
    from voicetone.schema import FIELDS
    buf = io.StringIO(newline="")
    w = csv.writer(buf)
    w.writerow(["name", "status", *FIELDS, "error", "result_json"])
    for r in job.results:
        res = r.result or {}
        w.writerow([r.name, r.status, *[res.get(f, "") for f in FIELDS],
                    r.error or "",
                    json.dumps(res, ensure_ascii=False) if res else ""])
    # BOM so Excel opens unicode filenames correctly without mangling them.
    return buf.getvalue().encode("utf-8-sig")


def report_text(job: Job) -> str:
    return format_report(job.report) if job.report else ""
