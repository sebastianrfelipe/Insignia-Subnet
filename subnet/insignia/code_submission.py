"""
Insignia Code Submission — Reproducible Researcher Submissions

Under the base researcher protocol a miner submits only a serialized model
artifact (``ModelSubmission.model_artifact``). The validator deserializes it and
scores its predictions against the proprietary benchmark. Nothing forces the
artifact to actually be the output of a genuine, runnable training/inference
pipeline: a miner could ship a hand-tuned lookup table, an artifact lifted from
another miner, or a blob whose behavior is impossible to audit.

This module adds a **code submission** alongside the model, mirroring the
approach used by Metanova Labs' NOVA subnet (SN68): miners ship signed source
code and validators re-execute it in an isolated sandbox to reproduce the
result before trusting (and scoring) it.

Design (provider-agnostic, no Docker/bittensor hard dependency):

  1. ``build_code_bundle`` packages the miner's inference/training source (plus,
     optionally, the serialized model) into a deterministic ``tar.gz`` with a
     manifest of per-file SHA-256 hashes. The miner attaches the bundle bytes,
     the bundle hash, and the entrypoint name to its ``ModelSubmission``.

  2. ``CodeBundleVerifier`` checks size/file limits, recomputes the manifest
     hash, confirms the declared entrypoint exists, and statically scans the
     source for patterns that are disallowed inside the sandbox (the validator
     is about to *execute* this code).

  3. ``SandboxRunner`` extracts the bundle into a throwaway working directory and
     runs ``python3 <entrypoint>`` as a subprocess under CPU/address-space/file
     resource limits, with a wall-clock budget and a minimal environment.
     Following the NOVA convention the entrypoint reads its input from
     ``input.json`` and writes its result to ``result.json`` in the working
     directory. Network egress is dropped via ``unshare -n`` when available.

  4. ``ReproducibilityChecker`` drives the sandbox on the validator's evaluation
     features and compares the reproduced predictions against the predictions
     obtained directly from the submitted artifact, yielding a reproducibility
     score in ``[0, 1]``. A submission whose code cannot reproduce its own
     artifact is unverifiable and is gated out of scoring.

  5. ``CodeFingerprinter`` normalizes source (strips comments/whitespace) and
     hashes it to detect verbatim or lightly-edited code plagiarism across
     miners, analogous to ``incentive.ModelFingerprinter`` for artifacts.

The module is stdlib-only so it can be imported by the protocol, the miner, the
validator, and the offline simulation/tuning stack without pulling in numpy,
sklearn, or the chain SDK. Numerical comparison degrades gracefully when numpy
is unavailable.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Fixed epoch used for every tar entry so the archive bytes are a pure function
# of the file contents (deterministic hashing across miners/validators).
_DETERMINISTIC_MTIME = 0

# Entrypoint I/O contract (mirrors NOVA's /workspace/input.json -> result.json).
DEFAULT_INPUT_FILENAME = "input.json"
DEFAULT_OUTPUT_FILENAME = "result.json"
DEFAULT_ENTRYPOINT = "inference.py"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CodeBundleConfig:
    """Limits applied when building and verifying a code bundle."""

    max_bundle_bytes: int = 5 * 1024 * 1024      # 5 MB compressed archive
    max_uncompressed_bytes: int = 50 * 1024 * 1024  # 50 MB extracted (zip-bomb guard)
    max_files: int = 200
    max_file_bytes: int = 20 * 1024 * 1024
    allowed_extensions: Tuple[str, ...] = (
        ".py", ".txt", ".json", ".md", ".cfg", ".toml", ".yaml", ".yml",
        ".joblib", ".pkl", ".onnx", ".bin", ".npy",
    )
    # Static-analysis denylist: substrings that must not appear in *.py source.
    # The validator executes this code, so network/process-spawn/filesystem
    # escapes are rejected before the sandbox ever runs.
    disallowed_source_patterns: Tuple[str, ...] = (
        "import socket",
        "import requests",
        "import urllib",
        "import http.client",
        "urllib.request",
        "socket.socket",
        "subprocess.",
        "import subprocess",
        "os.system",
        "os.popen",
        "pty.spawn",
        "ctypes",
        "import ctypes",
        "/etc/passwd",
        "shutil.rmtree",
    )


@dataclass
class SandboxConfig:
    """Resource/IO limits for executing an untrusted entrypoint."""

    time_budget_seconds: float = 30.0
    max_memory_mb: int = 2048
    max_output_bytes: int = 8 * 1024 * 1024
    cpu_seconds: int = 30
    input_filename: str = DEFAULT_INPUT_FILENAME
    output_filename: str = DEFAULT_OUTPUT_FILENAME
    drop_network: bool = True
    python_executable: str = sys.executable or "python3"


# ---------------------------------------------------------------------------
# Bundle data model + packaging
# ---------------------------------------------------------------------------

@dataclass
class CodeBundle:
    """
    A packaged, content-addressed code submission.

    ``archive`` is the raw ``tar.gz`` bytes a miner places on the wire.
    ``bundle_hash`` is the SHA-256 of those bytes (the value committed/signed).
    ``manifest`` records per-file hashes + metadata for auditing without
    unpacking.
    """

    archive: bytes
    bundle_hash: str
    entrypoint: str
    manifest: Dict[str, Any]

    @property
    def size_bytes(self) -> int:
        return len(self.archive)

    def to_submission_fields(self) -> Dict[str, Any]:
        """Map onto the code-related fields of ``ModelSubmission``."""
        return {
            "code_bundle": self.archive,
            "code_bundle_hash": self.bundle_hash,
            "code_entrypoint": self.entrypoint,
            "code_manifest": self.manifest,
        }


def _normalize_arcname(name: str) -> str:
    name = name.replace(os.sep, "/").lstrip("./")
    while name.startswith("../"):
        name = name[3:]
    return name


def build_code_bundle(
    files: Dict[str, bytes],
    entrypoint: str = DEFAULT_ENTRYPOINT,
    config: CodeBundleConfig | None = None,
) -> CodeBundle:
    """
    Package ``{relative_path: contents}`` into a deterministic ``tar.gz`` bundle.

    The archive is byte-reproducible (sorted entries, fixed mtime/uid/gid) so the
    same source always yields the same ``bundle_hash`` — a prerequisite for
    commit-reveal style signing and for cross-validator agreement.
    """
    cfg = config or CodeBundleConfig()
    if entrypoint not in files:
        raise ValueError(f"entrypoint {entrypoint!r} not present in bundle files")
    if len(files) > cfg.max_files:
        raise ValueError(f"too many files: {len(files)} > {cfg.max_files}")

    manifest_files: List[Dict[str, Any]] = []
    buf = io.BytesIO()
    # gzip mtime=0 keeps the compressed stream deterministic across runs.
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tar:
        gz = getattr(tar, "fileobj", None)
        if gz is not None and hasattr(gz, "mtime"):
            gz.mtime = 0  # type: ignore[attr-defined]
        for raw_name in sorted(files):
            name = _normalize_arcname(raw_name)
            data = files[raw_name]
            ext = os.path.splitext(name)[1].lower()
            if ext and ext not in cfg.allowed_extensions:
                raise ValueError(f"disallowed file extension: {name}")
            if len(data) > cfg.max_file_bytes:
                raise ValueError(f"file too large: {name} ({len(data)} bytes)")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mtime = _DETERMINISTIC_MTIME
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
            manifest_files.append({
                "path": name,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            })

    archive = buf.getvalue()
    if len(archive) > cfg.max_bundle_bytes:
        raise ValueError(
            f"bundle too large: {len(archive)} > {cfg.max_bundle_bytes} bytes"
        )

    bundle_hash = hashlib.sha256(archive).hexdigest()
    manifest = {
        "entrypoint": entrypoint,
        "n_files": len(manifest_files),
        "total_source_bytes": sum(f["size_bytes"] for f in manifest_files),
        "files": manifest_files,
        "bundle_hash": bundle_hash,
    }
    return CodeBundle(
        archive=archive,
        bundle_hash=bundle_hash,
        entrypoint=entrypoint,
        manifest=manifest,
    )


def build_code_bundle_from_dir(
    source_dir: str,
    entrypoint: str = DEFAULT_ENTRYPOINT,
    config: CodeBundleConfig | None = None,
    extra_files: Dict[str, bytes] | None = None,
) -> CodeBundle:
    """Walk ``source_dir`` and package its files (plus ``extra_files``)."""
    cfg = config or CodeBundleConfig()
    files: Dict[str, bytes] = {}
    for root, _dirs, names in os.walk(source_dir):
        for fn in names:
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, source_dir)
            ext = os.path.splitext(rel)[1].lower()
            if ext and ext not in cfg.allowed_extensions:
                continue
            if "__pycache__" in rel:
                continue
            with open(full, "rb") as fh:
                files[_normalize_arcname(rel)] = fh.read()
    if extra_files:
        for k, v in extra_files.items():
            files[_normalize_arcname(k)] = v
    return build_code_bundle(files, entrypoint=entrypoint, config=cfg)


def extract_code_bundle(
    archive: bytes,
    dest_dir: str,
    config: CodeBundleConfig | None = None,
) -> List[str]:
    """
    Safely extract a bundle into ``dest_dir``.

    Guards against path traversal, symlinks/hardlinks, oversized payloads, and
    decompression bombs. Returns the list of extracted relative paths.
    """
    cfg = config or CodeBundleConfig()
    extracted: List[str] = []
    total = 0
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        members = tar.getmembers()
        if len(members) > cfg.max_files:
            raise ValueError("bundle contains too many members")
        for m in members:
            if not (m.isfile() or m.isdir()):
                raise ValueError(f"unsupported tar member type: {m.name}")
            name = _normalize_arcname(m.name)
            target = os.path.realpath(os.path.join(dest_dir, name))
            if not (target == os.path.realpath(dest_dir)
                    or target.startswith(os.path.realpath(dest_dir) + os.sep)):
                raise ValueError(f"path traversal blocked: {m.name}")
            total += max(m.size, 0)
            if total > cfg.max_uncompressed_bytes:
                raise ValueError("uncompressed bundle exceeds limit (zip bomb?)")
            # Extraction must use the *normalized* name, not the raw archive name
            # (extractall honors ``m.name``), so members extract where we vetted.
            m.name = name
        try:
            # Python 3.12+ : defense-in-depth on top of the manual vetting above.
            tar.extractall(dest_dir, members=members, filter="data")
        except TypeError:
            tar.extractall(dest_dir, members=members)  # noqa: S202 - members vetted
        for m in members:
            if m.isfile():
                extracted.append(_normalize_arcname(m.name))
    return extracted


# ---------------------------------------------------------------------------
# Verification (structure + static safety)
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    ok: bool
    reason: str = ""
    bundle_hash: str = ""
    manifest_ok: bool = False
    static_scan_ok: bool = False
    flagged_patterns: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)


class CodeBundleVerifier:
    """
    Validates a received code bundle before it is executed.

    Checks, in order:
      - declared bundle hash matches the SHA-256 of the received bytes;
      - archive/extraction limits hold (size, file count, traversal, bombs);
      - the declared entrypoint exists in the bundle;
      - every per-file manifest hash matches the extracted contents;
      - no ``*.py`` file contains a disallowed (sandbox-escaping) pattern, and
        all sources parse as valid Python.
    """

    def __init__(self, config: CodeBundleConfig | None = None):
        self.config = config or CodeBundleConfig()

    def verify(
        self,
        archive: bytes,
        entrypoint: str,
        declared_hash: str = "",
        manifest: Dict[str, Any] | None = None,
    ) -> VerificationResult:
        bundle_hash = hashlib.sha256(archive).hexdigest()
        if declared_hash and declared_hash != bundle_hash:
            return VerificationResult(
                ok=False,
                reason="bundle hash mismatch (declared != received)",
                bundle_hash=bundle_hash,
            )
        if len(archive) > self.config.max_bundle_bytes:
            return VerificationResult(
                ok=False, reason="bundle exceeds max size", bundle_hash=bundle_hash
            )

        tmp = tempfile.mkdtemp(prefix="insignia_verify_")
        try:
            try:
                files = extract_code_bundle(archive, tmp, self.config)
            except Exception as exc:  # noqa: BLE001 - surfaced as rejection reason
                return VerificationResult(
                    ok=False, reason=f"extraction failed: {exc}", bundle_hash=bundle_hash
                )

            if entrypoint not in files:
                return VerificationResult(
                    ok=False,
                    reason=f"entrypoint {entrypoint!r} missing from bundle",
                    bundle_hash=bundle_hash,
                    files=files,
                )

            manifest_ok = self._verify_manifest(tmp, files, manifest)
            if manifest is not None and not manifest_ok:
                return VerificationResult(
                    ok=False,
                    reason="manifest hash mismatch",
                    bundle_hash=bundle_hash,
                    files=files,
                )

            flagged = self._static_scan(tmp, files)
            if flagged:
                return VerificationResult(
                    ok=False,
                    reason=f"static safety scan flagged: {', '.join(flagged)}",
                    bundle_hash=bundle_hash,
                    manifest_ok=manifest_ok,
                    static_scan_ok=False,
                    flagged_patterns=flagged,
                    files=files,
                )

            return VerificationResult(
                ok=True,
                reason="ok",
                bundle_hash=bundle_hash,
                manifest_ok=manifest_ok,
                static_scan_ok=True,
                files=files,
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _verify_manifest(
        self, root: str, files: List[str], manifest: Dict[str, Any] | None
    ) -> bool:
        if not manifest:
            return False
        declared = {f["path"]: f["sha256"] for f in manifest.get("files", [])}
        if set(declared) != set(files):
            return False
        for rel, want in declared.items():
            with open(os.path.join(root, rel), "rb") as fh:
                if hashlib.sha256(fh.read()).hexdigest() != want:
                    return False
        return True

    def _static_scan(self, root: str, files: List[str]) -> List[str]:
        flagged: List[str] = []
        for rel in files:
            if not rel.endswith(".py"):
                continue
            with open(os.path.join(root, rel), "r", encoding="utf-8", errors="replace") as fh:
                src = fh.read()
            for pat in self.config.disallowed_source_patterns:
                if pat in src:
                    flagged.append(f"{rel}:{pat}")
            try:
                ast.parse(src)
            except SyntaxError as exc:
                flagged.append(f"{rel}:syntax_error({exc.lineno})")
        return flagged


# ---------------------------------------------------------------------------
# Sandboxed execution
# ---------------------------------------------------------------------------

@dataclass
class SandboxResult:
    ok: bool
    output: Optional[Dict[str, Any]] = None
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    timed_out: bool = False
    return_code: Optional[int] = None
    reason: str = ""


def _build_rlimit_preexec(cfg: SandboxConfig):
    """Return a preexec_fn that applies POSIX resource limits, or None."""
    try:
        import resource  # noqa: WPS433 - POSIX only
    except ImportError:
        return None

    def _set_limits():  # pragma: no cover - runs in the forked child
        mem = cfg.max_memory_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        except (ValueError, OSError):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cfg.cpu_seconds, cfg.cpu_seconds + 1))
        except (ValueError, OSError):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        except (ValueError, OSError):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except (ValueError, OSError):
            pass
        os.setsid()

    return _set_limits


def _interpreter_site_paths() -> List[str]:
    """
    Resolve the running interpreter's site-package directories.

    Passed to the sandbox via PYTHONPATH so the entrypoint can import the same
    third-party stack (numpy, joblib, ...) the validator uses, independent of
    the scrubbed sandbox ``HOME``. Falls back to ``sys.path`` if ``site`` is
    unavailable.
    """
    paths: List[str] = []
    try:
        import site
        try:
            paths.extend(site.getsitepackages())
        except AttributeError:
            pass
        try:
            user = site.getusersitepackages()
            if user:
                paths.append(user)
        except AttributeError:
            pass
    except Exception:  # noqa: BLE001
        pass
    # Only fall back to *package* dirs on sys.path — never the project/source
    # tree — so the sandbox can import deps without importing validator code.
    paths.extend(
        p for p in sys.path
        if p and ("site-packages" in p or "dist-packages" in p)
    )
    seen, ordered = set(), []
    for p in paths:
        if p and p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


_UNSHARE_NET_OK: Optional[bool] = None


def _unshare_net_available() -> bool:
    """
    Probe (once) whether ``unshare -r -n`` actually works in this environment.

    ``unshare`` may be installed yet fail at runtime when user/network
    namespaces are disabled (common in CI/containers). Probing avoids wrapping
    real runs in a command that would always fail.
    """
    global _UNSHARE_NET_OK
    if _UNSHARE_NET_OK is not None:
        return _UNSHARE_NET_OK
    if not shutil.which("unshare"):
        _UNSHARE_NET_OK = False
        return False
    try:
        probe = subprocess.run(
            ["unshare", "-r", "-n", "true"],
            capture_output=True, timeout=10, check=False,
        )
        _UNSHARE_NET_OK = probe.returncode == 0
    except (OSError, subprocess.SubprocessError):
        _UNSHARE_NET_OK = False
    if not _UNSHARE_NET_OK:
        logger.warning(
            "sandbox: network-namespace isolation unavailable (unshare -n failed); "
            "falling back to resource limits + scrubbed env only"
        )
    return _UNSHARE_NET_OK


def _network_wrapper(cfg: SandboxConfig) -> List[str]:
    """Prefix argv with ``unshare -r -n`` to drop network egress when possible."""
    if not cfg.drop_network:
        return []
    if _unshare_net_available():
        return ["unshare", "-r", "-n"]
    return []


class SandboxRunner:
    """
    Executes a bundle's entrypoint in an isolated subprocess.

    Isolation layers (best-effort, no container runtime required):
      - throwaway working directory containing only the extracted bundle;
      - POSIX rlimits on address space, CPU time, process count, and core dumps;
      - wall-clock timeout with process-group kill;
      - minimal scrubbed environment;
      - network namespace drop via ``unshare -n`` when available.

    The entrypoint reads ``input.json`` and writes ``result.json`` in its CWD.
    """

    def __init__(self, config: SandboxConfig | None = None,
                 bundle_config: CodeBundleConfig | None = None):
        self.config = config or SandboxConfig()
        self.bundle_config = bundle_config or CodeBundleConfig()

    def run(self, archive: bytes, entrypoint: str,
            input_payload: Dict[str, Any]) -> SandboxResult:
        workdir = tempfile.mkdtemp(prefix="insignia_sandbox_")
        try:
            try:
                files = extract_code_bundle(archive, workdir, self.bundle_config)
            except Exception as exc:  # noqa: BLE001
                return SandboxResult(ok=False, reason=f"extraction failed: {exc}")
            if entrypoint not in files:
                return SandboxResult(ok=False, reason="entrypoint missing")

            with open(os.path.join(workdir, self.config.input_filename), "w") as fh:
                json.dump(input_payload, fh)
            out_path = os.path.join(workdir, self.config.output_filename)

            # ``-B`` (no .pyc) + a scrubbed env isolate the run while still
            # importing the same third-party stack (numpy/joblib/...) the
            # validator uses. We deliberately do not pass ``-I``/``-s`` because
            # those disable site-packages; instead we pin the interpreter's own
            # site paths via PYTHONPATH so imports resolve regardless of the
            # sandbox HOME.
            argv = _network_wrapper(self.config) + [
                self.config.python_executable, "-B", entrypoint,
            ]
            env = {
                "PATH": "/usr/bin:/bin",
                "PYTHONHASHSEED": "0",
                "HOME": workdir,
                "TMPDIR": workdir,
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "PYTHONPATH": os.pathsep.join(_interpreter_site_paths()),
                "INSIGNIA_INPUT": self.config.input_filename,
                "INSIGNIA_OUTPUT": self.config.output_filename,
            }

            t0 = time.perf_counter()
            timed_out = False
            try:
                proc = subprocess.run(
                    argv,
                    cwd=workdir,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=self.config.time_budget_seconds,
                    preexec_fn=_build_rlimit_preexec(self.config),
                    check=False,
                )
                rc = proc.returncode
                stdout, stderr = proc.stdout, proc.stderr
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                rc = None
                stdout = exc.stdout or ""
                stderr = (exc.stderr or "") + "\n[sandbox] wall-clock budget exceeded"
                if isinstance(stdout, bytes):
                    stdout = stdout.decode("utf-8", "replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode("utf-8", "replace")
            duration = time.perf_counter() - t0

            if timed_out:
                return SandboxResult(
                    ok=False, stdout=stdout[:4096], stderr=stderr[-4096:],
                    duration_seconds=duration, timed_out=True, reason="timeout",
                )
            if rc != 0:
                return SandboxResult(
                    ok=False, stdout=stdout[:4096], stderr=stderr[-4096:],
                    duration_seconds=duration, return_code=rc,
                    reason=f"non-zero exit ({rc})",
                )
            if not os.path.exists(out_path):
                return SandboxResult(
                    ok=False, stdout=stdout[:4096], stderr=stderr[-4096:],
                    duration_seconds=duration, return_code=rc,
                    reason="entrypoint did not write result file",
                )
            if os.path.getsize(out_path) > self.config.max_output_bytes:
                return SandboxResult(
                    ok=False, duration_seconds=duration, return_code=rc,
                    reason="result file exceeds max size",
                )
            try:
                with open(out_path, "r") as fh:
                    output = json.load(fh)
            except Exception as exc:  # noqa: BLE001
                return SandboxResult(
                    ok=False, stdout=stdout[:4096], stderr=stderr[-4096:],
                    duration_seconds=duration, return_code=rc,
                    reason=f"result file not valid JSON: {exc}",
                )
            return SandboxResult(
                ok=True, output=output, stdout=stdout[:4096], stderr=stderr[-4096:],
                duration_seconds=duration, return_code=rc, reason="ok",
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Reproducibility check
# ---------------------------------------------------------------------------

@dataclass
class ReproducibilityResult:
    ok: bool
    score: float = 0.0
    agreement: float = 0.0
    n_compared: int = 0
    reason: str = ""
    sandbox: Optional[SandboxResult] = None


class ReproducibilityChecker:
    """
    Confirms a researcher's submitted artifact is the genuine output of its
    submitted code by re-running the code in the sandbox and comparing.

    The validator passes the same feature matrix it scored the artifact on and
    the predictions it obtained directly from the deserialized artifact. The
    bundle's entrypoint produces its own predictions from that input; the two
    prediction vectors must agree for the submission to be considered
    reproducible. A low score means the artifact does not correspond to its
    code (tampering, a non-runnable blob, or hard-coded outputs) and the
    validator can gate the submission out of scoring.
    """

    def __init__(self, runner: SandboxRunner | None = None,
                 agreement_threshold: float = 0.99,
                 tolerance: float = 1e-4):
        self.runner = runner or SandboxRunner()
        self.agreement_threshold = agreement_threshold
        self.tolerance = tolerance

    def check(
        self,
        archive: bytes,
        entrypoint: str,
        features: Sequence[Sequence[float]],
        feature_names: Sequence[str],
        reference_predictions: Sequence[float],
    ) -> ReproducibilityResult:
        payload = {
            "features": [list(map(float, row)) for row in features],
            "feature_names": list(feature_names),
        }
        sandbox = self.runner.run(archive, entrypoint, payload)
        if not sandbox.ok:
            return ReproducibilityResult(
                ok=False, reason=f"sandbox: {sandbox.reason}", sandbox=sandbox
            )

        output = sandbox.output or {}
        preds = output.get("predictions")
        if not isinstance(preds, list) or not preds:
            return ReproducibilityResult(
                ok=False, reason="no 'predictions' list in result", sandbox=sandbox
            )

        ref = list(map(float, reference_predictions))
        n = min(len(preds), len(ref))
        if n == 0:
            return ReproducibilityResult(
                ok=False, reason="empty prediction overlap", sandbox=sandbox
            )

        agreement = _agreement(
            [float(p) for p in preds[:n]], ref[:n], self.tolerance
        )
        ok = agreement >= self.agreement_threshold
        return ReproducibilityResult(
            ok=ok,
            score=round(agreement, 6),
            agreement=round(agreement, 6),
            n_compared=n,
            reason="ok" if ok else "predictions diverge from artifact",
            sandbox=sandbox,
        )


def _agreement(a: List[float], b: List[float], tol: float) -> float:
    """Fraction of element pairs that match within ``tol`` (relative+absolute)."""
    matches = 0
    for x, y in zip(a, b):
        denom = max(abs(x), abs(y), 1.0)
        if abs(x - y) <= tol * denom + tol:
            matches += 1
    return matches / max(len(a), 1)


# ---------------------------------------------------------------------------
# Code fingerprinting (plagiarism detection)
# ---------------------------------------------------------------------------

_COMMENT_RE = re.compile(r"#.*$", re.MULTILINE)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_source(src: str) -> str:
    """
    Normalize Python source for plagiarism comparison.

    Strips comments and collapses whitespace; when the source parses, docstrings
    are dropped too via AST round-trip so that re-commenting or reflowing code
    does not defeat duplicate detection.
    """
    try:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef, ast.Module)):
                body = getattr(node, "body", [])
                if (body and isinstance(body[0], ast.Expr)
                        and isinstance(getattr(body[0], "value", None), ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    body.pop(0)
        try:
            src = ast.unparse(tree)  # py3.9+
        except AttributeError:
            pass
    except SyntaxError:
        pass
    src = _COMMENT_RE.sub("", src)
    return _WHITESPACE_RE.sub(" ", src).strip()


def code_fingerprint(files: Dict[str, bytes]) -> str:
    """Fingerprint the *.py sources of a bundle, invariant to formatting."""
    parts = []
    for name in sorted(files):
        if not name.endswith(".py"):
            continue
        text = files[name].decode("utf-8", "replace")
        parts.append(normalize_source(text))
    blob = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def fingerprint_archive(archive: bytes, config: CodeBundleConfig | None = None) -> str:
    """Extract a bundle in-memory and return its normalized code fingerprint."""
    files: Dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for m in tar.getmembers():
            if m.isfile() and m.name.endswith(".py"):
                fh = tar.extractfile(m)
                if fh is not None:
                    files[_normalize_arcname(m.name)] = fh.read()
    return code_fingerprint(files)


class CodeFingerprinter:
    """
    Detects duplicate or lightly-edited source across miners.

    Mirrors ``incentive.ModelFingerprinter`` but operates on normalized source
    rather than artifact bytes, so a miner that re-serializes a stolen model
    (changing its artifact hash) is still caught if it ships the same code.
    Correlated submitters can be made to share rewards, removing the incentive
    to plagiarize code.
    """

    def __init__(self):
        self._fingerprints: Dict[str, str] = {}

    def compute(self, archive: bytes) -> str:
        return fingerprint_archive(archive)

    def is_duplicate(self, miner_uid: str, fingerprint: str) -> bool:
        return any(
            uid != miner_uid and fp == fingerprint
            for uid, fp in self._fingerprints.items()
        )

    def find_duplicates(self, miner_uid: str, fingerprint: str) -> List[str]:
        return [
            uid for uid, fp in self._fingerprints.items()
            if uid != miner_uid and fp == fingerprint
        ]

    def register(self, miner_uid: str, fingerprint: str) -> None:
        self._fingerprints[miner_uid] = fingerprint
