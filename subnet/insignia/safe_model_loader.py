"""
Insignia Safe Model Loader — allowlisted deserialization of untrusted artifacts.

Researcher miners ship their trained model as a serialized ``model_artifact``
(``joblib.dump`` of an sklearn estimator/pipeline). Validators and trader miners
must turn those bytes back into a live object to score / inference them — but the
bytes arrive over the network from an *untrusted* peer.

``joblib.load`` (like ``pickle.load``) executes whatever the ``__reduce__`` of
the pickle says to, so loading a hostile artifact is remote code execution. A
malicious researcher could ship an artifact whose unpickling runs
``os.system(...)`` and own every validator and paired trader that scores it.

This module replaces the raw ``joblib.load`` with ``safe_load_model``, which
drives joblib's own numpy-aware unpickler but overrides ``find_class`` to a
strict **allowlist**: only inert container types and classes from the numerical
/ ML stack (numpy, scipy, sklearn, joblib) may be reconstructed. The dangerous
gadget sources a pickle RCE needs — ``os``/``posix``/``nt``, ``subprocess``,
``sys``, ``builtins.eval``/``exec``/``getattr``/``__import__``, ``operator``,
``functools`` — are never in the allowlist, so a ``REDUCE``/``STACK_GLOBAL`` that
references them raises ``UnsafeArtifactError`` *before* any callable is invoked.

The loader **fails closed**: anything it cannot load under the allowlist (an
unexpected global, a corrupt archive, an unsupported joblib version) raises
rather than falling back to the unsafe path, so callers can reject the
submission and score it zero.

Caveat (documented, not a TODO): allowlisted unpickling narrows the attack
surface to the numerical stack rather than eliminating it — a novel gadget built
purely from allowlisted numpy/sklearn classes is not categorically impossible.
It is the standard pragmatic mitigation for "must accept a pickled sklearn
model", and it composes with the reproducibility sandbox in
``insignia.code_submission`` for defense in depth. The categorical fix is to move
to a non-executable artifact format (e.g. ONNX); this keeps the existing joblib
wire format working while closing the RCE.
"""

from __future__ import annotations

import inspect
import io
import pickle
from typing import Any, Tuple


class UnsafeArtifactError(Exception):
    """Raised when an artifact references a global outside the allowlist or
    cannot be deserialized safely. Callers should treat this as a rejected,
    zero-scored submission rather than a crash."""


# ---------------------------------------------------------------------------
# Allowlist policy
# ---------------------------------------------------------------------------

# Top-level package names whose classes are considered safe to reconstruct.
# Everything an sklearn pipeline (scaler + GBDT/forest/linear model + numpy
# arrays) needs lives under these; none of them expose a process/network/eval
# primitive reachable through a pickle reduce.
_ALLOWED_MODULE_ROOTS: frozenset[str] = frozenset({
    "numpy",
    "scipy",
    "sklearn",
    "joblib",
})

# Inert builtins only: container/value *types*, never callables that can run
# code or traverse attributes (no eval/exec/getattr/setattr/open/__import__/
# compile/globals/breakpoint/memoryview).
_ALLOWED_BUILTINS: frozenset[str] = frozenset({
    "object", "list", "tuple", "dict", "set", "frozenset",
    "bytearray", "bytes", "complex", "int", "float", "bool", "str",
    "slice", "range", "NoneType",
})

# Specific (module, name) pairs needed for generic object reconstruction.
# ``copyreg._reconstructor`` / ``__newobj__`` receive the target class as a
# *separate* global argument, which is itself routed through ``find_class`` and
# therefore independently allowlisted — so permitting them does not widen the
# surface. ``defaultdict``'s ``default_factory`` is likewise a separate global.
_ALLOWED_EXPLICIT: frozenset[Tuple[str, str]] = frozenset({
    ("copyreg", "_reconstructor"),
    ("copyreg", "__newobj__"),
    ("copyreg", "__newobj_ex__"),
    ("collections", "OrderedDict"),
    ("collections", "defaultdict"),
})


def _is_allowed_global(module: str, name: str) -> bool:
    if (module, name) in _ALLOWED_EXPLICIT:
        return True
    if module in ("builtins", "__builtin__"):
        return name in _ALLOWED_BUILTINS
    root = module.split(".", 1)[0]
    return root in _ALLOWED_MODULE_ROOTS


# ---------------------------------------------------------------------------
# Restricted unpickler (joblib-compatible)
# ---------------------------------------------------------------------------

def _restricted_numpy_unpickler_class():
    """Build a ``NumpyUnpickler`` subclass that enforces the allowlist.

    Subclassing joblib's unpickler (a ``pickle._Unpickler``, which honors a
    Python-level ``find_class`` override) preserves byte-for-byte compatibility
    with ``joblib.dump`` output — including memmapped numpy arrays — while
    gating every global through the allowlist.
    """
    from joblib.numpy_pickle import NumpyUnpickler  # lazy: joblib is runtime-only

    class _RestrictedNumpyUnpickler(NumpyUnpickler):
        def find_class(self, module: str, name: str):  # noqa: D401
            if not _is_allowed_global(module, name):
                raise UnsafeArtifactError(
                    f"artifact references disallowed global {module}.{name}"
                )
            return super().find_class(module, name)

    return _RestrictedNumpyUnpickler


def _build_unpickler(cls, fobj):
    """Instantiate the restricted unpickler across joblib version skew.

    joblib 1.5+ added a required ``ensure_native_byte_order`` positional to
    ``NumpyUnpickler.__init__``; older releases use
    ``(filename, file_handle, mmap_mode=None)``. Adapt by inspecting the
    signature so the loader works on either.
    """
    params = inspect.signature(cls.__init__).parameters
    if "ensure_native_byte_order" in params:
        return cls("", fobj, True, mmap_mode=None)
    return cls("", fobj, mmap_mode=None)


def _open_artifact_fileobject(fobj):
    """Return a context manager yielding a (possibly decompressing) file object.

    Reuses joblib's own compression detection so transparently-compressed
    artifacts still load. Decompression executes no pickle opcodes; only the
    subsequent restricted unpickling can construct objects.
    """
    import contextlib

    try:  # joblib >= ~1.3
        from joblib.numpy_pickle import _validate_fileobject_and_memmap

        @contextlib.contextmanager
        def _ctx():
            with _validate_fileobject_and_memmap(fobj, "", None) as (fo, _mmap):
                yield fo

        return _ctx()
    except Exception:  # noqa: BLE001 - fall through to older / no-op path
        pass

    try:  # older joblib
        from joblib.numpy_pickle_utils import _read_fileobject

        return _read_fileobject(fobj, "", None)
    except Exception:  # noqa: BLE001 - no compression wrapper available

        @contextlib.contextmanager
        def _passthrough():
            yield fobj

        return _passthrough()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def safe_load_model(artifact: bytes) -> Any:
    """Deserialize an untrusted ``joblib``-serialized model under the allowlist.

    Returns the reconstructed estimator/pipeline. Raises ``UnsafeArtifactError``
    if the artifact references a disallowed global, is malformed, or cannot be
    loaded safely — callers must reject such submissions (score 0), never fall
    back to ``joblib.load``.
    """
    if not isinstance(artifact, (bytes, bytearray, memoryview)):
        raise UnsafeArtifactError(
            f"artifact must be bytes, got {type(artifact).__name__}"
        )

    try:
        cls = _restricted_numpy_unpickler_class()
    except Exception as exc:  # noqa: BLE001 - joblib unavailable/incompatible
        raise UnsafeArtifactError(f"safe loader unavailable: {exc}") from exc

    fobj = io.BytesIO(bytes(artifact))
    try:
        with _open_artifact_fileobject(fobj) as fo:
            unpickler = _build_unpickler(cls, fo)
            return unpickler.load()
    except UnsafeArtifactError:
        raise
    except (pickle.UnpicklingError, EOFError, ValueError, TypeError,
            AttributeError, ImportError, MemoryError) as exc:
        raise UnsafeArtifactError(f"artifact failed safe deserialization: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - never leak an unexpected loader fault
        raise UnsafeArtifactError(
            f"artifact rejected by safe loader: {type(exc).__name__}: {exc}"
        ) from exc
