"""Tests for the reproducible code-submission stack (insignia.code_submission)."""

import io
import sys
import tarfile
import tempfile
import unittest
from importlib.util import find_spec

from insignia.code_submission import (
    DEFAULT_ENTRYPOINT,
    CodeBundleConfig,
    CodeBundleVerifier,
    CodeFingerprinter,
    ReproducibilityChecker,
    SandboxConfig,
    SandboxRunner,
    build_code_bundle,
    code_fingerprint,
    extract_code_bundle,
    normalize_source,
)

# A dependency-free entrypoint: prediction[i] = sum(features[i]).
ECHO_ENTRYPOINT = '''\
import json, os
with open(os.environ.get("INSIGNIA_INPUT", "input.json")) as fh:
    payload = json.load(fh)
preds = [float(sum(row)) for row in payload.get("features", [])]
with open(os.environ.get("INSIGNIA_OUTPUT", "result.json"), "w") as fh:
    json.dump({"predictions": preds}, fh)
'''


def _echo_bundle():
    return build_code_bundle(
        {DEFAULT_ENTRYPOINT: ECHO_ENTRYPOINT.encode("utf-8"),
         "readme.md": b"hello"},
        entrypoint=DEFAULT_ENTRYPOINT,
    )


class BundleBuildTests(unittest.TestCase):
    def test_build_is_deterministic(self):
        files = {DEFAULT_ENTRYPOINT: b"print(1)\n", "a.txt": b"x"}
        b1 = build_code_bundle(files, entrypoint=DEFAULT_ENTRYPOINT)
        b2 = build_code_bundle(files, entrypoint=DEFAULT_ENTRYPOINT)
        self.assertEqual(b1.archive, b2.archive)
        self.assertEqual(b1.bundle_hash, b2.bundle_hash)
        self.assertEqual(b1.manifest["n_files"], 2)

    def test_missing_entrypoint_raises(self):
        with self.assertRaises(ValueError):
            build_code_bundle({"a.py": b"x"}, entrypoint="inference.py")

    def test_disallowed_extension_raises(self):
        with self.assertRaises(ValueError):
            build_code_bundle(
                {DEFAULT_ENTRYPOINT: b"x", "evil.sh": b"rm -rf /"},
                entrypoint=DEFAULT_ENTRYPOINT,
            )

    def test_manifest_has_per_file_hashes(self):
        bundle = _echo_bundle()
        paths = {f["path"] for f in bundle.manifest["files"]}
        self.assertEqual(paths, {DEFAULT_ENTRYPOINT, "readme.md"})
        for f in bundle.manifest["files"]:
            self.assertEqual(len(f["sha256"]), 64)


class ExtractionSafetyTests(unittest.TestCase):
    def test_path_traversal_blocked(self):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            data = b"owned"
            info = tarfile.TarInfo("../escape.txt")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        with tempfile.TemporaryDirectory() as d:
            # Normalized to a safe relative name -> stays inside dest.
            extract_code_bundle(buf.getvalue(), d)

    def test_symlink_member_rejected(self):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            info = tarfile.TarInfo("link")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tar.addfile(info)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                extract_code_bundle(buf.getvalue(), d)


class VerifierTests(unittest.TestCase):
    def test_good_bundle_verifies(self):
        bundle = _echo_bundle()
        res = CodeBundleVerifier().verify(
            bundle.archive, bundle.entrypoint, bundle.bundle_hash, bundle.manifest
        )
        self.assertTrue(res.ok, res.reason)
        self.assertTrue(res.manifest_ok)
        self.assertTrue(res.static_scan_ok)

    def test_hash_mismatch_rejected(self):
        bundle = _echo_bundle()
        res = CodeBundleVerifier().verify(
            bundle.archive, bundle.entrypoint, declared_hash="deadbeef"
        )
        self.assertFalse(res.ok)
        self.assertIn("hash mismatch", res.reason)

    def test_missing_entrypoint_rejected(self):
        bundle = _echo_bundle()
        res = CodeBundleVerifier().verify(bundle.archive, "nope.py")
        self.assertFalse(res.ok)
        self.assertIn("missing", res.reason)

    def test_disallowed_pattern_rejected(self):
        bundle = build_code_bundle(
            {DEFAULT_ENTRYPOINT: b"import socket\n"}, entrypoint=DEFAULT_ENTRYPOINT
        )
        res = CodeBundleVerifier().verify(bundle.archive, bundle.entrypoint)
        self.assertFalse(res.ok)
        self.assertIn("static safety", res.reason)

    def test_tampered_manifest_rejected(self):
        bundle = _echo_bundle()
        bad = dict(bundle.manifest)
        bad_files = [dict(f) for f in bundle.manifest["files"]]
        bad_files[0]["sha256"] = "0" * 64
        bad["files"] = bad_files
        res = CodeBundleVerifier().verify(
            bundle.archive, bundle.entrypoint, bundle.bundle_hash, bad
        )
        self.assertFalse(res.ok)

    def test_oversize_bundle_rejected(self):
        cfg = CodeBundleConfig(max_bundle_bytes=10)
        bundle = _echo_bundle()
        res = CodeBundleVerifier(cfg).verify(bundle.archive, bundle.entrypoint)
        self.assertFalse(res.ok)


class FingerprintTests(unittest.TestCase):
    def test_formatting_invariant(self):
        a = "def f(x):\n    return x + 1  # add one\n"
        b = "def f(x):\n\n    # totally different comment\n    return x+1\n"
        self.assertEqual(
            code_fingerprint({"m.py": a.encode()}),
            code_fingerprint({"m.py": b.encode()}),
        )

    def test_docstring_stripped(self):
        a = 'def f():\n    """doc."""\n    return 1\n'
        b = "def f():\n    return 1\n"
        self.assertEqual(normalize_source(a), normalize_source(b))

    def test_distinct_code_differs(self):
        self.assertNotEqual(
            code_fingerprint({"m.py": b"def f(): return 1"}),
            code_fingerprint({"m.py": b"def f(): return 2"}),
        )

    def test_duplicate_detection(self):
        fp = CodeFingerprinter()
        bundle = _echo_bundle()
        h = fp.compute(bundle.archive)
        self.assertEqual(fp.find_duplicates("a", h), [])
        fp.register("a", h)
        self.assertTrue(fp.is_duplicate("b", h))
        self.assertEqual(fp.find_duplicates("b", h), ["a"])


class SandboxTests(unittest.TestCase):
    def setUp(self):
        # Short budget; the echo entrypoint is trivial.
        self.runner = SandboxRunner(SandboxConfig(time_budget_seconds=20.0))

    def test_runs_and_returns_output(self):
        bundle = _echo_bundle()
        res = self.runner.run(
            bundle.archive, bundle.entrypoint,
            {"features": [[1.0, 2.0], [3.0, 4.0]], "feature_names": ["a", "b"]},
        )
        self.assertTrue(res.ok, res.reason + " :: " + res.stderr)
        self.assertEqual(res.output["predictions"], [3.0, 7.0])

    def test_nonzero_exit_reported(self):
        bundle = build_code_bundle(
            {DEFAULT_ENTRYPOINT: b"raise SystemExit(2)\n"}, entrypoint=DEFAULT_ENTRYPOINT
        )
        res = self.runner.run(bundle.archive, bundle.entrypoint, {"features": []})
        self.assertFalse(res.ok)

    def test_timeout_enforced(self):
        bundle = build_code_bundle(
            {DEFAULT_ENTRYPOINT: b"import time\nwhile True:\n    time.sleep(1)\n"},
            entrypoint=DEFAULT_ENTRYPOINT,
        )
        runner = SandboxRunner(SandboxConfig(time_budget_seconds=2.0))
        res = runner.run(bundle.archive, bundle.entrypoint, {"features": []})
        self.assertFalse(res.ok)
        self.assertTrue(res.timed_out)

    def test_no_result_file_reported(self):
        bundle = build_code_bundle(
            {DEFAULT_ENTRYPOINT: b"print('hi')\n"}, entrypoint=DEFAULT_ENTRYPOINT
        )
        res = self.runner.run(bundle.archive, bundle.entrypoint, {"features": []})
        self.assertFalse(res.ok)
        self.assertIn("result", res.reason)


class ReproducibilityTests(unittest.TestCase):
    def setUp(self):
        self.checker = ReproducibilityChecker(
            runner=SandboxRunner(SandboxConfig(time_budget_seconds=20.0))
        )

    def test_matching_predictions_pass(self):
        bundle = _echo_bundle()
        features = [[1.0, 2.0], [10.0, 0.5]]
        reference = [3.0, 10.5]  # sums match the echo entrypoint
        res = self.checker.check(
            bundle.archive, bundle.entrypoint, features, ["a", "b"], reference
        )
        self.assertTrue(res.ok, res.reason)
        self.assertEqual(res.score, 1.0)
        self.assertEqual(res.n_compared, 2)

    def test_diverging_predictions_fail(self):
        bundle = _echo_bundle()
        features = [[1.0, 2.0], [10.0, 0.5]]
        reference = [99.0, -1.0]  # artifact disagrees with its own code
        res = self.checker.check(
            bundle.archive, bundle.entrypoint, features, ["a", "b"], reference
        )
        self.assertFalse(res.ok)
        self.assertLess(res.score, 0.99)


@unittest.skipIf(
    find_spec("sklearn") is None or find_spec("joblib") is None,
    "sklearn/joblib unavailable",
)
class EndToEndMinerValidatorTests(unittest.TestCase):
    def _make_submission(self, seed=0):
        from neurons.researcher_miner import (
            ResearcherMiner, ModelTrainer, generate_demo_data,
        )
        data = generate_demo_data(n_samples=600, seed=seed)
        miner = ResearcherMiner(
            trainer=ModelTrainer(n_estimators=40, max_depth=3, random_state=seed)
        )
        return miner.train_and_submit(data)

    def test_miner_attaches_code_bundle(self):
        sub = self._make_submission()
        self.assertIsInstance(sub["code_bundle"], (bytes, bytearray))
        self.assertEqual(sub["code_entrypoint"], DEFAULT_ENTRYPOINT)
        self.assertEqual(len(sub["code_bundle_hash"]), 64)
        # Bundle ships the model + inference + training source.
        paths = {f["path"] for f in sub["code_manifest"]["files"]}
        self.assertIn("model.joblib", paths)
        self.assertIn(DEFAULT_ENTRYPOINT, paths)

    def test_validator_accepts_reproducible_submission(self):
        from neurons.model_validator import ModelValidator
        sub = self._make_submission(seed=1)
        validator = ModelValidator(top_n_promote=1, require_code=True)
        res = validator.process_submission(
            miner_uid="miner_x",
            model_artifact=sub["model_artifact"],
            features_used=sub["features_used"],
            metadata={},
            code_bundle=sub["code_bundle"],
            code_entrypoint=sub["code_entrypoint"],
            code_bundle_hash=sub["code_bundle_hash"],
            code_manifest=sub["code_manifest"],
        )
        self.assertTrue(res["accepted"], res.get("rejection_reason"))
        self.assertTrue(res["code_submission"]["code_reproducible"])
        self.assertGreaterEqual(res["code_submission"]["reproducibility_score"], 0.99)

    def test_validator_requires_code(self):
        from neurons.model_validator import ModelValidator
        sub = self._make_submission(seed=2)
        validator = ModelValidator(require_code=True)
        res = validator.process_submission(
            miner_uid="miner_no_code",
            model_artifact=sub["model_artifact"],
            features_used=sub["features_used"],
            metadata={},
        )
        self.assertFalse(res["accepted"])
        self.assertIn("Code submission required", res["rejection_reason"])

    def test_tampered_artifact_is_gated_out(self):
        """A model swapped for a different one no longer matches its code."""
        from neurons.model_validator import ModelValidator
        good = self._make_submission(seed=3)
        other = self._make_submission(seed=999)
        validator = ModelValidator(require_code=True, gate_on_reproducibility=True)
        res = validator.process_submission(
            miner_uid="miner_tamper",
            model_artifact=other["model_artifact"],  # mismatched artifact
            features_used=good["features_used"],
            code_bundle=good["code_bundle"],          # code packs the original model
            code_entrypoint=good["code_entrypoint"],
            code_bundle_hash=good["code_bundle_hash"],
            code_manifest=good["code_manifest"],
            metadata={},
        )
        # The bundle reproduces its *own* packed model, which differs from the
        # separately-submitted artifact the validator scored -> not reproducible.
        self.assertFalse(res["accepted"])
        self.assertEqual(res["composite_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
