"""Tests for allowlisted untrusted-artifact deserialization.

These guard the fix for the critical RCE where a researcher miner's
``model_artifact`` was fed straight to ``joblib.load`` (pickle) in the validator
process. A hostile artifact must be rejected *without executing*, while a
genuine ``joblib.dump`` of an sklearn pipeline must still load and predict.
"""

import io
import os
import pickle
import subprocess
import tempfile
import unittest
from importlib.util import find_spec

from insignia.safe_model_loader import safe_load_model, UnsafeArtifactError

_HAS_STACK = all(find_spec(m) for m in ("numpy", "joblib", "sklearn"))


@unittest.skipUnless(_HAS_STACK, "numpy/joblib/sklearn required")
class SafeModelLoaderTests(unittest.TestCase):
    def _legit_artifact(self):
        import numpy as np
        import joblib
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import HistGradientBoostingClassifier

        X = np.random.RandomState(0).rand(150, 6)
        y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", HistGradientBoostingClassifier(max_iter=15)),
        ])
        pipe.fit(X, y)
        buf = io.BytesIO()
        joblib.dump(pipe, buf)
        return buf.getvalue(), pipe, X

    def test_legit_artifact_round_trips(self):
        import numpy as np
        artifact, pipe, X = self._legit_artifact()
        loaded = safe_load_model(artifact)
        self.assertTrue(
            np.allclose(pipe.predict_proba(X), loaded.predict_proba(X)),
            "safe-loaded model must predict identically to the original",
        )

    def test_compressed_artifact_round_trips(self):
        import numpy as np
        import joblib
        _, pipe, X = self._legit_artifact()
        buf = io.BytesIO()
        joblib.dump(pipe, buf, compress=3)
        loaded = safe_load_model(buf.getvalue())
        self.assertTrue(np.allclose(pipe.predict_proba(X), loaded.predict_proba(X)))

    def test_os_system_reduce_is_blocked_and_not_executed(self):
        sentinel = os.path.join(tempfile.gettempdir(), "insignia_safe_loader_pwned")
        if os.path.exists(sentinel):
            os.remove(sentinel)

        class Evil:
            def __reduce__(self):
                return (os.system, (f'echo pwned > "{sentinel}"',))

        with self.assertRaises(UnsafeArtifactError):
            safe_load_model(pickle.dumps(Evil()))
        self.assertFalse(
            os.path.exists(sentinel),
            "os.system payload executed during deserialization (RCE)",
        )

    def test_eval_reduce_is_blocked(self):
        class EvilEval:
            def __reduce__(self):
                return (eval, ("1 + 1",))

        with self.assertRaises(UnsafeArtifactError):
            safe_load_model(pickle.dumps(EvilEval()))

    def test_subprocess_reduce_is_blocked(self):
        class EvilSub:
            def __reduce__(self):
                return (subprocess.Popen, (["whoami"],))

        with self.assertRaises(UnsafeArtifactError):
            safe_load_model(pickle.dumps(EvilSub()))

    def test_non_bytes_rejected(self):
        with self.assertRaises(UnsafeArtifactError):
            safe_load_model("not bytes")

    def test_garbage_rejected(self):
        with self.assertRaises(UnsafeArtifactError):
            safe_load_model(b"not a pickle at all")


@unittest.skipUnless(_HAS_STACK and find_spec("pandas"), "validator stack required")
class ValidatorRejectsHostileArtifactTests(unittest.TestCase):
    def test_process_submission_rejects_hostile_artifact_without_rce(self):
        from neurons.model_validator import ModelValidator

        sentinel = os.path.join(tempfile.gettempdir(), "insignia_validator_pwned")
        if os.path.exists(sentinel):
            os.remove(sentinel)

        class Evil:
            def __reduce__(self):
                return (os.system, (f'echo pwned > "{sentinel}"',))

        validator = ModelValidator()
        res = validator.process_submission(
            miner_uid="attacker",
            model_artifact=pickle.dumps(Evil()),
            features_used=["feature_0", "feature_1"],
            metadata={},
        )
        self.assertFalse(res["accepted"])
        self.assertEqual(res["composite_score"], 0.0)
        self.assertFalse(
            os.path.exists(sentinel),
            "validator executed a hostile artifact (RCE)",
        )


if __name__ == "__main__":
    unittest.main()
