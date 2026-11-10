"""Small command-boundary regressions; archive inspection never fits models."""

import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

from kahm_demo.contract import CONTRACT_SHA256


PROJECT = Path(__file__).resolve().parents[1]
TUTORIAL_COMMANDS = (("kahm_demo", "tutorial"), ("examples.guided_tutorial",))


class TutorialCommandLineTests(unittest.TestCase):
    def run_tutorial(self, command, history=None):
        arguments = [sys.executable, "-B", "-m", *command, "--project", str(PROJECT)]
        if history is not None:
            arguments.extend(("--history", *history))
        return subprocess.run(arguments, cwd=PROJECT, text=True, capture_output=True, timeout=30)

    def test_supported_decimals_exponents_and_default_preserve_history(self):
        cases = (
            (None, (1, 2)),
            (("0", "1"), (0, 3)),
            ((".2500", "+0.7500"), (1, 2)),
            (("2.50e-1", "7.5E-1"), (1, 2)),
        )
        for command in TUTORIAL_COMMANDS:
            for history, expected in cases:
                with self.subTest(command=command, history=history):
                    result = self.run_tutorial(command, history)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stderr, "")
                    self.assertIn("archive_identity_only", result.stdout)
                    labels = re.search(r'"state_labels":\s*\[\s*(\d+),\s*(\d+)\s*\]', result.stdout)
                    self.assertIsNotNone(labels, result.stdout)
                    self.assertEqual(tuple(map(int, labels.groups())), expected)

    def test_off_support_tokens_are_not_rounded_into_the_support(self):
        for command in TUTORIAL_COMMANDS:
            for token in ("0.25000000000000001", "2.5000000000000001e-1", "1e-400"):
                self.assertIn(float(token), (0.0, 0.25))
                for position in (0, 1):
                    with self.subTest(command=command, token=token, position=position):
                        history = ["0.25", "0.75"]
                        history[position] = token
                        result = self.run_tutorial(command, history)
                        self.assertEqual(result.returncode, 2, result.stderr)
                        self.assertEqual(result.stdout, "")
                        self.assertIn("exactly in the finite support", result.stderr)
                        self.assertNotIn("Traceback", result.stderr)

    def test_nonfinite_and_malformed_tokens_are_clean_errors(self):
        for command in TUTORIAL_COMMANDS:
            for token in ("NaN", "inf", "Infinity", "not-a-number", "1/0"):
                with self.subTest(command=command, token=token):
                    result = self.run_tutorial(command, (token, "0.75"))
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertEqual(result.stdout, "")
                    self.assertIn("history observations must be finite numbers", result.stderr)
                    self.assertNotIn("Traceback", result.stderr)


class EncoderCommandLineTests(unittest.TestCase):
    def test_nonobject_runtime_metadata_reports_a_controlled_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "malformed-encoder.json"
            path.write_text(json.dumps({
                "schema": "kahm.encoder_bank.v1",
                "contract_sha256": CONTRACT_SHA256,
                "status": "development_bank_unselected",
                "runtime": None,
            }), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-B", "-m", "kahm_demo", "encode", "--encoder", str(path),
                 "--history", "0.25", "0.75"],
                cwd=PROJECT, text=True, capture_output=True, timeout=30,
            )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertIn("error: encoder runtime metadata must be an object", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
