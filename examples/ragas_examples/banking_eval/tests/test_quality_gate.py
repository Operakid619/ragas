import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "evals.py"


class QualityGateTests(unittest.TestCase):
    def run_gate(self, threshold):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "report.json"
            result = subprocess.run([sys.executable, str(SCRIPT), "--predictions",
                str(SCRIPT.parent / "datasets/demo_predictions.jsonl"), "--output", str(output),
                "--min-accuracy", str(threshold)], capture_output=True, text=True)
            return result, json.loads(output.read_text()) if output.exists() else None

    def test_passing_gate(self):
        result, report = self.run_gate(0)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertTrue(report["models"])

    def test_failing_gate_keeps_report(self):
        result, report = self.run_gate(1)
        self.assertEqual(1, result.returncode, result.stderr)
        self.assertTrue(any(m["accuracy"] < 1 for m in report["models"]))

    def test_invalid_thresholds(self):
        for threshold in (-0.1, 1.1, "nan", "inf"):
            with self.subTest(threshold=threshold):
                result, report = self.run_gate(threshold)
                self.assertEqual(2, result.returncode)
                self.assertIsNone(report)

    def test_gate_requires_predictions(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--min-accuracy", "0.9"], capture_output=True)
        self.assertEqual(2, result.returncode)
