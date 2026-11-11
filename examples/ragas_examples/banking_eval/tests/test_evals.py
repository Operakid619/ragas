import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "evals.py"
SPEC = importlib.util.spec_from_file_location("banking_evals", SOURCE)
evals = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evals)


class BankingEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.cases = evals.load_cases()
        self.expected = self.cases[0]["reference"]

    def prediction(self, case, model="model-a"):
        return {"model": model, "id": case["id"], "response": case["reference"]}

    def test_dataset_has_twelve_unique_cases_and_four_categories(self):
        self.assertEqual(len(self.cases), 12)
        self.assertEqual(len({c["id"] for c in self.cases}), 12)
        self.assertEqual(
            {c["category"] for c in self.cases},
            {"fees", "balances", "idempotency", "reversals"},
        )

    def test_exact_response_passes(self):
        self.assertTrue(evals.score_response(self.expected, self.expected)["correct"])

    def test_wrong_action_and_wrong_amount_are_distinguished(self):
        expected = json.loads(self.expected)
        amount = evals.score_response({**expected, "amount_minor": 1}, expected)
        action = evals.score_response({**expected, "action": "review"}, expected)
        self.assertTrue(amount["action_correct"])
        self.assertFalse(amount["amount_correct"])
        self.assertTrue(action["amount_correct"])
        self.assertFalse(action["action_correct"])

    def test_invalid_schema_fails_without_crashing(self):
        invalid = [
            None,
            "not-json",
            "[]",
            {},
            {"action": [], "amount_minor": 0},
            {"action": "post", "amount_minor": True},
            {"action": "post", "amount_minor": 1000.0},
            {"action": "post", "amount_minor": -1},
            {"action": "post", "amount_minor": "1000"},
            {"action": "post", "amount_minor": 1000, "extra": 1},
        ]
        for response in invalid:
            with self.subTest(response=response):
                self.assertFalse(
                    evals.score_response(response, self.expected)["schema_valid"]
                )

    def test_missing_predictions_count_against_full_denominator(self):
        report = evals.compare(self.cases, [self.prediction(self.cases[0])])
        model = report["models"][0]
        self.assertEqual(model["correct"], 1)
        self.assertEqual(model["missing"], 11)
        self.assertAlmostEqual(model["accuracy"], 1 / 12)
        self.assertEqual(model["schema_errors"], 0)

    def test_row_order_does_not_change_scores(self):
        predictions = [self.prediction(c) for c in self.cases]
        self.assertEqual(
            evals.compare(self.cases, predictions),
            evals.compare(self.cases, predictions[::-1]),
        )

    def test_duplicate_or_unknown_prediction_is_rejected(self):
        row = self.prediction(self.cases[0])
        for rows in ([row, row], [{**row, "id": "unknown"}]):
            with self.assertRaises(ValueError):
                evals.compare(self.cases, rows)

    def test_empty_or_duplicate_dataset_is_rejected(self):
        for cases in ([], [self.cases[0], self.cases[0]]):
            with self.assertRaises(ValueError):
                evals.compare(cases, [self.prediction(self.cases[0])])

    def test_empty_predictions_or_blank_model_is_rejected(self):
        for rows in ([], [self.prediction(self.cases[0], " ")], [None]):
            with self.assertRaises(ValueError):
                evals.compare(self.cases, rows)

    def test_html_escapes_model_names(self):
        report = evals.compare(
            self.cases, [self.prediction(self.cases[0], "<script>alert(1)</script>")]
        )
        rendered = evals.render_html(report)
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;", rendered)

    def test_ragas_export_keeps_context_and_reference(self):
        exported = evals.ragas_rows(self.cases)
        self.assertEqual(
            exported[0]["reference_contexts"], [self.cases[0]["reference_context"]]
        )
        self.assertEqual(exported[0]["reference"], self.expected)
        self.assertEqual(len(exported), 12)

    def test_cli_writes_json_html_and_ragas_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SOURCE),
                    "--predictions",
                    str(SOURCE.parent / "datasets/demo_predictions.jsonl"),
                    "--output",
                    str(root / "report.json"),
                    "--html",
                    str(root / "report.html"),
                    "--export-ragas",
                    str(root / "dataset.json"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root / "report.json").read_text())
            models = {m["model"]: m for m in report["models"]}
            self.assertEqual(models["fixture-reference"]["correct"], 12)
            self.assertEqual(models["fixture-with-errors"]["correct"], 7)
            self.assertEqual(models["fixture-with-errors"]["missing"], 1)
            self.assertEqual(models["fixture-with-errors"]["schema_errors"], 1)
            self.assertIn("<table>", (root / "report.html").read_text())
            self.assertEqual(len(json.loads((root / "dataset.json").read_text())), 12)


if __name__ == "__main__":
    unittest.main()
