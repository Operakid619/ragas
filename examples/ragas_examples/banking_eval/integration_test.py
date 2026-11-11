"""Run separately in an environment with Ragas installed; makes no API calls."""

import unittest

from evals import as_ragas_metric, load_cases, ragas_rows

from ragas import EvaluationDataset


class RagasIntegrationTests(unittest.TestCase):
    def test_dataset_export_is_accepted_by_ragas(self):
        dataset = EvaluationDataset.from_list(ragas_rows(load_cases()))
        self.assertEqual(len(dataset), 12)

    def test_native_metric_accepts_correct_and_rejects_malformed_response(self):
        reference = load_cases()[0]["reference"]
        metric = as_ragas_metric()
        self.assertEqual(
            metric.score(prediction=reference, reference=reference).value, "correct"
        )
        self.assertEqual(
            metric.score(prediction="invalid", reference=reference).value, "incorrect"
        )


if __name__ == "__main__":
    unittest.main()
