"""Score saved responses against fictional banking rules without API calls."""

import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path

DATASET = Path(__file__).parent / "datasets" / "banking_cases.csv"
ACTIONS = {"post", "reject", "review", "noop"}


def parse_response(response):
    """Require exact fields and integer minor units, excluding booleans."""
    if isinstance(response, str):
        response = json.loads(response)
    if not isinstance(response, dict) or set(response) != {"action", "amount_minor"}:
        raise ValueError("Response must contain exactly action and amount_minor")
    if not isinstance(response["action"], str) or response["action"] not in ACTIONS:
        raise ValueError("Unknown action")
    if type(response["amount_minor"]) is not int or response["amount_minor"] < 0:
        raise ValueError("amount_minor must be a non-negative integer")
    return response


def score_response(response, reference):
    expected = parse_response(reference)
    try:
        actual = parse_response(response)
    except (ValueError, TypeError):
        return dict(
            schema_valid=False,
            action_correct=False,
            amount_correct=False,
            correct=False,
        )
    action = actual["action"] == expected["action"]
    amount = actual["amount_minor"] == expected["amount_minor"]
    return dict(
        schema_valid=True,
        action_correct=action,
        amount_correct=amount,
        correct=action and amount,
    )


def validate_cases(cases):
    if not cases:
        raise ValueError("Dataset is empty")
    ids = set()
    for case in cases:
        for field in ("id", "category", "user_input", "reference_context", "reference"):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise ValueError(f"Missing case field: {field}")
        if case["id"] in ids:
            raise ValueError(f"Duplicate case ID: {case['id']}")
        ids.add(case["id"])
        parse_response(case["reference"])


def load_cases(path=DATASET):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        cases = list(csv.DictReader(handle))
    validate_cases(cases)
    return cases


def ragas_rows(cases):
    """Rows suitable for ragas.EvaluationDataset.from_list()."""
    validate_cases(cases)
    return [
        {
            "user_input": c["user_input"],
            "reference": c["reference"],
            "reference_contexts": [c["reference_context"]],
        }
        for c in cases
    ]


def as_ragas_metric():
    """Use the same deterministic scorer in a native Ragas experiment."""
    from ragas.metrics.discrete import discrete_metric
    from ragas.metrics.result import MetricResult

    @discrete_metric(
        name="banking_exact_match", allowed_values=["correct", "incorrect"]
    )
    def banking_exact_match(prediction: str, reference: str):
        result = score_response(prediction, reference)
        return MetricResult(
            value="correct" if result["correct"] else "incorrect",
            reason=json.dumps(result, sort_keys=True),
        )

    return banking_exact_match


def compare(cases, predictions):
    """Use the full case set for every model; missing answers count as failures."""
    validate_cases(cases)
    case_ids = {c["id"] for c in cases}
    by_model = defaultdict(dict)
    for row in predictions:
        if not isinstance(row, dict):
            raise ValueError("Each prediction must be an object")
        model, case_id = row.get("model"), row.get("id")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Each prediction needs a non-empty model name")
        if not isinstance(case_id, str) or case_id not in case_ids:
            raise ValueError(f"Unknown case ID: {case_id}")
        if case_id in by_model[model]:
            raise ValueError(f"Duplicate prediction for {model}: {case_id}")
        by_model[model][case_id] = row.get("response")
    if not by_model:
        raise ValueError("No predictions supplied")
    results = []
    for model, answers in sorted(by_model.items()):
        scored = []
        categories = defaultdict(lambda: {"total": 0, "correct": 0})
        for case in cases:
            score = score_response(answers.get(case["id"]), case["reference"])
            scored.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "missing": case["id"] not in answers,
                    **score,
                }
            )
            categories[case["category"]]["total"] += 1
            categories[case["category"]]["correct"] += int(score["correct"])
        correct = sum(row["correct"] for row in scored)
        results.append(
            {
                "model": model,
                "total": len(cases),
                "submitted": len(answers),
                "missing": len(cases) - len(answers),
                "correct": correct,
                "accuracy": correct / len(cases),
                "schema_errors": sum(
                    not r["schema_valid"] and not r["missing"] for r in scored
                ),
                "categories": dict(categories),
                "cases": scored,
            }
        )
    return {"benchmark": "synthetic-banking-v1", "models": results}


def render_html(report):
    """A self-contained report, with untrusted names and case IDs escaped."""
    rows, details = [], []
    for model in report["models"]:
        name = html.escape(model["model"])
        rows.append(
            f"<tr><td>{name}</td><td>{model['correct']}/{model['total']}</td>"
            f"<td>{model['accuracy']:.1%}</td><td>{model['missing']}</td>"
            f"<td>{model['schema_errors']}</td></tr>"
        )
        categories = "".join(
            f"<li>{html.escape(c)}: {s['correct']}/{s['total']}</li>"
            for c, s in sorted(model["categories"].items())
        )
        failures = "".join(
            f"<li>{html.escape(c['id'])}: "
            f"{'missing' if c['missing'] else 'incorrect response'}</li>"
            for c in model["cases"]
            if not c["correct"]
        )
        details.append(
            f"<h2>{name}</h2><h3>Categories</h3><ul>{categories}</ul>"
            f"<h3>Cases to review</h3><ul>{failures or '<li>None</li>'}</ul>"
        )
    return (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Banking evaluation comparison</title><style>"
        "body{font:16px system-ui;max-width:1000px;margin:40px auto;padding:20px;color:#172033}"
        "table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:12px;"
        "border-bottom:1px solid #ccd3df}th{background:#edf2fa}h2{margin-top:36px}"
        "</style><h1>Banking evaluation comparison</h1>"
        "<p>Fictional rules and synthetic cases. Results describe only supplied responses; "
        "they do not establish production readiness. Missing answers count as failures.</p>"
        "<table><thead><tr><th>Model / run</th><th>Correct</th><th>Accuracy</th>"
        "<th>Missing</th><th>Schema errors</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        + "".join(details)
        + "</html>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument(
        "--predictions", type=Path, help="Saved responses in JSONL format"
    )
    parser.add_argument("--output", type=Path, default=Path("banking-comparison.json"))
    parser.add_argument("--html", type=Path, help="Optional self-contained HTML report")
    parser.add_argument(
        "--export-ragas", type=Path, help="Export Ragas-compatible dataset JSON"
    )
    args = parser.parse_args()
    if not args.predictions and not args.export_ragas:
        parser.error("Provide --predictions or --export-ragas")
    try:
        cases = load_cases(args.dataset)
        if args.export_ragas:
            args.export_ragas.write_text(
                json.dumps(ragas_rows(cases), indent=2), encoding="utf-8"
            )
        if args.predictions:
            with args.predictions.open(encoding="utf-8") as handle:
                predictions = [json.loads(line) for line in handle if line.strip()]
            report = compare(cases, predictions)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            if args.html:
                args.html.write_text(render_html(report), encoding="utf-8")
            for model in report["models"]:
                print(
                    f"{model['model']}: {model['correct']}/{model['total']} "
                    f"({model['accuracy']:.1%}), missing={model['missing']}"
                )
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
