# Banking evaluation workbench

An original example in Emmanuel's Ragas fork: compare saved model responses against
12 synthetic, fictional banking-rule cases. The upstream Ragas license and authorship
are retained. This example covers fee caps, balance boundaries, idempotent retries,
and reversals with unknown settlement status.

## Run without an API key

From the repository root:

```sh
make banking-eval-demo
make test-banking-eval
```

The demo writes a JSON report and a self-contained HTML comparison to
`_experiments/banking-eval/`. Open `comparison.html` locally. The two included
runs, `fixture-reference` and `fixture-with-errors`, are hand-authored test fixtures,
not measured performance of real models. Their expected scores are 12/12 and 7/12.

## Evaluate your own responses

Send each case's `reference_context` and `user_input` to the model under test.
Keep `reference` hidden from the model. Save a JSONL record per model and case:

```json
{"model":"your-model-or-run","id":"fee-01","response":{"action":"post","amount_minor":1000}}
```

The response may also be a string containing JSON. Only `action` and `amount_minor`
are accepted. Actions are `post`, `reject`, `review`, or `noop`; the amount must be
a non-negative integer in minor currency units. A case passes only when both fields
match. Invalid responses fail without stopping the comparison. Duplicate or unknown
case IDs are rejected. Every represented model is scored against all 12 cases;
missing responses count as failures, separately from schema errors. A model with
no records at all is not included.

```sh
uv run --no-project python examples/ragas_examples/banking_eval/evals.py \
  --predictions responses.jsonl --output comparison.json --html comparison.html
```

## Use with Ragas

`--export-ragas dataset.json` produces rows accepted by
`EvaluationDataset.from_list(json.load(open("dataset.json")))`. Each row includes
the question, reference answer, and reference contexts. Keep the source CSV IDs
when joining model outputs for comparison.

When Ragas is installed, `as_ragas_metric()` in `evals.py` returns a deterministic
`banking_exact_match` discrete metric for use in Ragas experiments. Its `score`
method takes `prediction` and `reference` strings and returns `correct` or
`incorrect` with field-level scoring details. The offline CLI and its regression
tests require only Python's standard library.

To run the two native integration checks in an isolated environment:

```sh
uv venv /tmp/ragas-banking-venv
uv pip install --python /tmp/ragas-banking-venv/bin/python -e . \
  -r examples/ragas_examples/banking_eval/requirements-integration.txt
RAGAS_DO_NOT_TRACK=true uv run --no-project \
  --python /tmp/ragas-banking-venv/bin/python \
  examples/ragas_examples/banking_eval/integration_test.py
```

The example records the LangChain versions used for validation because this
upstream checkout imports VertexAI community adapters removed in newer releases.
These example-specific constraints leave the project's core dependencies unchanged.
Neither integration check calls a model API.

These are example policies, not real bank policies or regulatory guidance. Twelve
cases are a demonstration set, not a general capability or production-readiness
benchmark. Extend the cases and review their references before using real results.
