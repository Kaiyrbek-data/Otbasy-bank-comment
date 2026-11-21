# Retrieval evaluation

Evaluation has not been run in this workspace. Per instruction, no model,
embedding, API process, or test suite was started locally.

Run `python eval/evaluate.py` in the cloud environment after manually reviewing
the proposed acceptable IDs and the 15 near-scope refusal cases in `eval.json`.
The script writes both pre-deduplication and post-deduplication ranking metrics,
plus the low-threshold refusal sweep, to this file.

The notebook's existing 16-query test results are not a deduplication
comparison and are intentionally not presented as one.
