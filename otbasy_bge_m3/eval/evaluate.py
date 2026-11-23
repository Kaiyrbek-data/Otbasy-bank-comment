"""Evaluate the existing retrieval index without rebuilding corpus embeddings."""

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import SETTINGS  # noqa: E402
from retriever import FAQRetriever  # noqa: E402


EVAL_PATH = ROOT / "eval" / "eval.json"
RESULTS_PATH = ROOT / "eval" / "results.md"
LOW_THRESHOLDS = tuple(
    sorted(
        {SETTINGS["confidence"]["low_threshold"]}
        | {round(value / 100, 2) for value in range(30, 61, 5)}
    )
)


def normalize_answer(answer):
    return re.sub(r"\s+", " ", answer.casefold()).strip()


def control_ids(document):
    question = document["question"].casefold()
    answer = document["answer"].casefold()
    reference = str(document.get("reference", "")).casefold()
    checks = (
        "сейчас владею долей" in question and "доля 25% в единственном жилье" in answer,
        "сейчас владею долей" in question and "доля 50% в единственном жилье" in answer,
        "продал свою квартиру" in question
        and "прошло два года" in question
        and "квартира, проданная в этот период, учитывается" in answer,
        "зарегистрирован в алматы два года." in question and "такой срок недостаточен" in answer,
        "не позднее пятнадцати рабочих дней со дня подачи заявления" in answer,
        "строка 9" in reference and "скачать документ с причиной отказа" in answer,
        "выезд на постоянное место жительства в другой населенный пункт является основанием для снятия"
        in answer,
        "умер супруг" in question and "очередность сохраняется" in answer,
        "семьи с ребенком с инвалидностью отдельную дополнительную комнату" in answer,
        "коммунальное жилье для этой категории предоставляет и распределяет акимат" in answer,
        "размер сертификата и перечень категорий получателей определяет маслихат" in answer,
        "не менее 20%" in answer and "детей-сирот и детей" in answer,
        "алименты на ребенка" in question and "такая выплата сама по себе не входит" in answer,
        "списки состоящих на учете нуждающихся" in answer and "ежемесячно" in answer,
        "государственная регистрация такого договора не требуется" in answer,
        "строка 177" in reference and "само по себе не ведет к снятию с учета" in answer,
    )
    return {f"control:{index:02d}" for index, matched in enumerate(checks, start=1) if matched}


def review_ids(document):
    answer = document["answer"].casefold()
    reference = str(document.get("reference", "")).casefold()
    ids = set()
    if "меньше установленного порога в 50%" in answer:
        ids.add("review:share-below-50")
    if "строка 9" in reference and "окончательный результат смотрите в разделе" in answer:
        ids.add("review:inventory-result")
    if "указанный срок превышает этот порог" in answer and "ст. 75 п. 5" in reference:
        ids.add("review:pregnancy-above-22")
    return ids


def document_ids(document, answer_labels):
    normalized = normalize_answer(document["answer"])
    return {
        answer_labels[normalized],
        *control_ids(document),
        *review_ids(document),
    }


def is_relevant(document, acceptable_ids, answer_labels):
    return bool(document_ids(document, answer_labels) & acceptable_ids)


def ranking_metrics(results, documents, acceptable_ids, answer_labels):
    if not results:
        return 0.0, 0.0, 0.0
    relevance = [
        is_relevant(document, acceptable_ids, answer_labels)
        for document in results[:10]
    ]
    hit_at_3 = float(any(relevance[:3]))
    reciprocal_rank = next(
        (1 / rank for rank, relevant in enumerate(relevance, start=1) if relevant),
        0.0,
    )
    relevant_count = sum(
        is_relevant(document, acceptable_ids, answer_labels)
        for document in documents
    )
    if relevant_count == 0:
        ndcg = 0.0
    else:
        dcg = sum(
            1 / math.log2(rank + 1)
            for rank, relevant in enumerate(relevance, start=1)
            if relevant
        )
        ideal_length = min(relevant_count, 10)
        ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_length + 1))
        ndcg = dcg / ideal_dcg
    return hit_at_3, reciprocal_rank, ndcg


def candidate_ids(entry, include_review_candidates):
    ids = set(entry["acceptable_ids"])
    if include_review_candidates:
        ids.update(entry.get("proposed_acceptable_ids", []))
    return ids


def mean(values):
    return sum(values) / len(values) if values else None


def percent(value):
    return "N/A" if value is None else f"{value * 100:.2f}%"


def is_found(matches, low_threshold):
    return bool(matches) and matches[0]["cosine"] >= low_threshold


def evaluate(retriever, entries, documents, include_review_candidates=False):
    normalized_answers = sorted({normalize_answer(document["answer"]) for document in documents})
    answer_labels = {
        answer: f"answer:{label}"
        for label, answer in enumerate(normalized_answers)
    }
    in_scope = [entry for entry in entries if entry["in_scope"]]
    out_of_scope = [
        entry
        for entry in entries
        if not entry["in_scope"]
        and (include_review_candidates or not entry.get("needs_review", False))
    ]

    ranked = {"without dedupe": {}, "with dedupe": {}}
    for entry in entries:
        for label, dedupe in (("without dedupe", False), ("with dedupe", True)):
            ranked[label][entry["case_id"]] = retriever.search(
                entry["query"],
                top_k=10,
                dedupe=dedupe,
            )

    ranking_results = {}
    for label, results_by_case in ranked.items():
        per_query = [
            ranking_metrics(
                results_by_case[entry["case_id"]],
                documents,
                candidate_ids(entry, include_review_candidates),
                answer_labels,
            )
            for entry in in_scope
        ]
        ranking_results[label] = tuple(
            mean([metrics[index] for metrics in per_query])
            for index in range(3)
        )

    threshold_results = []
    for label, results_by_case in ranked.items():
        for low_threshold in LOW_THRESHOLDS:
            in_scope_found = [
                is_found(results_by_case[entry["case_id"]], low_threshold)
                for entry in in_scope
                if results_by_case[entry["case_id"]]
            ]
            out_of_scope_found = [
                is_found(results_by_case[entry["case_id"]], low_threshold)
                for entry in out_of_scope
                if results_by_case[entry["case_id"]]
            ]
            false_refusals = (
                sum(not found for found in in_scope_found) / len(in_scope_found)
                if in_scope_found
                else None
            )
            correct_refusals = (
                sum(not found for found in out_of_scope_found) / len(out_of_scope_found)
                if out_of_scope_found
                else None
            )
            threshold_results.append(
                (label, low_threshold, false_refusals, correct_refusals, len(out_of_scope_found))
            )

    return ranking_results, threshold_results, len(in_scope), len(out_of_scope)


def render_results(ranking_results, threshold_results, in_scope_count, ood_count, pending_ood):
    now = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Retrieval evaluation",
        "",
        f"- Generated (UTC): {now}",
        f"- App version: `{SETTINGS['app_version']}`",
        f"- Index version: `{SETTINGS['index_version']}`",
        "- Index embeddings were not rebuilt; this script only encodes eval queries and searches the packaged index.",
        f"- In-scope queries scored: {in_scope_count}",
        f"- Confirmed out-of-scope queries scored: {ood_count}",
        f"- Out-of-scope candidates awaiting review: {pending_ood}",
        "",
        "## Ranking metrics",
        "",
        "| Retrieval mode | Hit@3 | MRR@10 | nDCG@10 |",
        "|---|---:|---:|---:|",
    ]
    for label, metrics in ranking_results.items():
        lines.append(
            f"| {label} | {percent(metrics[0])} | {percent(metrics[1])} | {percent(metrics[2])} |"
        )
    lines.extend(
        [
            "",
            "## Refusal threshold sweep",
            "",
            f"High threshold is fixed at `{SETTINGS['confidence']['high_threshold']:.2f}`. "
            "Refusal is `best cosine < low threshold`.",
            "",
            "| Retrieval mode | Low threshold | False refusal (in-scope) | Correct refusal (out-of-scope) | Confirmed OOD n |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, low, false_refusals, correct_refusals, count in threshold_results:
        lines.append(
            f"| {label} | {low:.2f} | {percent(false_refusals)} | "
            f"{percent(correct_refusals)} | {count} |"
        )
    lines.extend(
        [
            "",
            "Proposed alternative answer IDs in `eval.json` are excluded until manually approved. "
            "Out-of-scope cases marked `needs_review` are also excluded from refusal rates.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-review-candidates",
        action="store_true",
        help="Include proposed answer IDs and unreviewed out-of-scope cases in the metrics.",
    )
    args = parser.parse_args()

    payload = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    entries = payload["queries"]
    documents = json.loads((ROOT / "assets" / "documents.json").read_text(encoding="utf-8"))
    retriever = FAQRetriever(root=ROOT)
    ranking_results, threshold_results, in_scope_count, ood_count = evaluate(
        retriever,
        entries,
        documents,
        include_review_candidates=args.include_review_candidates,
    )
    pending_ood = sum(
        not entry["in_scope"] and entry.get("needs_review", False)
        for entry in entries
    )
    RESULTS_PATH.write_text(
        render_results(
            ranking_results,
            threshold_results,
            in_scope_count,
            ood_count,
            pending_ood,
        ),
        encoding="utf-8",
    )
    print(f"Saved evaluation results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
