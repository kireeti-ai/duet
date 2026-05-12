"""
evaluator.py
Computes the three original Equipoise metrics against SciFact ground truth.

SciFact design: each claim is labeled SUPPORT or CONTRADICT -- never both.
- Support Recall: avg recall on claims where evidence label is SUPPORT
- Contradiction Recall: avg recall on claims where evidence label is CONTRADICT
- Balance Score: Contradiction Recall / Support Recall
                 1.0 = retrieval finds both equally well
                 below 0.5 = system is biased toward supporting evidence
"""

import json


def load_scifact_labels() -> dict:
    """
    Loads SciFact labels from local jsonl files.
    Returns dict mapping claim_id -> {
        claim, supporting_ids, contradicting_ids, claim_type
    }
    claim_type is SUPPORT if any evidence is SUPPORT, else CONTRADICT.
    """
    print("Loading SciFact labels...")

    labels = {}

    for split in ["claims_train.jsonl", "claims_dev.jsonl"]:
        path = f"data/scifact/data/{split}"
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                claim_id = str(item["id"])
                supporting = set()
                contradicting = set()

                for doc_id, evidence_list in item["evidence"].items():
                    for ev in evidence_list:
                        if ev["label"] == "SUPPORT":
                            supporting.add(str(doc_id))
                        elif ev["label"] == "CONTRADICT":
                            contradicting.add(str(doc_id))

                if supporting:
                    claim_type = "SUPPORT"
                elif contradicting:
                    claim_type = "CONTRADICT"
                else:
                    claim_type = "NONE"

                labels[claim_id] = {
                    "claim": item["claim"],
                    "supporting_ids": supporting,
                    "contradicting_ids": contradicting,
                    "claim_type": claim_type
                }

    print(f"Loaded labels for {len(labels)} claims")
    return labels


def compute_recall(retrieved_ids: list, gold_ids: set) -> float:
    """Fraction of gold abstracts that appear in retrieved list."""
    if not gold_ids:
        return 0.0
    hits = set(retrieved_ids) & gold_ids
    return len(hits) / len(gold_ids)


def evaluate_single(claim_id: str, retrieved: list, labels: dict) -> dict:
    """Evaluates retrieval for a single claim."""
    if claim_id not in labels:
        return {"error": f"claim_id {claim_id} not found in labels"}

    claim_labels = labels[claim_id]
    retrieved_ids = [r["id"] for r in retrieved]
    claim_type = claim_labels["claim_type"]

    if claim_type == "SUPPORT":
        recall = compute_recall(retrieved_ids, claim_labels["supporting_ids"])
    elif claim_type == "CONTRADICT":
        recall = compute_recall(retrieved_ids, claim_labels["contradicting_ids"])
    else:
        recall = 0.0

    return {
        "claim_id": claim_id,
        "claim": claim_labels["claim"],
        "claim_type": claim_type,
        "recall": round(recall, 4),
        "retrieved_ids": retrieved_ids,
        "gold_ids": list(claim_labels["supporting_ids"] | claim_labels["contradicting_ids"])
    }


def evaluate_batch(claims: list, labels: dict, method: str = "dense") -> dict:
    """
    Evaluates all claims and computes Support Recall, Contradiction Recall, Balance Score.
    """
    from src.retriever import retrieve
    from src.reformulator import reformulate_query

    support_recalls = []
    contradict_recalls = []
    skipped = 0

    total = len(claims)
    for i, claim in enumerate(claims):
        claim_id = str(claim["id"])
        claim_text = claim.get("claim")
        if claim_text is None:
            claim_text = claim.get("text")

        if not isinstance(claim_text, str) or not claim_text.strip():
            print(f"  Error on claim {claim_id}: missing claim text field ('claim' or 'text')")
            skipped += 1
            continue

        if claim_id not in labels:
            skipped += 1
            continue

        if labels[claim_id]["claim_type"] == "NONE":
            skipped += 1
            continue

        try:
            if method == "queryreform":
                reformulated_query = reformulate_query(claim_text)
                retrieved = retrieve(
                    claim_text,
                    method=method,
                    reformulated_query=reformulated_query
                )
            else:
                retrieved = retrieve(claim_text, method=method)
            result = evaluate_single(claim_id, retrieved, labels)

            if labels[claim_id]["claim_type"] == "SUPPORT":
                support_recalls.append(result["recall"])
            elif labels[claim_id]["claim_type"] == "CONTRADICT":
                contradict_recalls.append(result["recall"])

            if (i + 1) % 20 == 0:
                print(f"  Evaluated {i+1}/{total} claims")

        except Exception as e:
            print(f"  Error on claim {claim_id}: {e}")
            skipped += 1

    avg_support = sum(support_recalls) / len(support_recalls) if support_recalls else 0.0
    avg_contradict = sum(contradict_recalls) / len(contradict_recalls) if contradict_recalls else 0.0
    balance = avg_contradict / avg_support if avg_support > 0 else 0.0

    return {
        "method": method,
        "support_claims_evaluated": len(support_recalls),
        "contradict_claims_evaluated": len(contradict_recalls),
        "skipped": skipped,
        "avg_support_recall": round(avg_support, 4),
        "avg_contradiction_recall": round(avg_contradict, 4),
        "avg_balance_score": round(balance, 4)
    }


if __name__ == "__main__":
    from src.retriever import retrieve

    labels = load_scifact_labels()

    support_claims = [(k, v) for k, v in labels.items() if v["claim_type"] == "SUPPORT"]
    contradict_claims = [(k, v) for k, v in labels.items() if v["claim_type"] == "CONTRADICT"]

    print(f"SUPPORT claims:    {len(support_claims)}")
    print(f"CONTRADICT claims: {len(contradict_claims)}")

    # test 3 support claims and 3 contradict claims
    test_cases = support_claims[:3] + contradict_claims[:3]

    print(f"\nTesting dense retrieval on 6 claims\n")

    for claim_id, data in test_cases:
        retrieved = retrieve(data["claim"], method="dense")
        result = evaluate_single(claim_id, retrieved, labels)
        print(f"[{result['claim_type']}] {data['claim'][:60]}")
        print(f"  Recall: {result['recall']} | Gold IDs: {result['gold_ids']}")
        print()
