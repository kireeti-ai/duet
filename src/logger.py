"""
src/logger.py

Two responsibilities:
  1. LangSmith tracing  — wraps any pipeline run in a traced context
  2. SQLite storage     — persists every run to results/equipoise.db

Usage as module:
    from src.logger import log_run, get_runs, init_db

    init_db()                          # call once at investigation startup
    run_id = log_run(pipeline_output)  # call after every run_pipeline()

Usage as CLI (inspect stored runs):
    PYTHONPATH=. python src/logger.py
"""

import os
import sys
import json
import sqlite3
import logging
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LangSmith setup
# ---------------------------------------------------------------------------

LANGCHAIN_API_KEY      = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT      = os.getenv("LANGCHAIN_PROJECT", "equipoise-rag")
LANGSMITH_ENABLED      = bool(LANGCHAIN_API_KEY)

if LANGSMITH_ENABLED:
    os.environ["LANGCHAIN_TRACING_V2"]  = "true"
    os.environ["LANGCHAIN_API_KEY"]     = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"]     = LANGCHAIN_PROJECT
else:
    os.environ["LANGCHAIN_TRACING_V2"]  = "false"

# ---------------------------------------------------------------------------
# SQLite setup
# ---------------------------------------------------------------------------

DB_PATH = os.getenv("EQUIPOISE_DB_PATH", "results/equipoise.db")

CREATE_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_uuid        TEXT    NOT NULL UNIQUE,
    timestamp       TEXT    NOT NULL,
    claim           TEXT    NOT NULL,
    method          TEXT    NOT NULL,
    top_k           INTEGER NOT NULL,
    prompt_variant  TEXT    NOT NULL,
    reformulated_query TEXT,
    verdict         TEXT    NOT NULL,
    support_recall      REAL,
    contradiction_recall REAL,
    balance_score       REAL,
    faithfulness        REAL,
    answer_relevancy    REAL,
    context_precision   REAL,
    context_recall      REAL,
    ragas_pass          INTEGER,
    retrieved_json  TEXT,
    ragas_json      TEXT,
    investigation   TEXT
);
"""

CREATE_IDX_METHOD    = "CREATE INDEX IF NOT EXISTS idx_method    ON runs (method);"
CREATE_IDX_PROMPT    = "CREATE INDEX IF NOT EXISTS idx_prompt    ON runs (prompt_variant);"
CREATE_IDX_INV       = "CREATE INDEX IF NOT EXISTS idx_inv       ON runs (investigation);"
CREATE_IDX_TIMESTAMP = "CREATE INDEX IF NOT EXISTS idx_timestamp ON runs (timestamp);"


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def init_db(db_path: str = DB_PATH) -> None:
    """
    Create the SQLite database and runs table if they don't exist.
    Safe to call multiple times — all statements use IF NOT EXISTS.
    """
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    with _connect(db_path) as conn:
        conn.execute(CREATE_RUNS_TABLE)
        conn.execute(CREATE_IDX_METHOD)
        conn.execute(CREATE_IDX_PROMPT)
        conn.execute(CREATE_IDX_INV)
        conn.execute(CREATE_IDX_TIMESTAMP)
    print(f"DB ready at {db_path}")


@contextmanager
def _connect(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Core log function
# ---------------------------------------------------------------------------

def log_run(
    pipeline_output: dict,
    eval_scores: Optional[dict] = None,
    ragas_scores: Optional[dict] = None,
    investigation: Optional[str] = None,
    db_path: str = DB_PATH,
) -> int:
    """
    Persist one pipeline run to SQLite and optionally trace it to LangSmith.

    Args:
        pipeline_output:  dict returned by run_pipeline().
        eval_scores:      dict from evaluator.py (support_recall, contradiction_recall,
                          balance_score). Pass None if not computed for this run.
        ragas_scores:     dict from ragas_eval.py evaluate_ragas(). Pass None if not computed.
        investigation:    label for the run, e.g. "inv01_dense", "inv02_k5", "inv03_neutral".
        db_path:          path to SQLite file.

    Returns:
        The integer row id of the inserted run.
    """

    import uuid
    run_uuid  = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    claim              = pipeline_output.get("claim", "")
    method             = pipeline_output.get("method", "")
    top_k              = len(pipeline_output.get("retrieved", []))
    prompt_variant     = pipeline_output.get("prompt_variant", "")
    reformulated_query = pipeline_output.get("reformulated_query", None)
    verdict            = pipeline_output.get("verdict", "")
    retrieved          = pipeline_output.get("retrieved", [])

    # eval_scores from evaluator.py
    support_recall       = None
    contradiction_recall = None
    balance_score        = None
    if eval_scores:
        support_recall       = eval_scores.get("support_recall")
        contradiction_recall = eval_scores.get("contradiction_recall")
        balance_score        = eval_scores.get("balance_score")

    # ragas_scores from ragas_eval.py
    faithfulness      = None
    answer_relevancy  = None
    context_precision = None
    context_recall    = None
    ragas_pass        = None
    if ragas_scores:
        faithfulness      = ragas_scores.get("faithfulness")
        answer_relevancy  = ragas_scores.get("answer_relevancy")
        context_precision = ragas_scores.get("context_precision")
        context_recall    = ragas_scores.get("context_recall")
        ragas_pass        = int(ragas_scores.get("pass_thresholds", False))

    # Full retrieved list stored as JSON (abstract IDs + rerank scores)
    retrieved_json = json.dumps(
        [
            {
                "id":           r.get("id"),
                "title":        r.get("title", ""),
                "score":        r.get("score"),
                "rerank_score": r.get("rerank_score"),
                "method":       r.get("method", method),
            }
            for r in retrieved
        ]
    )

    ragas_json = json.dumps(ragas_scores) if ragas_scores else None

    row = (
        run_uuid,
        timestamp,
        claim,
        method,
        top_k,
        prompt_variant,
        reformulated_query,
        verdict,
        support_recall,
        contradiction_recall,
        balance_score,
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        ragas_pass,
        retrieved_json,
        ragas_json,
        investigation,
    )

    insert_sql = """
    INSERT INTO runs (
        run_uuid, timestamp, claim, method, top_k, prompt_variant,
        reformulated_query, verdict,
        support_recall, contradiction_recall, balance_score,
        faithfulness, answer_relevancy, context_precision, context_recall,
        ragas_pass, retrieved_json, ragas_json, investigation
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    with _connect(db_path) as conn:
        cursor = conn.execute(insert_sql, row)
        row_id = cursor.lastrowid

    if LANGSMITH_ENABLED:
        _trace_to_langsmith(
            run_uuid=run_uuid,
            claim=claim,
            method=method,
            prompt_variant=prompt_variant,
            verdict=verdict,
            retrieved=retrieved,
            eval_scores=eval_scores,
            ragas_scores=ragas_scores,
            investigation=investigation,
        )

    return row_id


# ---------------------------------------------------------------------------
# LangSmith tracing
# ---------------------------------------------------------------------------

def _trace_to_langsmith(
    run_uuid: str,
    claim: str,
    method: str,
    prompt_variant: str,
    verdict: str,
    retrieved: list,
    eval_scores: Optional[dict],
    ragas_scores: Optional[dict],
    investigation: Optional[str],
) -> None:
    """
    Log a completed run to LangSmith as a chain trace.
    Uses the LangSmith REST client directly — no LangChain callback required.
    Failures are caught and logged as warnings so they never block a run.
    """
    try:
        from langsmith import Client

        client = Client()

        inputs = {
            "claim":         claim,
            "method":        method,
            "prompt_variant": prompt_variant,
            "investigation": investigation,
        }

        outputs = {
            "verdict":       verdict,
            "retrieved_ids": [r.get("id") for r in retrieved],
        }
        if eval_scores:
            outputs["eval_scores"] = eval_scores
        if ragas_scores:
            outputs["ragas_scores"] = ragas_scores

        client.create_run(
            id=run_uuid,
            name="equipoise_pipeline_run",
            run_type="chain",
            inputs=inputs,
            outputs=outputs,
            project_name=LANGCHAIN_PROJECT,
            tags=[method, prompt_variant] + ([investigation] if investigation else []),
        )

    except Exception as e:
        logger.warning(f"LangSmith trace failed (run continues): {e}")


# ---------------------------------------------------------------------------
# Query helpers (used by investigations/ for analysis)
# ---------------------------------------------------------------------------

def get_runs(
    investigation: Optional[str] = None,
    method: Optional[str] = None,
    prompt_variant: Optional[str] = None,
    limit: int = 1000,
    db_path: str = DB_PATH,
) -> list[dict]:
    """
    Fetch runs from SQLite with optional filters.
    Returns a list of dicts (one per run).
    """
    conditions = []
    params     = []

    if investigation:
        conditions.append("investigation = ?")
        params.append(investigation)
    if method:
        conditions.append("method = ?")
        params.append(method)
    if prompt_variant:
        conditions.append("prompt_variant = ?")
        params.append(prompt_variant)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql   = f"SELECT * FROM runs {where} ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    with _connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()

    return [dict(row) for row in rows]


def get_summary(db_path: str = DB_PATH) -> dict:
    """
    Return aggregate counts and averages per investigation + method.
    Useful for quick sanity checks during INV-01/02/03.
    """
    sql = """
    SELECT
        investigation,
        method,
        prompt_variant,
        COUNT(*)                        AS run_count,
        AVG(support_recall)             AS avg_support_recall,
        AVG(contradiction_recall)       AS avg_contradiction_recall,
        AVG(balance_score)              AS avg_balance_score,
        AVG(faithfulness)               AS avg_faithfulness,
        AVG(answer_relevancy)           AS avg_answer_relevancy
    FROM runs
    GROUP BY investigation, method, prompt_variant
    ORDER BY investigation, method
    """
    with _connect(db_path) as conn:
        rows = conn.execute(sql).fetchall()
    return [dict(row) for row in rows]


def export_to_json(
    out_path: str,
    investigation: Optional[str] = None,
    db_path: str = DB_PATH,
) -> None:
    """
    Export runs (optionally filtered by investigation) to a JSON file.
    Called by investigation scripts to produce results/*.json files.
    """
    rows = get_runs(investigation=investigation, db_path=db_path)

    # Expand stored JSON strings back to dicts
    for row in rows:
        if row.get("retrieved_json"):
            row["retrieved"] = json.loads(row["retrieved_json"])
            del row["retrieved_json"]
        if row.get("ragas_json"):
            row["ragas_scores"] = json.loads(row["ragas_json"])
            del row["ragas_json"]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"Exported {len(rows)} runs to {out_path}")


# ---------------------------------------------------------------------------
# CLI — inspect the database
# ---------------------------------------------------------------------------

def _run_cli() -> None:
    init_db()

    print()
    summary = get_summary()

    if not summary:
        print("No runs in database yet.")
        print("Run an investigation script to populate it.")
        return

    col_w = [16, 12, 12, 8, 12, 14, 12]
    headers = [
        "investigation", "method", "prompt", "runs",
        "supp_recall", "contra_recall", "balance",
    ]

    header_row = "  ".join(h.ljust(col_w[i]) for i, h in enumerate(headers))
    print(header_row)
    print("-" * len(header_row))

    for row in summary:
        def fmt(v):
            return f"{v:.4f}" if v is not None else "N/A   "

        line = "  ".join([
            str(row.get("investigation") or "").ljust(col_w[0]),
            str(row.get("method") or "").ljust(col_w[1]),
            str(row.get("prompt_variant") or "").ljust(col_w[2]),
            str(row.get("run_count") or 0).ljust(col_w[3]),
            fmt(row.get("avg_support_recall")).ljust(col_w[4]),
            fmt(row.get("avg_contradiction_recall")).ljust(col_w[5]),
            fmt(row.get("avg_balance_score")).ljust(col_w[6]),
        ])
        print(line)

    print()
    total = sum(r.get("run_count", 0) for r in summary)
    print(f"Total runs in database: {total}")
    print(f"Database: {DB_PATH}")
    print(f"LangSmith: {'enabled  project=' + LANGCHAIN_PROJECT if LANGSMITH_ENABLED else 'disabled (no LANGCHAIN_API_KEY)'}")


if __name__ == "__main__":
    _run_cli()