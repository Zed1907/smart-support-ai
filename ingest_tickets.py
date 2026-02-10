"""
SmartSupport AI — Ticket Ingestion Script
Reads cleaned_tickets.csv → embeds descriptions → upserts into Endee.

SDK limits (confirmed from source):
  MAX_VECTORS_PER_BATCH = 1000
  MAX_DIMENSION_ALLOWED = 10000
  Index name: alphanumeric + underscores, max 48 chars
"""

import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from backend.embedder import embed_batch
from backend.endee_client import (
    INDEX_NAME,
    DIMENSION,
    check_connection,
    create_index,
    insert_batch,
    invalidate_cache,
    list_indexes,
)

# ── Configuration ──────────────────────────────────────────────────────────────
CSV_PATH    = "data/cleaned_tickets.csv"
BATCH_SIZE  = 500   # Well under SDK max of 1000; sweet spot for speed + reliability

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    """Load and validate the CSV file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"CSV not found: {path}\n"
            "  → Copy cleaned_tickets.csv into the data/ directory."
        )

    df = pd.read_csv(path)

    required = ["ticket_id", "description", "team", "resolution"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}\n  Found: {list(df.columns)}")

    before = len(df)
    df = df.dropna(subset=["description", "team", "resolution"])
    dropped = before - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped} rows with null description/team/resolution")

    logger.info(f"Loaded {len(df):,} valid tickets from {path}")
    return df


def ingest(df: pd.DataFrame) -> tuple[int, int]:
    """Embed descriptions and upsert all vectors into Endee."""
    total    = len(df)
    inserted = 0
    failed   = 0

    with tqdm(total=total, unit="ticket", desc="Ingesting") as bar:
        for start in range(0, total, BATCH_SIZE):
            chunk = df.iloc[start : start + BATCH_SIZE]

            try:
                # ① Embed all descriptions in the chunk at once (GPU/CPU batched)
                embeddings = embed_batch(
                    chunk["description"].tolist(),
                    normalize=True        # Cosine similarity needs unit vectors
                )

                # ② Build SDK-compatible vector list
                vectors = [
                    {
                        "id":       str(row["ticket_id"]),
                        "values":   emb,
                        "metadata": {
                            "team":       str(row["team"]),
                            "resolution": str(row["resolution"]),
                        },
                    }
                    for (_, row), emb in zip(chunk.iterrows(), embeddings)
                ]

                # ③ Upsert via SDK  →  msgpack POST /index/tickets/vector/insert
                insert_batch(vectors)
                inserted += len(vectors)

            except Exception as exc:
                logger.error(f"Batch {start}–{start+len(chunk)} failed: {exc}")
                failed += len(chunk)

            bar.update(len(chunk))

    return inserted, failed


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print()
    print("╔══════════════════════════════════════════════╗")
    print("║   SmartSupport AI  —  Data Ingestion         ║")
    print("╚══════════════════════════════════════════════╝")

    # Step 1 — Endee health check
    print("\n[1/4]  Checking Endee server…")
    if not check_connection():
        print("  ❌  Endee is not running. Start it first:")
        print("       cd ~/endee")
        print("       export NDD_DATA_DIR=$(pwd)/data")
        print("       ./build/ndd-avx2")
        sys.exit(1)
    print("  ✅  Endee running at localhost:8080")

    # Step 2 — Load CSV
    print("\n[2/4]  Loading CSV…")
    try:
        df = load_csv(CSV_PATH)
        print(f"  ✅  {len(df):,} tickets ready")
    except Exception as exc:
        print(f"  ❌  {exc}")
        sys.exit(1)

    # Step 3 — Create index (idempotent)
    print("\n[3/4]  Creating index…")
    try:
        msg = create_index(
            index_name=INDEX_NAME,
            dimension=DIMENSION,
            space_type="cosine",
        )
        invalidate_cache()   # Clear LRU so get_index fetches fresh metadata
        print(f"  ✅  {msg}")
    except Exception as exc:
        print(f"  ❌  Failed to create index: {exc}")
        sys.exit(1)

    # Step 4 — Ingest
    print(f"\n[4/4]  Ingesting {len(df):,} tickets (batch size = {BATCH_SIZE})…\n")
    try:
        inserted, failed = ingest(df)
    except KeyboardInterrupt:
        print("\n  ⚠️   Interrupted by user")
        sys.exit(1)
    except Exception as exc:
        logger.exception("Ingestion crashed")
        sys.exit(1)

    # Summary
    print()
    print("╔══════════════════════════════════════════════╗")
    print(f"║  ✅  Inserted : {inserted:>6,} tickets               ║")
    if failed:
        print(f"║  ⚠️   Failed  : {failed:>6,} tickets               ║")
    print("╚══════════════════════════════════════════════╝")

    if inserted == 0:
        print("\n❌  No vectors were inserted. Check the errors above.")
        sys.exit(1)

    print("\n🎉  Ingestion complete! Next step:")
    print("     uvicorn backend.main:app --reload\n")


if __name__ == "__main__":
    main()