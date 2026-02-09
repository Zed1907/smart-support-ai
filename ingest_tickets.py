import pandas as pd

from backend.embedder import embed_text
from backend.endee_client import create_index, insert_batch

CSV_PATH = "data/cleaned_tickets.csv"
BATCH_SIZE = 64  # batching improves performance

def main():
    # Load dataset
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} tickets")

    # Create index (safe to call multiple times)
    print("Creating index (if not exists)...")
    print(create_index())

    batch = []
    inserted = 0

    for _, row in df.iterrows():
        vector = embed_text(row["description"])

        batch.append({
            "id": str(row["ticket_id"]),
            "values": vector,
            "metadata": {
                "team": row["team"],
                "resolution": row["resolution"]
            }
        })

        if len(batch) >= BATCH_SIZE:
            insert_batch(batch)
            inserted += len(batch)
            print(f"Ingested {inserted} tickets")
            batch.clear()

    # Insert remaining vectors
    if batch:
        insert_batch(batch)
        inserted += len(batch)
        print(f"Ingested {inserted} tickets")

    print("✅ Ingestion complete")

if __name__ == "__main__":
    main()
