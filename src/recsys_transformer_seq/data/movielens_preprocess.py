from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from recsys_transformer_seq.config import PROJECT_ROOT, load_config

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "movielens-20m"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SEQS_PATH = PROCESSED_DIR / "movielens_sequences.csv"


def build_sequences() -> None:
    cfg = load_config()
    max_len = cfg["data"]["max_sequence_length"]
    min_interactions = cfg["data"]["min_user_interactions"]

    # Try common MovieLens filenames in order of preference
    ratings_candidates = [
        RAW_DIR / "ratings.csv",
        RAW_DIR / "rating.csv",  # fallback name
    ]
    ratings_path = None
    for candidate in ratings_candidates:
        if candidate.exists():
            ratings_path = candidate
            break

    if ratings_path is None:
        raise FileNotFoundError(
            f"Missing ratings file in {RAW_DIR}. "
            f"Tried: {[str(c) for c in ratings_candidates]}"
        )

    print(f"[INFO] Loading ratings from {ratings_path}")
    df = pd.read_csv(ratings_path)

    for col in ("userId", "movieId", "timestamp"):
        if col not in df.columns:
            raise ValueError(f"Expected column {col} not in ratings file")

    df = df.sort_values(["userId", "timestamp"])

    counts = df["userId"].value_counts()
    keep_users = counts[counts >= min_interactions].index
    df = df[df["userId"].isin(keep_users)]

    print(f"[INFO] Users after filter: {df['userId'].nunique():,}")

    rows = []
    for user_id, user_df in tqdm(df.groupby("userId"), desc="Building sequences"):
        items = user_df["movieId"].tolist()
        if len(items) < min_interactions:
            continue

        for i in range(1, len(items)):
            seq = items[max(0, i - max_len) : i]
            target = items[i]
            rows.append(
                {
                    "userId": user_id,
                    "sequence": " ".join(map(str, seq)),
                    "target": target,
                }
            )

    out_df = pd.DataFrame(rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(SEQS_PATH, index=False)
    print(f"[INFO] Wrote {len(out_df):,} rows to {SEQS_PATH}")


if __name__ == "__main__":
    build_sequences()