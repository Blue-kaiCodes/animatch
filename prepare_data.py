"""
prepare_data.py
----------------
Cleans the raw anime dataset (one row per anime-genre pair) into a single
row per anime, with genres collapsed into a list. Saves the result to
data/anime_clean.csv for use by the recommender and the Streamlit app.

Run this once before using app.py:
    python prepare_data.py
"""

import pandas as pd
import os

RAW_PATH = "tidy_anime.csv"
OUT_DIR = "data"
OUT_PATH = os.path.join(OUT_DIR, "anime_clean.csv")


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only the columns we actually need for recommendations
    keep_cols = [
        "animeID", "name", "type", "genre", "studio",
        "episodes", "score", "scored_by", "popularity",
        "members", "synopsis",
    ]
    df = df[keep_cols].copy()

    # Drop rows with no genre or no synopsis -- useless for content-based filtering
    df = df.dropna(subset=["genre", "synopsis"])

    # Collapse the one-row-per-genre structure into one row per anime,
    # with all of that anime's genres joined into a single string.
    grouped = (
        df.groupby("animeID")
        .agg({
            "name": "first",
            "type": "first",
            "genre": lambda g: ", ".join(sorted(set(g))),
            "studio": "first",
            "episodes": "first",
            "score": "first",
            "scored_by": "first",
            "popularity": "first",
            "members": "first",
            "synopsis": "first",
        })
        .reset_index()
    )

    # Drop duplicate titles (some anime have multiple MAL entries, e.g. re-releases)
    grouped = grouped.drop_duplicates(subset=["name"]).reset_index(drop=True)

    # Filter out anime with very few scorers -- these skew similarity results
    grouped = grouped[grouped["scored_by"].fillna(0) >= 1000].reset_index(drop=True)

    # Build the single text field the recommender will vectorize:
    # genres are repeated 3x so they weigh more heavily than synopsis words
    grouped["combined_text"] = (
        (grouped["genre"] + " ") * 3 + grouped["synopsis"].fillna("")
    )

    return grouped


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    raw = load_raw()
    cleaned = clean(raw)
    cleaned.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(cleaned)} unique anime to {OUT_PATH}")


if __name__ == "__main__":
    main()
