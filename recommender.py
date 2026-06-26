"""
recommender.py
----------------
Content-based anime recommender.

Approach:
    1. Vectorize each anime's combined text (genres + synopsis) using TF-IDF.
    2. Compute cosine similarity between all anime pairs.
    3. For a given title, return the most similar anime, filtered by a
       minimum score so we don't recommend something similar-but-bad.

This is intentionally simple and explainable -- a good baseline before
trying collaborative filtering or hybrid approaches (see README "Next Steps").
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class AnimeRecommender:
    def __init__(self, data_path: str = "data/anime_clean.csv"):
        self.df = pd.read_csv(data_path)
        self.df["name_lower"] = self.df["name"].str.lower()

        # TF-IDF over the combined genre+synopsis text.
        # max_features keeps the vocabulary (and memory use) reasonable.
        self.vectorizer = TfidfVectorizer(
            stop_words="english", max_features=20000
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.df["combined_text"].fillna("")
        )

        # Precompute the full similarity matrix once at startup.
        # Fine at this dataset size (~5-6k anime); would need approximate
        # nearest-neighbor search (e.g. FAISS) at much larger scale.
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    def search_titles(self, query: str, limit: int = 8) -> list[str]:
        """Return titles containing the query string, sorted by popularity."""
        query = query.lower().strip()
        matches = self.df[self.df["name_lower"].str.contains(query, na=False)]
        matches = matches.sort_values("members", ascending=False)
        return matches["name"].head(limit).tolist()

    def recommend(
        self,
        title: str,
        top_n: int = 10,
        min_score: float = 6.5,
    ) -> pd.DataFrame:
        """
        Return the top_n anime most similar to `title`, excluding anime
        with an average score below min_score (filters out "similar genre,
        but actually bad" results).
        """
        matches = self.df.index[self.df["name_lower"] == title.lower()]
        if len(matches) == 0:
            raise ValueError(f"'{title}' not found in dataset.")

        idx = matches[0]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i, sim in sim_scores[1:]:  # skip the anime itself
            row = self.df.iloc[i]
            if row["score"] >= min_score:
                results.append(
                    {
                        "name": row["name"],
                        "genre": row["genre"],
                        "score": row["score"],
                        "episodes": row["episodes"],
                        "studio": row["studio"],
                        "synopsis": row["synopsis"],
                        "similarity": round(float(sim), 3),
                    }
                )
            if len(results) >= top_n:
                break

        return pd.DataFrame(results)
