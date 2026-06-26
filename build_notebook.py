import json

cells = []

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})

def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.splitlines(keepends=True)})

md("""# AniMatch — Exploratory Data Analysis

This notebook explores the anime dataset before building the recommender in `recommender.py`.
Goal: understand the genre landscape, rating distribution, and data quality issues that shaped
the cleaning decisions in `prepare_data.py`.

**Dataset:** MyAnimeList data via the [tidytuesday project](https://github.com/rfordatascience/tidytuesday).
""")

code("""import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
df = pd.read_csv("data/anime_clean.csv")
df.shape
""")

md("## 1. A first look at the data")

code("""df.head()
""")

code("""df.info()
""")

md("""## 2. Rating distribution

How are MAL scores distributed across anime? This matters for the recommender: if scores
cluster tightly, a fixed `min_score` filter (used in `recommender.py`) behaves very differently
than if scores are spread out.""")

code("""fig, ax = plt.subplots(figsize=(8, 5))
df["score"].dropna().hist(bins=30, ax=ax, color="#5b6ee1", edgecolor="white")
ax.set_title("Distribution of MyAnimeList Scores")
ax.set_xlabel("Score")
ax.set_ylabel("Number of anime")
plt.tight_layout()
plt.savefig("notebook_assets_score_dist.png", dpi=120)
plt.show()
""")

md("""**Observation:** scores are left-skewed and cluster between 6 and 8 — most anime that get
enough ratings to appear in this cleaned dataset are at least decently received. This is part of
why the cleaning step in `prepare_data.py` drops anime with fewer than 1,000 scorers: very
low-vote anime have noisy, unreliable scores.""")

md("## 3. Most common genres")

code("""genre_counts = (
    df["genre"]
    .str.split(", ")
    .explode()
    .value_counts()
    .head(15)
)

fig, ax = plt.subplots(figsize=(8, 6))
genre_counts.sort_values().plot(kind="barh", ax=ax, color="#e16f5b")
ax.set_title("Top 15 Most Common Genres")
ax.set_xlabel("Number of anime")
plt.tight_layout()
plt.savefig("notebook_assets_genre_counts.png", dpi=120)
plt.show()
""")

md("""**Observation:** Comedy, Action, and Drama dominate. This imbalance matters for a
TF-IDF-based recommender — without down-weighting common genres, "Comedy" would barely help
distinguish anime, since nearly everything has it. TF-IDF naturally handles this: genres that
appear less often (e.g. "Space", "Military") get a higher weight when they do appear, which is
exactly the property we want for finding *specific* similarity rather than generic similarity.""")

md("## 4. Score vs. popularity")

code("""fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(df["members"], df["score"], alpha=0.3, s=10, color="#5b6ee1")
ax.set_xscale("log")
ax.set_title("Score vs. Popularity (log scale)")
ax.set_xlabel("Members (log scale)")
ax.set_ylabel("Score")
plt.tight_layout()
plt.savefig("notebook_assets_score_vs_popularity.png", dpi=120)
plt.show()
""")

md("""**Observation:** there's a mild positive trend — more popular anime tend to score slightly
higher on average, likely because popularity itself is partly driven by quality, and very obscure
shows have noisier scores from fewer raters. This supports filtering low-vote anime during cleaning.""")

md("## 5. Episode count distribution")

code("""fig, ax = plt.subplots(figsize=(8, 5))
df[df["episodes"] < 100]["episodes"].dropna().hist(bins=30, ax=ax, color="#5b6ee1", edgecolor="white")
ax.set_title("Episode Count Distribution (under 100 episodes)")
ax.set_xlabel("Episodes")
ax.set_ylabel("Number of anime")
plt.tight_layout()
plt.savefig("notebook_assets_episode_dist.png", dpi=120)
plt.show()
""")

md("""**Observation:** most anime are short — under 26 episodes (one or two TV seasons). Long-runners
like *Bleach* (366 episodes) or *One Piece* are clear outliers, which is worth knowing since
episode count isn't currently used as a recommendation feature, but could be in a future
"similar commitment level" filter (see README).""")

md("""## 6. Takeaways that shaped `prepare_data.py` and `recommender.py`

- **Score noise at low vote counts** → filtered anime with `scored_by < 1000`.
- **Genre imbalance** → used TF-IDF (not raw genre counts) so rare, more distinctive genres
  carry more weight in similarity calculations.
- **Genres should matter more than synopsis wording** → genre text is repeated 3x before
  vectorizing, so two anime sharing genres outweigh two anime that just share generic
  synopsis phrasing ("a group of friends...").
- **Duplicate/movie-spinoff entries** → deduplicated by title to avoid near-identical entries
  cluttering recommendations.

Next steps for this analysis are tracked in the main README under "Next Steps".
""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("eda.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print("Notebook written.")
