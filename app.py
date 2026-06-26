"""
app.py
-------
Streamlit front-end for AniMatch, a content-based anime recommender.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
from recommender import AnimeRecommender

st.set_page_config(page_title="AniMatch", page_icon="🍥", layout="wide")


@st.cache_resource
def load_recommender():
    return AnimeRecommender()


recommender = load_recommender()

st.title("🍥 AniMatch")
st.caption(
    "A content-based anime recommender. Pick a show you like, "
    "and AniMatch finds others with similar genres and themes."
)

with st.sidebar:
    st.header("Settings")
    top_n = st.slider("Number of recommendations", min_value=3, max_value=20, value=8)
    min_score = st.slider(
        "Minimum MAL score for recommendations",
        min_value=0.0, max_value=9.0, value=6.5, step=0.5,
        help="Filters out anime that are genre-similar but poorly rated.",
    )
    st.markdown("---")
    st.markdown(
        "**How it works**\n\n"
        "AniMatch converts each anime's genres + synopsis into a TF-IDF "
        "vector, then uses cosine similarity to find the closest matches "
        "in that space. Genres are weighted more heavily than synopsis text."
    )

query = st.text_input("Search for an anime you like", placeholder="e.g. Bleach")

selected_title = None
if query:
    matches = recommender.search_titles(query)
    if matches:
        selected_title = st.selectbox("Did you mean:", matches)
    else:
        st.warning("No matches found. Try a different spelling or title.")

if selected_title and st.button("Get Recommendations", type="primary"):
    try:
        results = recommender.recommend(selected_title, top_n=top_n, min_score=min_score)
    except ValueError as e:
        st.error(str(e))
        results = pd.DataFrame()

    if results.empty:
        st.info("No recommendations met the minimum score filter. Try lowering it.")
    else:
        st.subheader(f"Because you liked {selected_title}:")
        for _, row in results.iterrows():
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"### {row['name']}")
                    st.markdown(f"**Genres:** {row['genre']}")
                    st.markdown(f"**Studio:** {row['studio']}  |  **Episodes:** {row['episodes']}")
                    synopsis = str(row["synopsis"])
                    st.write(synopsis[:280] + ("..." if len(synopsis) > 280 else ""))
                with col2:
                    st.metric("MAL Score", row["score"])
                    st.metric("Similarity", f"{row['similarity']*100:.1f}%")

st.markdown("---")
st.caption("Data: MyAnimeList, via the tidytuesday open dataset.")
