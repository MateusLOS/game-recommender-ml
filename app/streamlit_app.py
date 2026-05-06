import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import streamlit as st
from content_based import build_tfidf_matrix, get_recommendations
from utils import load_steam_data

st.set_page_config(page_title="Game Recommender", page_icon="🎮", layout="centered")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


@st.cache_data
def load_data():
    df = load_steam_data(DATA_DIR)
    matrix, _ = build_tfidf_matrix(df)
    return df, matrix


df, tfidf_matrix = load_data()

st.title("🎮 Game Recommendation System")
st.caption(f"Base: {len(df):,} jogos do Steam · Modelo: TF-IDF + Cosine Similarity")

st.markdown("---")

game_names = sorted(df["name"].dropna().unique().tolist())
selected_game = st.selectbox("Escolha um jogo:", game_names)
top_n = st.slider("Quantas recomendações?", min_value=3, max_value=20, value=5)

if st.button("Recomendar"):
    try:
        recs = get_recommendations(selected_game, df, tfidf_matrix, top_n=top_n)
        st.subheader(f"Jogos similares a **{selected_game}**")

        for _, row in recs.iterrows():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{row['name']}**")
                genres_val = row.get("genres", "")
                if pd.notna(genres_val) and str(genres_val).strip():
                    st.caption(genres_val)
            with col2:
                st.metric("Similaridade", f"{row['similarity_score']:.1%}")
            st.divider()

    except ValueError as e:
        st.error(str(e))

st.markdown("---")
st.caption("Dados: Kaggle Steam Dataset · Código: github.com/seu-usuario/game-recommender")
