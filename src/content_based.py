# Content-Based Filtering
# Abordagem: vetorizar as features textuais de cada jogo com TF-IDF
# e usar similaridade de cosseno pra encontrar os mais próximos.
#
# Por que TF-IDF e não embeddings?
# Embeddings (sentence-transformers etc) dariam resultados semânticos melhores,
# mas pra 27k jogos o TF-IDF já funciona bem e é muito mais rápido de iterar.
# Fica como melhoria futura.

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_steam_data, build_feature_string


def build_tfidf_matrix(df: pd.DataFrame):
    df = df.copy()
    df["features"] = df.apply(build_feature_string, axis=1)

    # max_features=10_000 depois de testar com valores menores (5k ficava ruim
    # pra jogos de nicho) e sem limite (ficava lento sem ganho real de qualidade)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=10_000)
    tfidf_matrix = vectorizer.fit_transform(df["features"])
    return tfidf_matrix, vectorizer


def get_recommendations(game_name: str, df: pd.DataFrame, tfidf_matrix, top_n: int = 10) -> pd.DataFrame:
    name_lower = game_name.lower()
    matches = df[df["name"].str.lower() == name_lower]

    if matches.empty:
        # busca parcial como fallback — útil quando o usuário digita
        # "Counter Strike" em vez de "Counter-Strike"
        matches = df[df["name"].str.lower().str.contains(name_lower, na=False)]

    if matches.empty:
        raise ValueError(f"Jogo '{game_name}' não encontrado no dataset.")

    idx = matches.index[0]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores[idx] = 0  # remove o próprio jogo do resultado

    top_indices = np.argsort(sim_scores)[::-1][:top_n]
    result = df.iloc[top_indices][["name", "genres", "steamspy_tags"]].copy()
    result["similarity_score"] = sim_scores[top_indices].round(4)
    return result.reset_index(drop=True)


if __name__ == "__main__":
    import os

    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    df = load_steam_data(DATA_DIR)
    print(f"Jogos carregados: {len(df)}")

    matrix, _ = build_tfidf_matrix(df)

    # testando com Counter-Strike porque é um caso fácil de validar manualmente —
    # qualquer FPS multiplayer competitivo deveria aparecer na lista
    query = "Counter-Strike"
    recs = get_recommendations(query, df, matrix, top_n=5)
    print(f"\nRecomendações para '{query}':\n")
    print(recs.to_string(index=False))
