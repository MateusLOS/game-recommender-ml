# Collaborative Filtering — filtragem baseada em comportamento de usuários
#
# O problema central desse dataset: o Steam não disponibiliza histórico
# individual de usuários publicamente. O steam.csv tem ratings AGREGADOS
# por jogo (positive_ratings, negative_ratings), não por pessoa.
#
# A solução foi simular interações sintéticas onde cada usuário tem preferências
# de gênero definidas e avalia jogos com ruído gaussiano sobre o rating real.
# Não é o ideal — mas modela o comportamento de forma razoável e a arquitetura
# do modelo em si seria idêntica com dados reais.
#
# Com a Steam Web API (requer chave + perfis públicos) daria pra puxar
# playtime real por usuário. Fica como próximo passo.

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_steam_data, rmse, precision_at_k, recall_at_k


def build_user_item_matrix(interactions: pd.DataFrame) -> pd.DataFrame:
    return interactions.pivot_table(
        index="user_id", columns="game_name", values="rating", fill_value=0
    )


def get_similar_users(user_id: str, user_item_matrix: pd.DataFrame, top_n: int = 10) -> pd.Series:
    if user_id not in user_item_matrix.index:
        raise ValueError(f"Usuário '{user_id}' não encontrado.")

    sim_matrix = cosine_similarity(user_item_matrix)
    sim_df = pd.DataFrame(sim_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)
    return sim_df[user_id].drop(user_id).sort_values(ascending=False).head(top_n)


def get_recommendations(user_id: str, user_item_matrix: pd.DataFrame, top_n: int = 10, n_similar_users: int = 20) -> pd.DataFrame:
    similar_users = get_similar_users(user_id, user_item_matrix, top_n=n_similar_users)
    played = set(user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index)

    scores = {}
    weights_sum = {}

    for sim_user, sim_score in similar_users.items():
        for game, rating in user_item_matrix.loc[sim_user].items():
            if game in played or rating == 0:
                continue
            scores[game] = scores.get(game, 0) + sim_score * rating
            weights_sum[game] = weights_sum.get(game, 0) + sim_score

    if not scores:
        return pd.DataFrame(columns=["game_name", "predicted_rating"])

    result = {g: scores[g] / weights_sum[g] for g in scores if weights_sum[g] > 0}

    return (
        pd.DataFrame.from_dict(result, orient="index", columns=["predicted_rating"])
        .sort_values("predicted_rating", ascending=False)
        .head(top_n)
        .reset_index()
        .rename(columns={"index": "game_name"})
        .assign(predicted_rating=lambda x: x["predicted_rating"].round(4))
    )


def evaluate(user_item_matrix: pd.DataFrame, test_interactions: pd.DataFrame, k: int = 10) -> dict:
    precisions, recalls = [], []

    for user_id in test_interactions["user_id"].unique():
        if user_id not in user_item_matrix.index:
            continue
        true_games = test_interactions[test_interactions["user_id"] == user_id]["game_name"].tolist()
        try:
            recs = get_recommendations(user_id, user_item_matrix, top_n=k)
        except ValueError:
            continue
        recommended = recs["game_name"].tolist()
        precisions.append(precision_at_k(recommended, true_games, k))
        recalls.append(recall_at_k(recommended, true_games, k))

    return {
        f"precision@{k}": round(float(np.mean(precisions)), 4) if precisions else 0.0,
        f"recall@{k}": round(float(np.mean(recalls)), 4) if recalls else 0.0,
    }


def build_interactions_from_steam(df: pd.DataFrame, n_users: int = 300) -> pd.DataFrame:
    # Cada usuário recebe 1-3 gêneros favoritos e avalia jogos majoritariamente
    # desses gêneros + alguns aleatórios (simula descoberta orgânica).
    # O ruído gaussiano sobre o rating real serve pra diferenciar gostos
    # dentro do mesmo gênero — sem ele, todos os usuários de "Action" seriam idênticos.
    rng = np.random.default_rng(42)

    genre_map: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        for genre in str(row.get("genres", "")).split():
            genre_map.setdefault(genre, []).append(row["name"])

    all_genres = [g for g, games in genre_map.items() if len(games) >= 10]
    game_names = df["name"].tolist()
    rating_lookup = df.set_index("name")["rating"].fillna(3.0).to_dict()

    rows = []
    for i in range(n_users):
        fav_genres = rng.choice(all_genres, size=rng.integers(1, 4), replace=False).tolist()
        fav_pool = list({g for genre in fav_genres for g in genre_map.get(genre, [])})

        n_fav = int(rng.integers(8, 25))
        n_rand = int(rng.integers(2, 8))

        chosen_fav = rng.choice(fav_pool, size=min(n_fav, len(fav_pool)), replace=False).tolist()
        chosen_rand = rng.choice(game_names, size=n_rand, replace=False).tolist()

        for g in set(chosen_fav + chosen_rand):
            base = rating_lookup.get(g, 3.0)
            noisy = float(np.clip(base + rng.normal(0, 0.4), 1, 5))
            rows.append({"user_id": f"User_{i}", "game_name": g, "rating": round(noisy, 1)})

    return pd.DataFrame(rows)


def user_train_test_split(interactions: pd.DataFrame, test_ratio: float = 0.25) -> tuple:
    # Split por usuário, não por linha — garantia de que os jogos de teste
    # não estão na matriz de treino (evita data leakage)
    train_parts, test_parts = [], []
    for _, group in interactions.groupby("user_id"):
        n_test = max(2, int(len(group) * test_ratio))
        if len(group) <= n_test:
            train_parts.append(group)
            continue
        test_rows = group.sample(n=n_test, random_state=42)
        train_parts.append(group.drop(test_rows.index))
        test_parts.append(test_rows)
    return pd.concat(train_parts).reset_index(drop=True), pd.concat(test_parts).reset_index(drop=True)


if __name__ == "__main__":
    import os

    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    df = load_steam_data(DATA_DIR)
    print(f"Jogos carregados: {len(df)}")

    interactions = build_interactions_from_steam(df, n_users=300)
    print(f"Interações geradas: {len(interactions)} para {interactions['user_id'].nunique()} usuários")

    train_df, test_df = user_train_test_split(interactions, test_ratio=0.25)
    train_matrix = build_user_item_matrix(train_df)
    print(f"Matriz de treino: {train_matrix.shape[0]} usuários x {train_matrix.shape[1]} jogos")

    recs = get_recommendations("User_0", train_matrix, top_n=5)
    print(f"\nTop 5 recomendações para User_0:\n")
    print(recs.to_string(index=False))

    metrics = evaluate(train_matrix, test_df, k=10)
    print("\nMétricas de avaliação:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
