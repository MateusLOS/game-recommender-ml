import os
import pandas as pd
import numpy as np


def load_steam_data(data_dir: str) -> pd.DataFrame:
    # O dataset do Kaggle vem em arquivos separados. O principal é steam.csv,
    # mas as descrições ficam em steam_description_data.csv — vale fazer o merge
    # porque o short_description melhora bastante a qualidade do TF-IDF.
    main_path = os.path.join(data_dir, "steam.csv")
    desc_path = os.path.join(data_dir, "steam_description_data.csv")

    df = pd.read_csv(main_path)
    df = df.dropna(subset=["name"])
    df = df.drop_duplicates(subset=["name"])

    if os.path.exists(desc_path):
        desc_df = pd.read_csv(desc_path, usecols=["steam_appid", "short_description"])
        desc_df = desc_df.rename(columns={"steam_appid": "appid"})
        df = df.merge(desc_df, on="appid", how="left")
    else:
        df["short_description"] = ""

    # O Steam usa ";" como separador dentro dos campos de gênero e tags
    # (ex: "Action;RPG;Indie"). Precisa virar espaço pra o TF-IDF tratar
    # cada valor como token separado.
    for col in ("genres", "steamspy_tags", "categories"):
        if col in df.columns:
            df[col] = df[col].fillna("").str.replace(";", " ", regex=False)

    # O dataset não tem rating direto — tem positive_ratings e negative_ratings.
    # Converti pra escala 1-5 usando aprovação relativa. Jogos sem nenhum voto
    # ficam com 3.0 como neutro.
    if "positive_ratings" in df.columns and "negative_ratings" in df.columns:
        total = df["positive_ratings"] + df["negative_ratings"]
        df["rating"] = (df["positive_ratings"] / total.replace(0, 1) * 4 + 1).round(2)

    df = df.reset_index(drop=True)
    return df


def build_feature_string(row: pd.Series) -> str:
    # Tentei incluir 'categories' aqui também (tem coisas como "Single-player",
    # "Multi-player") mas estava adicionando ruído — jogos completamente diferentes
    # acabavam similares só por serem single-player. Melhor usar só gênero + tags + descrição.
    parts = []
    for col in ("genres", "steamspy_tags", "short_description"):
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            parts.append(str(val))
    return " ".join(parts)


def precision_at_k(recommended: list, relevant: list, k: int) -> float:
    if k == 0:
        return 0.0
    hits = len(set(recommended[:k]) & set(relevant))
    return hits / k


def recall_at_k(recommended: list, relevant: list, k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(recommended[:k]) & set(relevant))
    return hits / len(relevant)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))
