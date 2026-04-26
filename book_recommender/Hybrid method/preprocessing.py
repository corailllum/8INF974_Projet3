"""
preprocessing.py — Prétraitement partagé entre le module BERT et le module GNN.

Produit un DataFrame unifié avec :
  - user_id, title, rating, description, authors, categories
  - user_id_idx, item_id_idx  (indices continus pour PyG)
Et exporte les LabelEncoders + catalogue livres ordonné.
"""

import os
import numpy as np
import pandas as pd
import sklearn.preprocessing as pp
from sklearn.model_selection import train_test_split

from config import (
    RATINGS_FILE, BOOKS_FILE, CACHE_FILE, BOOK_TITLES_FILE,
    MIN_USER_INTERACTIONS, MIN_BOOK_RATINGS,
    MIN_DESCRIPTION_LEN, POSITIVE_RATING_THRESHOLD,
    TEST_SIZE, RANDOM_STATE, SAMPLE_SIZE, CACHE_DIR
)


# ─────────────────────────────────────────────────────────────────────────────
# Chargement & fusion
# ─────────────────────────────────────────────────────────────────────────────

def load_raw() -> pd.DataFrame:
    print("Chargement des fichiers CSV...")
    ratings = pd.read_csv(RATINGS_FILE, on_bad_lines="skip", low_memory=False)
    books   = pd.read_csv(BOOKS_FILE,   on_bad_lines="skip", low_memory=False)

    ratings = ratings.rename(columns={
        "User_id":      "user_id",
        "Title":        "title",
        "review/score": "rating",
    })
    books = books.rename(columns={
        "Title":        "title",
        "description":  "description",
        "authors":      "authors",
        "categories":   "categories",
    })

    print(f"  Ratings bruts : {len(ratings):,} | Livres : {len(books):,}")

    # Jointure sur le titre
    df = ratings.merge(
        books[["title", "description", "authors", "categories"]],
        on="title", how="inner"
    )

    # Nettoyage
    df.dropna(subset=["user_id", "title", "rating", "description"], inplace=True)
    df.dropna(subset=["rating"], inplace=True)
    df = df[df["description"].str.len() > MIN_DESCRIPTION_LEN]

    # Filtre ASCII
    ascii_ratio = (
        df["description"].str.encode("ascii", errors="ignore").str.len()
        / df["description"].str.len()
    )
    df = df[ascii_ratio > 0.8]

    # Filtre min interactions
    user_counts = df["user_id"].value_counts()
    df = df[df["user_id"].isin(
        user_counts[user_counts >= MIN_USER_INTERACTIONS].index
    )]
    book_counts = df["title"].value_counts()
    df = df[df["title"].isin(
        book_counts[book_counts >= MIN_BOOK_RATINGS].index
    )]

    print(f"  Après jointure + nettoyage + filtrage : {len(df):,} interactions")
    return df.reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée principal
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess(use_cache: bool = True) -> pd.DataFrame:
    """
    Charge, fusionne, filtre et retourne le DataFrame complet.
    Utilise un cache parquet pour éviter de recharger les CSV à chaque fois.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    if use_cache and os.path.exists(CACHE_FILE):
        print(f"\nCache trouvé → chargement rapide ({CACHE_FILE})")
        df = pd.read_parquet(CACHE_FILE)
        print(f"  {df['user_id'].nunique():,} users | "
              f"{df['title'].nunique():,} livres | {len(df):,} interactions")
        return df

    df = load_raw()
    df = df[df["rating"] >= POSITIVE_RATING_THRESHOLD].copy()
    print(f"  Interactions positives (≥{POSITIVE_RATING_THRESHOLD}) : {len(df):,}")

    df.to_parquet(CACHE_FILE, index=False)
    print(f"  Cache sauvegardé → {CACHE_FILE}")
    return df


def split_and_encode(df: pd.DataFrame) -> tuple:
    """
    Split train/test + LabelEncoders.
    Sauvegarde book_titles_ordered.npy pour que le module BERT
    puisse encoder les descriptions dans le bon ordre.

    Retourne : (train_df, test_df, le_user, le_book, n_users, n_items)
    """
    # Sous-échantillonnage optionnel (comme dans le code collègue)
    books_unique = df["title"].unique()
    if SAMPLE_SIZE and len(books_unique) > SAMPLE_SIZE:
        sampled_titles = pd.Series(books_unique).sample(
            SAMPLE_SIZE, random_state=RANDOM_STATE
        ).values
        df = df[df["title"].isin(sampled_titles)].copy()
        print(f"Sous-échantillon : {df['title'].nunique():,} livres")

    train_arr, test_arr = train_test_split(
        df.values, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_df = pd.DataFrame(train_arr, columns=df.columns)
    test_df  = pd.DataFrame(test_arr,  columns=df.columns)

    le_user = pp.LabelEncoder()
    le_book = pp.LabelEncoder()
    train_df["user_id_idx"] = le_user.fit_transform(train_df["user_id"].values)
    train_df["item_id_idx"] = le_book.fit_transform(train_df["title"].values)

    # Filtrer le test : seulement users/livres vus en train
    test_df = test_df[
        test_df["user_id"].isin(set(train_df["user_id"].unique())) &
        test_df["title"].isin(set(train_df["title"].unique()))
    ].copy()
    test_df["user_id_idx"] = le_user.transform(test_df["user_id"].values)
    test_df["item_id_idx"] = le_book.transform(test_df["title"].values)

    n_users = train_df["user_id_idx"].nunique()
    n_items = train_df["item_id_idx"].nunique()

    # Sauvegarde de l'ordre des titres → utilisé par bert_encoder pour
    # garantir que embeddings[i] correspond bien à le_book.classes_[i]
    np.save(BOOK_TITLES_FILE, le_book.classes_)
    print(f"Ordre des titres sauvegardé → {BOOK_TITLES_FILE}")
    print(f"Train : {len(train_df):,} | Test : {len(test_df):,}")
    print(f"Users : {n_users:,} | Livres : {n_items:,}")

    return train_df, test_df, le_user, le_book, n_users, n_items