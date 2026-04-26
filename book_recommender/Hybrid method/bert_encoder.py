"""
bert_encoder.py — Code de la collègue, conservé tel quel.

Seuls ajouts pour l'intégration dans le pipeline hybride :
  - imports depuis config.py  (remplace les constantes locales)
  - fonction run_bert_pipeline()  (colle au main() original mais retourne knn + embeddings)
  - le main() original est conservé intact en bas du fichier
"""

import os
import time
import random
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
import torch

from config import (
    SBERT_MODEL_NAME  as MODEL_NAME,
    SBERT_BATCH_SIZE  as BATCH_SIZE,
    BERT_EMBEDDINGS_FILE,
    BOOK_TITLES_FILE,
    KNN_FILE,
    CACHE_DIR,
    RANDOM_STATE,
)

# Constantes locales
DESCRIPTION_COL = "description"
TITLE_COL       = "title"
TOP_K           = 5

# Device GPU/Mac
def detect_device() -> str:
    if torch.cuda.is_available():
        device = "cuda"
        name   = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   CUDA detecte  -> {name} ({mem_gb:.1f} GB VRAM)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("   MPS detecte   -> Apple Silicon GPU")
    else:
        device = "cpu"
        n_cores = torch.get_num_threads()
        print(f"   CPU uniquement -> {n_cores} threads")
    return device



# Encodage des descriptions
def encode_descriptions(model, descriptions: list, split_name: str, device: str) -> np.ndarray:
    n         = len(descriptions)
    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
    all_embeds = []

    print(f"\nEncodage BERT - {split_name} ({n} livres, {n_batches} batches)")
    print(f"   Modele : {MODEL_NAME}    Batch size : {BATCH_SIZE}    Device : {device.upper()}")
    print(f"{'Batch':>6}   {'Temps (s)':>10}   {'Temps cum. (s)':>14}   Progression")

    epoch_start = time.time()

    for i in range(n_batches):
        batch      = descriptions[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        t0         = time.time()
        embeddings = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        elapsed    = time.time() - t0
        cumul      = time.time() - epoch_start
        pct        = (i + 1) / n_batches * 100

        print(f"{i+1:>6}   {elapsed:>10.3f}   {cumul:>14.1f}   {pct:5.1f}%")
        all_embeds.append(embeddings)

    epoch_time     = time.time() - epoch_start
    embeddings_all = np.vstack(all_embeds)

    print(f"Encodage termine en {epoch_time:.1f}s  |  Shape : {embeddings_all.shape}")
    return embeddings_all


# Fine-tuning KNN
def train_knn(train_embeddings: np.ndarray, k: int = TOP_K + 1) -> NearestNeighbors:
    print(f"\nEntrainement KNN (k={k}, metric=cosine)...")
    t0  = time.time()
    knn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=-1)
    knn.fit(train_embeddings)
    print(f"   Fit termine en {time.time() - t0:.3f}s")
    return knn


# Evaluation
def evaluate(
    knn: NearestNeighbors,
    test_df: pd.DataFrame,
    test_embeddings: np.ndarray,
    k: int = TOP_K,
):
    print(f"\nEvaluation sur {len(test_df)} livres de test...")

    sim_scores = []
    eval_start = time.time()

    print(f"{'Livre #':>8}   {'Sim moy':>9}")

    for idx in range(len(test_df)):
        query_embed        = test_embeddings[idx].reshape(1, -1)
        distances, indices = knn.kneighbors(query_embed, n_neighbors=k + 1)

        sims = 1 - distances[0]
        sims = sims[sims < 0.9999][:k]

        sim_mean = float(np.mean(sims)) if len(sims) > 0 else 0.0
        sim_scores.append(sim_mean)

        if idx % max(1, len(test_df) // 10) == 0:
            print(f"{idx:>8}   {sim_mean:>9.4f}")

    elapsed = time.time() - eval_start
    avg_sim = float(np.mean(sim_scores)) * 100

    print(f"\nResultats finaux :")
    print(f"   Similarite cosinus moyenne : {avg_sim:.1f}%")
    print(f"   Temps d'evaluation         : {elapsed:.1f}s")
    return avg_sim


# Test visuel
def qualitative_test(
    knn: NearestNeighbors,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_embeddings: np.ndarray,
    k: int = TOP_K,
):
    idx  = random.randint(0, len(test_df) - 1)
    book = test_df.iloc[idx]

    print("un livre tire au sort dans le test set")
    print(f"\nLIVRE REQUETE (index {idx})")
    print(f"   Titre       : {book[TITLE_COL]}")
    if "authors" in book:
        print(f"   Auteur      : {book['authors']}")
    if "categories" in book:
        print(f"   Categories  : {book['categories']}")
    desc_preview = str(book[DESCRIPTION_COL])[:300].replace("\n", " ")
    print(f"   Description : {desc_preview}...")

    query_embed        = test_embeddings[idx].reshape(1, -1)
    distances, indices = knn.kneighbors(query_embed, n_neighbors=k + 1)
    sims               = 1 - distances[0]
    indices            = indices[0]

    mask    = sims < 0.9999
    sims    = sims[mask][:k]
    indices = indices[mask][:k]

    print(f"\nTOP {k} livres similaires")

    for rank, (ni, sim) in enumerate(zip(indices, sims), 1):
        neighbor = train_df.iloc[ni]
        n_desc   = str(neighbor[DESCRIPTION_COL])[:200].replace("\n", " ")
        sim_pct  = sim * 100

        print(f"\n  #{rank}  {neighbor[TITLE_COL]}")
        if "authors" in neighbor:
            print(f"       Auteur      : {neighbor['authors']}")
        if "categories" in neighbor:
            print(f"       Categories  : {neighbor['categories']}")
        print(f"       Similarite  : {sim_pct:.1f}%")
        print(f"       Description : {n_desc}...")

    print("\n" + "=" * 70)


# Pipeline necessaie au pipeline hybride
def run_bert_pipeline(
    desc_lookup: dict,
    use_cache: bool = True,
) -> tuple:
    """
    Appelée par pipeline_hybrid.py pour intégrer le module BERT dans le pipeline hybride.

    Différence avec le main() original :
      - L'encodage se fait dans l'ordre de le_book.classes_ (book_titles_ordered.npy)
        pour garantir que embeddings[i] <-> le_book.classes_[i] <-> noeud livre i dans PyG.
      - Retourne (knn, embeddings) au lieu de tout afficher et quitter.
      - Le cache evite de re-encoder a chaque lancement.

    Parametres
    ----------
    desc_lookup : dict  {title -> description}  — fourni par pipeline_hybrid.py apres preprocessing
    use_cache   : bool  — si True, charge depuis le disque si disponible
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Ordre des titres = le_book.classes_ sauvegarde par preprocessing.split_and_encode()
    if not os.path.exists(BOOK_TITLES_FILE):
        raise FileNotFoundError(
            f"{BOOK_TITLES_FILE} introuvable. "
            "Lance d'abord preprocessing.split_and_encode() dans pipeline_hybrid.py."
        )
    book_titles  = np.load(BOOK_TITLES_FILE, allow_pickle=True)
    descriptions = [desc_lookup.get(t, "") for t in book_titles]

    missing = sum(1 for d in descriptions if len(d) < 10)
    if missing:
        print(f"   {missing}/{len(book_titles)} livres sans description")

    # Embeddings
    if use_cache and os.path.exists(BERT_EMBEDDINGS_FILE):
        print(f"\nCache trouve, chargement embeddings -> {BERT_EMBEDDINGS_FILE}")
        embeddings = np.load(BERT_EMBEDDINGS_FILE)
        assert embeddings.shape[0] == len(book_titles), (
            f"Cache incoherent : {embeddings.shape[0]} embeddings "
            f"pour {len(book_titles)} livres. Supprime le cache et relance."
        )
    else:
        print(f"\nDetection du device d'acceleration...")
        device = detect_device()
        print(f"\n   Chargement du modele SBERT : {MODEL_NAME}...")
        model = SentenceTransformer(MODEL_NAME, device=device)
        print("   Modele pret")

        embeddings = encode_descriptions(model, descriptions, "ALL BOOKS", device)

        np.save(BERT_EMBEDDINGS_FILE, embeddings)
        print(f"Embeddings sauvegardes -> {BERT_EMBEDDINGS_FILE}")

    # KNN
    if use_cache and os.path.exists(KNN_FILE):
        print(f"Cache trouve, chargement KNN -> {KNN_FILE}")
        knn = joblib.load(KNN_FILE)
    else:
        knn = train_knn(embeddings)
        joblib.dump(knn, KNN_FILE)
        print(f"Modele sauvegarde -> {KNN_FILE}")

    return knn, embeddings


# main
def main() -> None:
    from preprocessing import load_and_preprocess, split_and_encode

    start_total = time.time()

    # 1. Preprocessing partage — avec cache parquet pour eviter de recharger le CSV a chaque fois
    df = load_and_preprocess(use_cache=True)
    train_df, test_df, le_user, le_book, n_users, n_items = split_and_encode(df)

    books_df = (
        df.drop_duplicates(subset=["title"])
          [["title", "description", "authors", "categories"]]
          .reset_index(drop=True)
    )

    # 2. Split (sur les livres uniques, comme dans le code original)
    train_books, test_books = train_test_split(

        books_df, test_size=0.15, random_state=RANDOM_STATE
    )
    train_books = train_books.reset_index(drop=True)
    test_books  = test_books.reset_index(drop=True)
    print(f"\nSplit 85/15 : {len(train_books)} train / {len(test_books)} test")

    # 3. Device & modele SBERT
    print(f"\nDetection du device d'acceleration...")
    device = detect_device()
    print(f"\n   Chargement du modele SBERT : {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    print("   Modele pret")

    # 4. Encodage
    train_embeddings = encode_descriptions(model, train_books[DESCRIPTION_COL].tolist(), "TRAIN", device)
    test_embeddings  = encode_descriptions(model, test_books[DESCRIPTION_COL].tolist(),  "TEST",  device)

    # 5. KNN
    knn = train_knn(train_embeddings)

    # 6. Sauvegarde du modele
    joblib.dump(knn, KNN_FILE)
    np.save(BERT_EMBEDDINGS_FILE, train_embeddings)
    train_books.to_parquet(os.path.join(CACHE_DIR, "train_df.parquet"), index=False)
    print(f"\nModele sauvegarde -> {KNN_FILE}")
    print(f"Embeddings sauvegardes -> {BERT_EMBEDDINGS_FILE}")

    # 7. Evaluation
    evaluate(knn, test_books, test_embeddings)

    # 8. Test qualitatif
    qualitative_test(knn, train_books, test_books, test_embeddings)

    print(f"\nTemps total : {time.time() - start_total:.1f}s")
    print("Pipeline fini")

if __name__ == "__main__":
    main()
