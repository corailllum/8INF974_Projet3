"""
Système de recommandation de livres — BERT + KNN
Dataset : Goodreads 100k (kaggle.com/datasets/mdhamani/goodreads-books-100k)

Usage :
    pip install pandas scikit-learn sentence-transformers torch tqdm numpy
    python book_recommendation.py --csv path/to/books.csv
"""

import argparse
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DESCRIPTION_COL = "desc"          # nom de la colonne description dans le CSV
TITLE_COL       = "title"         # nom de la colonne titre
GENRE_COL       = "genre"         # nom de la colonne genre dans le CSV
SAMPLE_SIZE     = None            # nb de livres à utiliser (None = tout le dataset)
TEST_SIZE       = 0.15            # 85/15 split
TOP_K           = 10              # nb de livres similaires à retourner
MODEL_NAME      = "all-MiniLM-L6-v2"  # modèle SBERT léger et performant
BATCH_SIZE      = 64              # batch pour l'encodage BERT
RANDOM_STATE    = 42


# ─────────────────────────────────────────────
# DÉTECTION AUTOMATIQUE DU DEVICE
# ─────────────────────────────────────────────
def detect_device() -> str:
    """
    Sélectionne automatiquement le meilleur device disponible :
      - CUDA  : GPU NVIDIA (Linux / Windows)
      - MPS   : GPU Apple Silicon (macOS M1/M2/M3)
      - CPU   : fallback universel
    """
    if torch.cuda.is_available():
        device = "cuda"
        name   = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   CUDA détecté  -> {name} ({mem_gb:.1f} GB VRAM)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("   MPS détecté   -> Apple Silicon GPU")
    else:
        device = "cpu"
        n_cores = torch.get_num_threads()
        print(f"   CPU uniquement -> {n_cores} threads")
    return device

# ─────────────────────────────────────────────
# 1. CHARGEMENT & NETTOYAGE
# ─────────────────────────────────────────────
def load_data(csv_path: str) -> pd.DataFrame:
    print("\nChargement du dataset...")
    df = pd.read_csv(csv_path, on_bad_lines="skip", low_memory=False)
    print(f"   {len(df)} livres chargés")

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    for col in [DESCRIPTION_COL, TITLE_COL]:
        if col not in df.columns:
            raise ValueError(
                f"Colonne '{col}' introuvable. Colonnes disponibles : {list(df.columns)}"
            )

    df = df.dropna(subset=[DESCRIPTION_COL, TITLE_COL])
    df[DESCRIPTION_COL] = df[DESCRIPTION_COL].astype(str).str.strip()
    df = df[df[DESCRIPTION_COL].str.len() > 50]
    df = df.drop_duplicates(subset=[TITLE_COL])
    df = df.reset_index(drop=True)

    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"   Sous-échantillon : {len(df)} livres retenus")

    print(f"   Apres nettoyage : {len(df)} livres exploitables")
    return df


def split_data(df: pd.DataFrame):
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)
    print(f"\nSplit 85/15 : {len(train_df)} train | {len(test_df)} test")
    return train_df, test_df


# ─────────────────────────────────────────────
# 3. ENCODAGE BERT — avec suivi par époque/batch
# ─────────────────────────────────────────────
def encode_descriptions(model, descriptions: list, split_name: str, device: str) -> np.ndarray:
    """Encode les descriptions par batch avec affichage de la progression."""
    n          = len(descriptions)
    n_batches  = (n + BATCH_SIZE - 1) // BATCH_SIZE
    all_embeds = []
    total_loss = 0.0

    print(f"\nEncodage BERT -- {split_name} ({n} livres, {n_batches} batches)")
    print(f"   Modele : {MODEL_NAME}  |  Batch size : {BATCH_SIZE}  |  Device : {device.upper()}")
    print("-" * 60)
    print(f"{'Batch':>6}   {'Temps (s)':>10}   {'Norme L2 moy':>13}   {'Delta norme':>11}   Progression")
    print("-" * 60)

    prev_norm   = None
    epoch_start = time.time()

    for i in range(n_batches):
        batch      = descriptions[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        t0         = time.time()
        embeddings = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        elapsed    = time.time() - t0

        norm    = float(np.mean(np.linalg.norm(embeddings, axis=1)))
        delta   = abs(norm - prev_norm) if prev_norm is not None else 0.0
        prev_norm = norm
        total_loss += delta

        pct = (i + 1) / n_batches * 100
        print(f"{i+1:>6}   {elapsed:>10.3f}   {norm:>13.4f}   {delta:>11.5f}   {pct:5.1f}%")

        all_embeds.append(embeddings)

    epoch_time     = time.time() - epoch_start
    embeddings_all = np.vstack(all_embeds)

    print("-" * 60)
    print(f"Encodage termine en {epoch_time:.1f}s")
    print(f"   Delta norme cumulee (proxy loss) : {total_loss:.5f}")
    print(f"   Shape embeddings : {embeddings_all.shape}")
    return embeddings_all


def train_knn(train_embeddings: np.ndarray, k: int = TOP_K + 1) -> NearestNeighbors:
    print(f"\nEntrainement KNN (k={k}, metric=cosine)...")
    t0  = time.time()
    knn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=-1)
    knn.fit(train_embeddings)
    print(f"   Fit termine en {time.time() - t0:.3f}s")
    return knn


def evaluate(
    knn: NearestNeighbors,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_embeddings: np.ndarray,
    train_embeddings: np.ndarray,
    k: int = TOP_K,
):
    print(f"\nEvaluation sur {len(test_df)} livres de test...")

    hits       = 0
    sim_scores = []
    eval_start = time.time()

    print("-" * 50)
    print(f"{'Livre #':>8}   {'Acc@k hit':>10}   {'Sim moy':>9}")
    print("-" * 50)

    for idx in range(len(test_df)):
        query_embed        = test_embeddings[idx].reshape(1, -1)
        distances, indices = knn.kneighbors(query_embed, n_neighbors=k + 1)

        sims    = 1 - distances[0]
        indices = indices[0]

        mask    = sims < 0.9999
        sims    = sims[mask][:k]
        indices = indices[mask][:k]

        sim_mean = float(np.mean(sims)) if len(sims) > 0 else 0.0
        sim_scores.append(sim_mean)

        # Accuracy@k via colonne genre
        hit         = False
        query_genre = str(test_df.iloc[idx].get(GENRE_COL, ""))
        if query_genre and query_genre != "nan":
            for ni in indices:
                neighbor_genre = str(train_df.iloc[ni].get(GENRE_COL, ""))
                if query_genre in neighbor_genre:
                    hit = True
                    break
        else:
            hit = sim_mean > 0.50

        if hit:
            hits += 1

        if idx % max(1, len(test_df) // 10) == 0:
            print(f"{idx:>8}   {'oui' if hit else 'non':>10}   {sim_mean:>9.4f}")

    elapsed  = time.time() - eval_start
    accuracy = hits / len(test_df) * 100
    avg_sim  = float(np.mean(sim_scores)) * 100

    print("-" * 50)
    print(f"\nResultats finaux :")
    print(f"   Accuracy@{k}        : {accuracy:.1f}%")
    print(f"   Similarite cos moy  : {avg_sim:.1f}%")
    print(f"   Temps d'evaluation  : {elapsed:.1f}s")
    return accuracy, avg_sim


def qualitative_test(
    knn: NearestNeighbors,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_embeddings: np.ndarray,
    k: int = TOP_K,
):
    idx  = random.randint(0, len(test_df) - 1)
    book = test_df.iloc[idx]

    print("\n" + "=" * 70)
    print("TEST QUALITATIF -- livre tire au sort dans le test set")
    print("=" * 70)
    print(f"\nLIVRE REQUETE (index {idx})")
    print(f"   Titre       : {book[TITLE_COL]}")
    if "author" in book:
        print(f"   Auteur      : {book['author']}")
    if GENRE_COL in book:
        print(f"   Genre       : {book[GENRE_COL]}")
    desc_preview = str(book[DESCRIPTION_COL])[:300].replace("\n", " ")
    print(f"   Description : {desc_preview}...")

    query_embed        = test_embeddings[idx].reshape(1, -1)
    distances, indices = knn.kneighbors(query_embed, n_neighbors=k + 1)
    sims               = 1 - distances[0]
    indices            = indices[0]

    mask    = sims < 0.9999
    sims    = sims[mask][:k]
    indices = indices[mask][:k]

    print(f"\nTOP {k} LIVRES SIMILAIRES")
    print("-" * 70)

    for rank, (ni, sim) in enumerate(zip(indices, sims), 1):
        neighbor = train_df.iloc[ni]
        n_desc   = str(neighbor[DESCRIPTION_COL])[:200].replace("\n", " ")
        sim_pct  = sim * 100

        print(f"\n  #{rank}  {neighbor[TITLE_COL]}")
        if "author" in neighbor:
            print(f"       Auteur      : {neighbor['author']}")
        if GENRE_COL in neighbor:
            print(f"       Genre       : {neighbor[GENRE_COL]}")
        print(f"       Similarite  : {sim_pct:.1f}%")
        print(f"       Description : {n_desc}...")

    print("\n" + "=" * 70)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Recommandation de livres — SBERT + KNN"
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Chemin vers le fichier CSV Goodreads 100k",
    )
    args = parser.parse_args()

    start_total = time.time()

    # 1. Données
    df                  = load_data(args.csv)
    train_df, test_df   = split_data(df)

    # 2. Détection device & chargement modèle BERT
    print(f"\nDetection du device d'acceleration...")
    device = detect_device()
    print(f"\n   Chargement du modele SBERT : {MODEL_NAME}...")
    model  = SentenceTransformer(MODEL_NAME, device=device)
    print("   Modele pret")

    # 3. Encodage
    train_descriptions = train_df[DESCRIPTION_COL].tolist()
    test_descriptions  = test_df[DESCRIPTION_COL].tolist()

    train_embeddings = encode_descriptions(model, train_descriptions, "TRAIN", device)
    test_embeddings  = encode_descriptions(model, test_descriptions,  "TEST",  device)

    # 4. KNN
    knn = train_knn(train_embeddings)

    # 5. Évaluation
    accuracy, avg_sim = evaluate(
        knn, train_df, test_df, test_embeddings, train_embeddings
    )

    # 6. Test qualitatif
    qualitative_test(knn, train_df, test_df, test_embeddings)

    print(f"\nTemps total : {time.time() - start_total:.1f}s")
    print("Pipeline termine !")


if __name__ == "__main__":
    main()