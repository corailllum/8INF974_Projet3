"""
recommender.py — Recommandations et fusion des scores GNN + BERT.

Trois modes :
  1. GNN seul          : score LightGCN (collaboratif)
  2. BERT+KNN seul     : score cosinus (contenu)
  3. Fusion hybride    : alpha * score_bert + (1-alpha) * score_gnn
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from config import K, ALPHA, ALPHA_GRID


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internes
# ─────────────────────────────────────────────────────────────────────────────

def _minmax(x: np.ndarray) -> np.ndarray:
    r = x.max() - x.min()
    return (x - x.min()) / r if r > 0 else x


def _gnn_scores(
    user_idx: int,
    model,
    train_edge_index: torch.Tensor,
    n_users: int,
    n_items: int,
    device: torch.device,
) -> np.ndarray:
    """Retourne le vecteur de scores GNN (n_items,) pour un utilisateur."""
    model.eval()
    with torch.no_grad():
        _, out = model(train_edge_index)
        user_emb, item_emb = torch.split(out, (n_users, n_items))
    return torch.matmul(user_emb[user_idx], item_emb.T).cpu().numpy()


def _bert_scores(
    user_idx: int,
    train_df: pd.DataFrame,
    book_bert_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Score BERT moyen : moyenne des similarités cosinus entre
    les livres lus par l'utilisateur et tous les livres du catalogue.
    """
    train_read = train_df[train_df["user_id_idx"] == user_idx]["item_id_idx"].values
    if len(train_read) == 0:
        return np.zeros(len(book_bert_embeddings))
    ref_vecs   = book_bert_embeddings[train_read]          # (n_lus, 384)
    scores     = (ref_vecs @ book_bert_embeddings.T).mean(axis=0)   # (n_items,)
    return scores


def _mask_seen(scores: np.ndarray, seen_idx: np.ndarray) -> np.ndarray:
    scores = scores.copy()
    scores[seen_idx] = -np.inf
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Recommandation pour un utilisateur
# ─────────────────────────────────────────────────────────────────────────────

def recommend(
    user_raw_id,
    model,
    train_edge_index: torch.Tensor,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    books_df: pd.DataFrame,
    le_user,
    le_book,
    book_bert_embeddings: np.ndarray,
    n_users: int,
    n_items: int,
    device: torch.device,
    k: int = K,
    alpha: float = ALPHA,
    mode: str = "hybrid",   # "gnn" | "bert" | "hybrid"
) -> pd.DataFrame:
    """
    Recommande k livres à user_raw_id.

    Paramètres
    ----------
    alpha : float
        Poids du score BERT dans la fusion (0=GNN pur, 1=BERT pur).
    mode : str
        "gnn"    → score GNN uniquement
        "bert"   → score BERT+KNN uniquement
        "hybrid" → alpha * BERT + (1-alpha) * GNN

    Retourne un DataFrame avec colonnes :
      rang, titre, auteurs, catégories, score GNN, score BERT, score final, hit ✅
    """
    if user_raw_id not in le_user.classes_:
        print(f"[recommender] User '{user_raw_id}' inconnu.")
        return pd.DataFrame()

    user_idx  = le_user.transform([user_raw_id])[0]
    train_read = train_df[train_df["user_id_idx"] == user_idx]["item_id_idx"].values
    test_read  = set(test_df[test_df["user_id_idx"] == user_idx]["item_id_idx"].values)

    # ── Calcul des scores bruts ───────────────────────────────────────────────
    s_gnn  = _gnn_scores(user_idx, model, train_edge_index, n_users, n_items, device)
    s_bert = _bert_scores(user_idx, train_df, book_bert_embeddings)

    # ── Normalisation Min-Max → [0, 1] ────────────────────────────────────────
    s_gnn_norm  = _minmax(s_gnn)
    s_bert_norm = _minmax(s_bert)

    # ── Fusion selon le mode ──────────────────────────────────────────────────
    if mode == "gnn":
        scores_final = s_gnn_norm
    elif mode == "bert":
        scores_final = s_bert_norm
    else:   # hybrid
        scores_final = alpha * s_bert_norm + (1 - alpha) * s_gnn_norm

    scores_final = _mask_seen(scores_final, train_read)

    # ── Top-K ─────────────────────────────────────────────────────────────────
    top_k_idx    = np.argsort(scores_final)[::-1][:k]
    top_k_titles = le_book.inverse_transform(top_k_idx)

    # ── Métadonnées ───────────────────────────────────────────────────────────
    meta_lookup = books_df.set_index("title")[["authors", "categories"]].to_dict("index")

    rows = []
    for rank, (title, idx) in enumerate(zip(top_k_titles, top_k_idx), 1):
        meta = meta_lookup.get(title, {})
        rows.append({
            "rang"       : rank,
            "titre"      : title,
            "auteurs"    : meta.get("authors", ""),
            "catégories" : str(meta.get("categories", ""))[:50],
            "score GNN"  : round(float(s_gnn_norm[idx]), 3),
            "score BERT" : round(float(s_bert_norm[idx]), 3),
            "score final": round(float(scores_final[idx]), 3),
            "hit ✅"     : idx in test_read,
        })

    result = pd.DataFrame(rows).set_index("rang")

    n_hits      = result["hit ✅"].sum()
    print(f"[recommender] User {user_raw_id} | mode={mode} | alpha={alpha} | "
          f"Hits : {n_hits}/{k} | "
          f"Precision@{k} : {n_hits/k:.2f} | "
          f"Recall@{k} : {n_hits/max(len(test_read),1):.2f}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Métriques comparatives sur N utilisateurs
# ─────────────────────────────────────────────────────────────────────────────

def compare_methods(
    model,
    train_edge_index: torch.Tensor,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    le_user,
    le_book,
    book_bert_embeddings: np.ndarray,
    n_users: int,
    n_items: int,
    device: torch.device,
    k: int = K,
    n_samples: int = 300,
) -> pd.DataFrame:
    """
    Compare Recall@K et Precision@K pour :
      - Baseline aléatoire
      - BERT seul
      - GNN seul
      - Hybride (alpha optimal sur la grille ALPHA_GRID)

    Retourne un DataFrame de résultats.
    """
    test_grp  = test_df.groupby("user_id_idx")["item_id_idx"].apply(set).to_dict()
    train_grp = train_df.groupby("user_id_idx")["item_id_idx"].apply(list).to_dict()
    sample_users = list(test_grp.keys())[:n_samples]

    def eval_scores(get_score_fn):
        recalls, precisions = [], []
        for uid in sample_users:
            seen  = set(train_grp.get(uid, []))
            true  = test_grp.get(uid, set())
            scores = get_score_fn(uid)
            scores_masked = _mask_seen(scores, np.array(list(seen)))
            top_k = set(np.argsort(scores_masked)[::-1][:k])
            hits  = len(top_k & true)
            recalls.append(hits / max(len(true), 1))
            precisions.append(hits / k)
        return float(np.mean(recalls)), float(np.mean(precisions))

    # Aléatoire
    def random_scores(uid):
        s = np.random.rand(n_items)
        return s
    rand_r, rand_p = eval_scores(random_scores)

    # GNN seul
    s_gnn_all = {}
    def gnn_score_fn(uid):
        if uid not in s_gnn_all:
            s_gnn_all[uid] = _minmax(_gnn_scores(uid, model, train_edge_index, n_users, n_items, device))
        return s_gnn_all[uid]
    gnn_r, gnn_p = eval_scores(gnn_score_fn)

    # BERT seul
    def bert_score_fn(uid):
        return _minmax(_bert_scores(uid, train_df, book_bert_embeddings))
    bert_r, bert_p = eval_scores(bert_score_fn)

    # Hybride : chercher le meilleur alpha
    best_alpha, best_r, best_p = ALPHA, 0.0, 0.0
    alpha_results = []
    for alpha in ALPHA_GRID:
        def hybrid_fn(uid, a=alpha):
            s_g = _minmax(_gnn_scores(uid, model, train_edge_index, n_users, n_items, device))
            s_b = _minmax(_bert_scores(uid, train_df, book_bert_embeddings))
            return a * s_b + (1 - a) * s_g
        r, p = eval_scores(hybrid_fn)
        alpha_results.append({"alpha": alpha, f"Recall@{k}": r, f"Precision@{k}": p})
        if r > best_r:
            best_r, best_p, best_alpha = r, p, alpha

    results = pd.DataFrame([
        {"Méthode": "Aléatoire",            f"Recall@{k}": rand_r, f"Precision@{k}": rand_p},
        {"Méthode": "BERT+KNN seul",        f"Recall@{k}": bert_r, f"Precision@{k}": bert_p},
        {"Méthode": "GNN seul",             f"Recall@{k}": gnn_r,  f"Precision@{k}": gnn_p},
        {"Méthode": f"Hybride (α={best_alpha})", f"Recall@{k}": best_r, f"Precision@{k}": best_p},
    ])

    print(f"\n{'Méthode':<30} {'Recall@'+str(k):>12} {'Precision@'+str(k):>14}")
    print("-" * 60)
    for _, row in results.iterrows():
        print(f"{row['Méthode']:<30} {row[f'Recall@{k}']:>12.4f} {row[f'Precision@{k}']:>14.4f}")
    print(f"\nMeilleur alpha : {best_alpha}  (Recall@{k} = {best_r:.4f})")

    return results, pd.DataFrame(alpha_results)


# ─────────────────────────────────────────────────────────────────────────────
# Similarité BERT pure (test qualitatif — conservé depuis le notebook)
# ─────────────────────────────────────────────────────────────────────────────

def find_similar_books(
    query_title: str,
    le_book,
    book_bert_embeddings: np.ndarray,
    knn: NearestNeighbors,
    desc_lookup: dict,
    k: int = 10,
) -> pd.DataFrame:
    """Trouve les k livres les plus similaires sémantiquement (BERT+KNN)."""
    if query_title not in le_book.classes_:
        matches = [t for t in le_book.classes_ if query_title.lower() in t.lower()]
        print(f"Titre non trouvé. Proches : {matches[:5]}")
        return pd.DataFrame()

    idx       = le_book.transform([query_title])[0]
    query_vec = book_bert_embeddings[idx].reshape(1, -1)

    dists, idxs = knn.kneighbors(query_vec, n_neighbors=k + 1)
    sims  = 1 - dists[0]
    idxs  = idxs[0]
    mask  = sims < 0.9999
    sims, idxs = sims[mask][:k], idxs[mask][:k]

    rows = []
    for rank, (ni, sim) in enumerate(zip(idxs, sims), 1):
        title = le_book.classes_[ni]
        rows.append({
            "rang"       : rank,
            "titre"      : title,
            "similarité" : f"{sim*100:.1f}%",
            "description": desc_lookup.get(title, "")[:120] + "...",
        })
    return pd.DataFrame(rows).set_index("rang")