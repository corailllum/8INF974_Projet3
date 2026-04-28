import numpy as np
import pandas as pd
import torch

# Normalisation du score  [0, 1]
def minmax(x: np.ndarray) -> np.ndarray:
    r = x.max() - x.min()
    return (x - x.min()) / r if r > 0 else x

# Score aléatoire
def random_scores(n_items: int) -> np.ndarray:
    return np.random.rand(n_items)

# Score BERT
def bert_scores(
    user_idx: int,
    train_df: pd.DataFrame,
    book_bert_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Score BERT moyen : moyenne des similarités cosinus entre
    les livres lus par l'utilisateur et tous les livres du catalogue
    """
    train_read = train_df[train_df["user_id_idx"] == user_idx]["item_id_idx"].values
    if len(train_read) == 0:
        return np.zeros(len(book_bert_embeddings))
    ref_vecs = book_bert_embeddings[train_read]          # (n_lus, 384)
    scores = (ref_vecs @ book_bert_embeddings.T).mean(axis=0)   # (n_items,)
    return scores

# Score GNN
def gnn_scores(
    user_idx: int,
    model,
    train_edge_index: torch.Tensor,
    n_users: int,
    n_items: int,
) -> np.ndarray:
    """Vecteur de scores GNN (n_items,) pour un utilisateur"""
    model.eval()
    with torch.no_grad():
        _, out = model(train_edge_index)
        user_emb, item_emb = torch.split(out, (n_users, n_items))
    return torch.matmul(user_emb[user_idx], item_emb.T).cpu().numpy()


# Masque les livres déjà lus
def mask_read(scores: np.ndarray, read_idx: np.ndarray) -> np.ndarray:
    scores = scores.copy()
    for idx in read_idx:
        scores[idx] = -np.inf
    return scores


# Recommandation pour un utilisateur
def recommend_books(
    user_raw_id,
    model,
    train_edge_index: torch.Tensor,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    books_df: pd.DataFrame,
    le_user,
    le_book,
    n_users: int,
    n_items: int,
    K: int = 10,
) -> pd.DataFrame:
    if user_raw_id not in le_user.classes_:
        print(f"User '{user_raw_id}' inconnu.")
        return pd.DataFrame()

    # User et ses livres lus dans le train (à masquer) ET test (vérité terrain)
    user_idx  = le_user.transform([user_raw_id])[0]
    train_read = train_df[train_df["user_id_idx"] == user_idx]["item_id_idx"].values
    test_read  = set(test_df[test_df["user_id_idx"] == user_idx]["item_id_idx"].values)

    # Score et masque
    s_gnn  = gnn_scores(user_idx, model, train_edge_index, n_users, n_items)
    s_gnn_norm = minmax(s_gnn)
    scores= mask_read(s_gnn_norm, train_read)

    # Top-K recommandations
    top_k_idx    = np.argsort(scores)[::-1][:K]
    top_k_titles = le_book.inverse_transform(top_k_idx)

    # Métadonnées
    meta_lookup = books_df.set_index("title")[["authors", "categories"]].to_dict("index")

    rows = []
    for rank, (title, idx) in enumerate(zip(top_k_titles, top_k_idx), 1):
        meta = meta_lookup.get(title, {})
        rows.append({
            "rang"          : rank,
            "titre"         : title,
            "auteurs"       : meta.get("authors", ""),
            "catégories"    : str(meta.get("categories", ""))[:50],
            "score"         : round(float(scores[idx]), 3),
            "hit"           : idx in test_read,
        })

    result = pd.DataFrame(rows).set_index("rang")

    #  Résumé des performances
    n_hits = result["hit"].sum()
    precision_u = n_hits / K
    recall_u = n_hits / max(len(test_read), 1)
    print(f"User {user_raw_id} — "
          f"Livres dans le test : {len(test_read)} | "
          f"Hits : {n_hits}/{K} | "
          f"Precision@{K} : {precision_u:.2f} | "
          f"Recall@{K} : {recall_u:.2f}")

    return result

# Affichage des livres déjà lus
def show_already_read(
    user_raw_id,
    df: pd.DataFrame,
    books_df: pd.DataFrame,
    le_user,
    le_book,
    label: str = "",
) -> pd.DataFrame:
    """Affiche les livres lus par un utilisateur"""
    user_idx = le_user.transform([user_raw_id])[0]
    read_idx = df[df['user_id_idx'] == user_idx]['item_id_idx'].values
    title_list = le_book.inverse_transform(read_idx)
    result = (
        books_df[books_df["title"].isin(title_list)]
        [["title", "authors", "categories"]]
        .reset_index(drop=True)
    )
    print(f"Livres lus ({label}) — User {user_raw_id} : {len(result)} livres")
    return result


# Métriques comparatives sur N utilisateurs
def compare_recommendations(
        user_raw_id,
        model,
        train_edge_index,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        books_df: pd.DataFrame,
        le_user,
        le_book,
        book_bert_embeddings,
        n_users: int,
        n_items: int,
        K: int = 10,
) -> pd.DataFrame:
    if user_raw_id not in le_user.classes_:
        print(f"User '{user_raw_id}' inconnu du modèle.")
        return pd.DataFrame()

    # User et ses livres lus dans le train (à masquer) ET test (vérité terrain)
    user_idx = le_user.transform([user_raw_id])[0]
    train_read = train_df[train_df["user_id_idx"] == user_idx]["item_id_idx"].values
    test_read = set(test_df[test_df["user_id_idx"] == user_idx]["item_id_idx"].values)

    # Calcul des scores pour chaque méthode
    scores = {
        "Aléatoire": minmax(random_scores(n_items)),
        "BERT+KNN": minmax(bert_scores(user_idx, train_df, book_bert_embeddings)),
        "GNN hybride": minmax(gnn_scores(user_idx, model, train_edge_index, n_users, n_items)),
    }

    # Top-K par méthode, en masquant les livres déjà lus
    meta_lookup = books_df.set_index("title")[["authors"]].to_dict("index")

    results = {}
    summaries = []

    for method_name, s in scores.items():
        s_masked = mask_read(s, train_read)
        top_k_idx = np.argsort(s_masked)[::-1][:K]
        top_k_titles = le_book.inverse_transform(top_k_idx)

        hits = [idx in test_read for idx in top_k_idx]
        n_hits = sum(hits)
        precision = n_hits / K
        recall = n_hits / max(len(test_read), 1)

        summaries.append({
            "Méthode": method_name,
            f"Recall@{K}": round(recall, 4),
            f"Precision@{K}": round(precision, 4),
            "Hits": f"{n_hits}/{K}",
        })

        results[method_name] = pd.DataFrame({
            "titre": top_k_titles,
            "auteurs": [meta_lookup.get(t, {}).get("authors", "")[:30] for t in top_k_titles],
            "score": [round(float(s_masked[i]), 3) for i in top_k_idx],
            "hit": ["✅" if h else "❌" for h in hits],
        }, index=range(1, K + 1))
        results[method_name].index.name = "rang"

    # Affichage du résumé
    summary_df = pd.DataFrame(summaries).set_index("Méthode")

    print(f"\nUser {user_raw_id} — Test : {len(test_read)} livres | K = {K}")
    print("=" * 55)
    print(summary_df.to_string())
    print("=" * 55)

    print("\n--- Recommandations par méthode ---\n")
    for method_name, df_recs in results.items():
        print(f"[ {method_name} ]")
        print(df_recs[["titre", "score", "hit"]].to_string())
        print()

    return summary_df