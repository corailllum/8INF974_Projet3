import argparse
from config import *
from preprocessing import *
from bert_encoder import *
from lightgcn_trainer import *
from recommender import *
from model import HybridLightGCN

# Device GPU/Mac
def get_device() -> torch.device:
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"CUDA : {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        d = torch.device("mps")
        print("MPS : Apple Silicon")
    else:
        d = torch.device("cpu")
        print(f"CPU : {torch.get_num_threads()} threads")
    return d


def run_pipeline(args):
    device = get_device()
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 1. Preprocessing partage  ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ÉTAPE 1 — Prétraitement")
    print("=" * 60)
    df = load_and_preprocess(use_cache=not args.no_cache)
    train_df, test_df, le_user, le_book, n_users, n_items = split_and_encode(df)

    desc_lookup = (
        df.drop_duplicates(subset=["title"])
        .set_index("title")["description"]
        .to_dict()
    )

    # 2. Encodage des descriptions avec BERT  ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ÉTAPE 2 — Encodage BERT des descriptions")
    print("=" * 60)
    knn, book_bert_embeddings = run_bert_pipeline(
        desc_lookup,
        use_cache=not args.skip_bert
    )
    book_bert_tensor = torch.tensor(
        book_bert_embeddings, dtype=torch.float32
    ).to(device)

    # 3.Graphe biparti  ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ÉTAPE 3 — Construction du graphe biparti")
    print("=" * 60)
    train_edge_index = build_edge_index(train_df, n_users, device)
    print(f"Edge index : {train_edge_index.shape}  (2 × 2N arêtes non orientés)")

    # 4. Entrainement du modèle HybridLightGCN  ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ÉTAPE 4 — Modèle HybridLightGCN")
    print("=" * 60)
    model = HybridLightGCN(
        num_users=n_users,
        num_items=n_items,
        book_bert_embeddings=book_bert_tensor,
        latent_dim=LGCN_LATENT_DIM,
        num_layers=LGCN_N_LAYERS,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres apprenables : {n_params:,}")
    print(f"  • user_embedding : {n_users} × {LGCN_LATENT_DIM}")
    print(f"  • book_proj      : {book_bert_embeddings.shape[1]} → {LGCN_LATENT_DIM}")
    print(f"  • book_bert      : GELÉ (non entraîné)")

    if args.load_checkpoint and os.path.exists(GNN_CHECKPOINT):
        model.load_state_dict(torch.load(GNN_CHECKPOINT, map_location=device))
        print(f"Checkpoint chargé → {GNN_CHECKPOINT}")
        history = None
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LGCN_LR)
        history = train_and_eval(
            model, optimizer, train_df, test_df,
            train_edge_index, n_users, n_items, device,
            epochs=LGCN_EPOCHS, batch_size=LGCN_BATCH_SIZE, decay=LGCN_DECAY, k=K
        )
        torch.save(model.state_dict(), GNN_CHECKPOINT)
        print(f"Checkpoint sauvegardé → {GNN_CHECKPOINT}")

        best_recall    = max(r for r in history['recall']    if r >= 0)
        best_precision = max(p for p in history['precision'] if p >= 0)
        print(f"Meilleur Recall@{K}    : {best_recall:.4f}")
        print(f"Meilleure Precision@{K}: {best_precision:.4f}")
        plot_training(history)

    # 5. Recommandations  ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ÉTAPE 5 — Recommandations")
    print("=" * 60)

    example_user = args.user or test_df["user_id"].iloc[0]
    print(f"\n[main] Utilisateur exemple : {example_user}\n")

    for mode in ["gnn", "bert", "hybrid"]:
        print(f"\n--- Mode : {mode.upper()} ---")
        recs = recommend(
            user_raw_id=example_user,
            model=model,
            train_edge_index=train_edge_index,
            train_df=train_df,
            test_df=test_df,
            books_df=df.drop_duplicates(subset=["title"]),
            le_user=le_user,
            le_book=le_book,
            book_bert_embeddings=book_bert_embeddings,
            n_users=n_users,
            n_items=n_items,
            device=device,
            k=10,
            mode=mode,
        )
        print(recs.to_string())
    # ── 6. Comparaison des méthodes ───────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline hybride LightGCN + BERT pour la recommandation de livres"
    )
    parser.add_argument(
        "--skip-bert", action="store_true",
        help="Charge les embeddings BERT depuis le cache (évite le ré-encodage)"
    )
    parser.add_argument(
        "--load-checkpoint", action="store_true",
        help="Charge le checkpoint GNN sauvegardé (évite le ré-entraînement)"
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Force le rechargement des CSV (ignore le cache parquet)"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Affiche la comparaison des méthodes"
    )
    parser.add_argument(
        "--user", type=str, default=None,
        help="User-ID pour lequel générer des recommandations"
    )
    args = parser.parse_args()
    run_pipeline(args)
