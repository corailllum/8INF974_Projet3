import os
import random

import numpy as np
import torch.nn.functional as F
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from config import LGCN_EPOCHS, LGCN_BATCH_SIZE, LGCN_DECAY, K, CACHE_DIR
EVAL_EVERY = 5

# Graphe biparti non orienté
def build_edge_index(
    train_df, n_users: int, device: torch.device
) -> torch.Tensor:
    u_t = torch.LongTensor(train_df.user_id_idx.values.copy())
    i_t = torch.LongTensor(train_df.item_id_idx.values.copy()) + n_users #indices livres décalés de +n_users pour avoir des IDs uniques.
    train_edge_index = torch.stack([
        torch.cat([u_t, i_t]),
        torch.cat([i_t, u_t]),
    ]).to(device)
    return train_edge_index


def build_user_items_dict(train_df: pd.DataFrame) -> dict:
    """
    {user_idx: set(item_indices)} précalculé une seule fois.
    """
    return {
        int(uid): set(items)
        for uid, items in train_df.groupby("user_id_idx")["item_id_idx"].apply(list).items()
    }

def build_train_mask(
        train_df: pd.DataFrame,
        n_users: int,
        n_items: int,
        device: torch.device,
) -> torch.Tensor:
    rows = torch.LongTensor(train_df["user_id_idx"].values.copy())
    cols = torch.LongTensor(train_df["item_id_idx"].values.copy())
    i = torch.stack([rows, cols])
    v = torch.ones(len(train_df), dtype=torch.float32)

    mask = torch.sparse_coo_tensor(i, v, (n_users, n_items)).coalesce().to(device)
    return mask  # sparse (n_users, n_items)


def build_test_lookup(test_df: pd.DataFrame) -> dict:
    """
    {user_idx: set(item_indices)} pour le test.
    """
    return {
        int(uid): set(items)
        for uid, items in test_df.groupby("user_id_idx")["item_id_idx"].apply(list).items()
    }

# Dataloader
def data_loader_fast(
        user_items_dict: dict,
        all_users: list,          # précalculé dans train_and_eval, pas recalculé ici
        n_users: int,
        n_items: int,
        batch_size: int,
        device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batch sampling BPR ultra-rapide.

    Optimisations pour accélérer l'entrainement :
      - Utilise user_items_dict précalculé (pas de groupby à chaque appel)
      - all_users précalculé une fois dans train_and_eval
      - sample_neg utilise un set Python
      - Pas de Pandas merge → numpy pur
    """
    if n_users >= batch_size:
        batch_users = random.sample(all_users, batch_size)
    else:
        batch_users = [random.choice(all_users) for _ in range(batch_size)]

    pos_items = []
    neg_items = []

    for uid in batch_users:
        seen = user_items_dict[uid]

        # Positif : un item vu au hasard
        pos_items.append(random.choice(list(seen)))

        # Négatif : un item jamais vu
        neg = random.randint(0, n_items - 1)
        while neg in seen:
            neg = random.randint(0, n_items - 1)
        neg_items.append(neg)

    return (
        torch.LongTensor(batch_users).to(device),
        torch.LongTensor(pos_items).to(device) + n_users,
        torch.LongTensor(neg_items).to(device) + n_users,
    )

#BPR Loss
def bpr_loss(
    users: torch.Tensor,
    users_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    user_emb0: torch.Tensor,
    pos_emb0: torch.Tensor,
    neg_emb0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    reg_loss = (1 / 2) * (
        user_emb0.norm().pow(2) +
        pos_emb0.norm().pow(2)  +
        neg_emb0.norm().pow(2)
    ) / float(len(users))

    pos_scores = torch.mul(users_emb, pos_emb).sum(dim=1)
    neg_scores = torch.mul(users_emb, neg_emb).sum(dim=1)
    bpr_loss   = torch.mean(F.softplus(neg_scores - pos_scores))

    return bpr_loss, reg_loss

# Métriques
def get_metrics(
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        n_users: int,
        n_items: int,
        train_mask: torch.Tensor,  # précalculé dans train()
        test_lookup: dict,  # précalculé dans train()
        k: int,
) -> tuple[float, float]:
    # Scores (n_users, n_items)
    relevance = torch.matmul(user_emb, item_emb.T)
    # Masquer les interactions du train
    relevance = relevance - train_mask.to_dense() * 1e9

    # Top-K indices (sur GPU)
    topk_indices = torch.topk(relevance, k).indices.cpu().numpy()  # (n_users, k)

    recalls, precisions = [], []
    for uid, true_items in test_lookup.items():
        if uid >= n_users:
            continue
        recs = set(topk_indices[uid].tolist())
        hits = len(recs.intersection(true_items))
        recalls.append(hits / max(len(true_items), 1))
        precisions.append(hits / k)

    if not recalls:
        return 0.0, 0.0
    return float(np.mean(recalls)), float(np.mean(precisions))


# Entraînement
def train_and_eval(
        model,
        optimizer: torch.optim.Optimizer,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_edge_index: torch.Tensor,
        n_users: int,
        n_items: int,
        device: torch.device,
        epochs: int = LGCN_EPOCHS,
        batch_size: int = LGCN_BATCH_SIZE,
        decay: float = LGCN_DECAY,
        k: int = K,
        eval_every: int = EVAL_EVERY,
) -> dict:
    # Précalculs (une seule fois)
    print("Précalcul des structures d'accélération...")
    user_items_dict = build_user_items_dict(train_df)
    all_users       = list(user_items_dict.keys())   # précalculé ici, pas dans chaque batch
    train_mask = build_train_mask(train_df, n_users, n_items, device)
    test_lookup = build_test_lookup(test_df)
    n_batch = max(1, len(train_df) // batch_size)
    print(f"{n_users} users | {n_items} livres | "
          f"{n_batch} batches/époque | éval tous les {eval_every} époques")

    history = {key: [] for key in ["loss", "bpr_loss", "reg_loss", "recall", "precision"]}

    for epoch in tqdm(range(epochs), desc="Training HybridLightGCN"):
        ep_loss, ep_bpr, ep_reg = [], [], []

        model.train()
        for _ in range(n_batch):
            optimizer.zero_grad()

            users, pos_items, neg_items = data_loader_fast(
                user_items_dict, all_users, n_users, n_items, batch_size, device
            )
            u_emb, pos_emb, neg_emb, u_emb0, pos_emb0, neg_emb0 = \
                model.encode_minibatch(users, pos_items, neg_items, train_edge_index)

            loss_bpr, loss_reg = bpr_loss(
                users, u_emb, pos_emb, neg_emb, u_emb0, pos_emb0, neg_emb0
            )
            loss = loss_bpr + decay * loss_reg
            loss.backward()
            optimizer.step()

            ep_loss.append(loss.item())
            ep_bpr.append(loss_bpr.item())
            ep_reg.append(loss_reg.item())

        history["loss"].append(round(float(np.mean(ep_loss)), 4))
        history["bpr_loss"].append(round(float(np.mean(ep_bpr)), 4))
        history["reg_loss"].append(round(float(np.mean(ep_reg)), 4))

        # Évaluation tous les eval_every époques seulement
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                _, out = model(train_edge_index)
                user_emb, item_emb = torch.split(out, (n_users, n_items))
                recall, precision = get_metrics(
                    user_emb, item_emb, n_users, n_items,
                    train_mask, test_lookup, k
                )
            history["recall"].append(round(recall, 4))
            history["precision"].append(round(precision, 4))
            tqdm.write(f"  Époque {epoch + 1:>3} | loss {history['loss'][-1]:.4f} "
                       f"| Recall@{k} {recall:.4f} | Precision@{k} {precision:.4f}")
        else:
            history["recall"].append(-1)
            history["precision"].append(-1)

    return history


# Courbes d'apprentissage
def plot_training(history: dict, save_dir: str = CACHE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    epochs = list(range(1, len(history["loss"]) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    axes[0].plot(epochs, history["bpr_loss"], label="BPR Loss", color="tab:blue")
    axes[0].plot(epochs, history["loss"], label="Loss totale", color="tab:green", linestyle="--")
    axes[0].set(xlabel="Époque", ylabel="Loss", title="BPR Loss — HybridLightGCN")
    axes[0].legend()

    axes[1].plot(epochs, history["reg_loss"], label="Reg Loss", color="tab:orange")
    axes[1].set(xlabel="Époque", ylabel="Loss", title="Reg Loss — HybridLightGCN")
    axes[1].legend()

    eval_eps = [e for e, r in zip(epochs, history["recall"]) if r >= 0]
    recalls = [r for r in history["recall"] if r >= 0]
    precisions = [p for p in history["precision"] if p >= 0]
    axes[2].plot(eval_eps, recalls, marker="o", label=f"Recall@{K}", color="tab:blue")
    axes[2].plot(eval_eps, precisions, marker="s", label=f"Precision@{K}", color="tab:orange")
    axes[2].set(xlabel="Époque", ylabel="Métrique", title=f"Métriques test — HybridLightGCN")
    axes[2].set_ylim(bottom=0)
    axes[2].legend()

    plt.tight_layout()
    out = os.path.join(save_dir, "training_curves.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Courbes sauvegardées → {out}")

