import torch
import torch.nn as nn
from torch_geometric.nn import LGConv
from config import LGCN_LATENT_DIM,  LGCN_N_LAYERS

class HybridLightGCN(nn.Module):

    def __init__(
        self,
        num_users: int,
        num_items: int,
        book_bert_embeddings: torch.Tensor,   # (n_items, sbert_dim)
        latent_dim: int = LGCN_LATENT_DIM,
        num_layers: int =  LGCN_N_LAYERS,
    ):
        super().__init__()
        self.num_users  = num_users
        self.num_items  = num_items
        self.num_layers = num_layers
        sbert_dim = book_bert_embeddings.shape[1]

        # Embeddings utilisateurs (appris)
        self.user_embedding = nn.Embedding(num_users, latent_dim) # aléatoire (appris)
        nn.init.normal_(self.user_embedding.weight, std=0.1)

        # Embeddings livres : BERT gelé → projection linéaire (apprise) → latent_dim
        # register_buffer → sauvegardé avec le modèle mais PAS mis à jour avec optiiseur
        self.register_buffer("book_bert", book_bert_embeddings)
        self.book_proj = nn.Linear(sbert_dim, latent_dim, bias=False)
        nn.init.xavier_uniform_(self.book_proj.weight)

        # Couches LightGCN
        self.convs = nn.ModuleList(LGConv() for _ in range(num_layers))

    def get_initial_embeddings(self) -> torch.Tensor:
        user_emb = self.user_embedding.weight       # (n_users, D)
        book_emb = self.book_proj(self.book_bert)   # (n_items, D)
        return torch.cat([user_emb, book_emb], dim=0) # moyenne de toutes les couches

    def forward(self, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb0 = self.get_initial_embeddings()
        embs = [emb0]
        x = emb0
        for conv in self.convs:
            x = conv(x=x, edge_index=edge_index)
            embs.append(x)
        out = torch.mean(torch.stack(embs, dim=0), dim=0)
        return emb0, out

    def encode_minibatch(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple:
        emb0, out = self(edge_index)
        return (
            out[users],
            out[pos_items],
            out[neg_items],
            emb0[users],
            emb0[pos_items],
            emb0[neg_items],
        )