from pathlib import Path

import torch

# Chemins ───────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
DATA_DIR        = BASE_DIR / "data" / "Amazon Books Reviews"
RATINGS_FILE    = DATA_DIR / "Books_rating.csv"
BOOKS_FILE  = DATA_DIR / "books_data.csv"

# Fichiers de sauvegarde ───────────────────────────────────────────────────
CACHE_DIR               = Path(__file__).resolve().parent / "cache"
CACHE_FILE              = CACHE_DIR / "preprocessed_cache.parquet"
BERT_EMBEDDINGS_FILE    = CACHE_DIR / "book_bert_embeddings.npy"
BOOK_TITLES_FILE        =  CACHE_DIR /"book_titles_ordered.npy"
KNN_FILE                = CACHE_DIR / "knn_model.joblib"
GNN_CHECKPOINT          = CACHE_DIR / "lightgcn_checkpoint.pt"

# Preprocessing ────────────────────────────────────────────────────────────
MIN_USER_INTERACTIONS       = 20
MIN_BOOK_RATINGS            = 20
MIN_DESCRIPTION_LEN         = 50
POSITIVE_RATING_THRESHOLD   = 4
TEST_SIZE                   = 0.2
RANDOM_STATE                = 42

# SBERT ────────────────────────────────────────────────────────────────────
SBERT_MODEL_NAME = "all-MiniLM-L6-v2" # dim=384
SBERT_BATCH_SIZE = 64
SBERT_DIM        = 384
SAMPLE_SIZE      = 20000      # nombre de livres uniques à utiliser (je les utilise pas tous sinon trop lourd)
TOP_K = 5

# LightGCN ──────────────────────────────────────────────────────────────────
LGCN_LATENT_DIM         = 64
LGCN_N_LAYERS           = 3
LGCN_EPOCHS             = 50
LGCN_BATCH_SIZE         = 4096
LGCN_N_BATCH_PER_EPOCH  = 50
LGCN_LR                 = 5e-3
LGCN_DECAY              = 1e-6


# Evaluation ─────────────────────────────────────────────────────────────────
K           = 20
ALPHA       = 0.5    # poids BERT dans la fusion (0=GNN pur, 1=BERT pur)
ALPHA_GRID  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
