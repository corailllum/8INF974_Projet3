
# Projet 3 : Système de recommandation de livres

## Structure du projet (avril 2026)

- **gnn_for_book_recommender_v2.ipynb**  
	Notebook d’expérimentation pour la recommandation de livres avec GNN.
- **test_reco_simi.py**  
	Script de test pour le système de recommandation BERT+KNN.
- **visu.py**  
	Dashboard Streamlit pour visualiser/analyser le dataset Goodreads.
- **visufilm.py**  
	Dashboard Streamlit pour visualiser le dataset Book Crossing.

### Dossier `Book Crossing Dataset/`
- BX-Book-Ratings.csv, BX-Books.csv, BX-Users.csv : fichiers du dataset Book Crossing (notes, livres, utilisateurs).

### Dossier `book_recommender/`
- **Bert-KNN method/**
	- Recommentation_Similarite_GoodReads.py : Script de test pour la méthode BERT+KNN.
- **Hybrid method/**
	- bert_encoder.py : Encodage des livres avec BERT.
	- config.py : Fichier de configuration (chemins, hyperparamètres).
	- lightgcn_trainer.py : Entraînement du modèle LightGCN.
	- model.py : Définition du modèle hybride LightGCN.
	- pipeline_hybrid.py : Pipeline complet pour la méthode hybride.
	- preprocessing.py : Prétraitement des données pour les modèles.
	- recommender.py : Mesure des scores et recommandations.
- **LightGCN method/**
	- gnn_for_book_recommender.ipynb : Notebook pour la méthode LightGCN seule.
- **data/Book Crossing Dataset/**
	- BX-Book-Ratings.csv, BX-Books.csv, BX-Users.csv : copie locale des fichiers du dataset Book Crossing.

---

## Mode d'emploi

### 1. Installation des dépendances

Installer les librairies nécessaires (exemple pour l’approche hybride) :
```bash
pip install pandas scikit-learn sentence-transformers torch tqdm numpy streamlit plotly
```

### 2. Lancer les notebooks

Ouvre le notebook dans Jupyter ou VS Code :
- `book_recommender/LightGCN method/gnn_for_book_recommender_v4.ipynb` (LightGCN seul)

### 3. Lancer les dashboards

Pour visualiser les données :
```bash
streamlit run visu.py
streamlit run visufilm.py
```

### 4. Tester la recommandation BERT+KNN

Exécuter le script de test :
```bash
python book_recommender/Bert-KNN method/Recommentation_Similarite_GoodReads.py
```

### 5. Tester la recommandation Hybride : BERT+ LightGCN

Exécuter le script de test :
```bash
python book_recommender/Hybrid method/pipeline_hybrid.py
```


### 6. Organisation des données

Les datasets doivent être placés à la racine ou dans les dossiers indiqués dans les scripts/configs.
Adapter les chemins dans `config.py` si besoin.

---
