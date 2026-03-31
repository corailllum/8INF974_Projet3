import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# CONFIG

FILE_PATH = "GoodReads_100k_books.csv"

st.set_page_config(page_title="📚 Dashboard Livres", layout="wide")
st.title("📚 Dashboard - Analyse des livres GoodReads")


# LOAD DATA

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(FILE_PATH, encoding='utf-8-sig', on_bad_lines='skip')
    except Exception:
        df = pd.read_csv(FILE_PATH, encoding='latin-1', on_bad_lines='skip')
    df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=False)
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Erreur chargement fichier: {e}")
    st.stop()


# CLEANING

if 'pages' in df.columns:
    df['pages'] = pd.to_numeric(df['pages'], errors='coerce')
    df['pages'] = df['pages'].replace(0, float('nan'))

if 'rating' in df.columns:
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df.loc[~df['rating'].between(0, 5), 'rating'] = float('nan')

for col in ['reviews', 'totalratings']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

if 'genre' in df.columns:
    df['genre_primary'] = (
        df['genre'].astype(str)
        .str.split(',').str[0].str.strip()
        .replace('nan', float('nan'))
    )


# APERÇU & DIAGNOSTICS

with st.expander(" Aperçu des données brutes"):
    st.dataframe(df.head(10))

with st.expander(" Diagnostics"):
    st.write("**Colonnes :**", df.columns.tolist())
    st.write("**Lignes :**", len(df))
    st.dataframe(df[['rating', 'pages', 'reviews', 'totalratings']].describe())


# KPIs

st.subheader(" Résumé global")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Livres total", f"{len(df):,}")
with c2:
    avg = df['rating'].mean() if 'rating' in df.columns else None
    st.metric("Note moyenne", f"{avg:.2f} ⭐" if avg else "N/A")
with c3:
    med = df['pages'].median() if 'pages' in df.columns else None
    st.metric("Pages médianes", f"{int(med):,}" if med and pd.notna(med) else "N/A")
with c4:
    st.metric("Auteurs uniques", f"{df['author'].nunique():,}" if 'author' in df.columns else "N/A")

st.divider()


# HELPER : histogramme manuel (compatible toutes versions Plotly)
# Calcule les bins avec numpy et passe des barres simples → pas de bdata

import numpy as np

def make_histogram(values, nbins, color, title, xlabel):
    """Crée un histogramme via go.Bar (compatible Plotly v1/v2/v3)."""
    counts, edges = np.histogram(values, bins=nbins)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    # Convertit en listes Python natives — pas de numpy array = pas de bdata
    fig = go.Figure(go.Bar(
        x=bin_centers.tolist(),
        y=counts.tolist(),
        width=(edges[1] - edges[0]).item() * 0.9,
        marker_color=color,
        opacity=0.85,
    ))
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title="Nombre de livres",
        plot_bgcolor='white',
        paper_bgcolor='white',
        bargap=0.02,
    )
    fig.update_xaxes(showgrid=True, gridcolor='#eeeeee')
    fig.update_yaxes(showgrid=True, gridcolor='#eeeeee')
    return fig


# 1. Distribution des ratings

st.subheader(" Distribution des notes")
if 'rating' in df.columns:
    vals = df['rating'].dropna()
    vals = vals[vals.between(0, 5)].to_numpy().astype(float)
    st.caption(f"Livres avec note valide : {len(vals):,} | Min: {vals.min():.2f} | Max: {vals.max():.2f} | Moy: {vals.mean():.2f}")
    fig = make_histogram(vals, nbins=40, color='royalblue',
                         title="Distribution des notes (0 à 5)", xlabel="Note")
    fig.update_xaxes(range=[0, 5])
    st.plotly_chart(fig, use_container_width=True)


# 2. Distribution des pages

st.subheader(" Distribution du nombre de pages")
if 'pages' in df.columns:
    vals = df['pages'].dropna()
    vals = vals[(vals > 0) & (vals <= 2000)].to_numpy().astype(float)
    st.caption(f"Livres avec pages valides (1–2000) : {len(vals):,}")
    fig = make_histogram(vals, nbins=60, color='mediumseagreen',
                         title="Distribution du nombre de pages", xlabel="Pages")
    st.plotly_chart(fig, use_container_width=True)


# 3. Top 10 auteurs

st.subheader(" Top 10 auteurs")
if 'author' in df.columns:
    top = df['author'].dropna().value_counts().head(10)
    fig = go.Figure(go.Bar(
        x=top.values.tolist(),
        y=top.index.tolist(),
        orientation='h',
        marker_color='coral',
    ))
    fig.update_layout(
        title="Top 10 auteurs avec le plus de livres",
        xaxis_title="Nombre de livres",
        yaxis={'categoryorder': 'total ascending'},
        plot_bgcolor='white', paper_bgcolor='white',
    )
    st.plotly_chart(fig, use_container_width=True)


# 4. Top 15 genres

st.subheader(" Top 15 genres")
if 'genre_primary' in df.columns:
    top = df['genre_primary'].dropna().value_counts().head(15)
    fig = go.Figure(go.Bar(
        x=top.values.tolist(),
        y=top.index.tolist(),
        orientation='h',
        marker_color='mediumpurple',
    ))
    fig.update_layout(
        title="Top 15 genres les plus fréquents",
        xaxis_title="Nombre de livres",
        yaxis={'categoryorder': 'total ascending'},
        plot_bgcolor='white', paper_bgcolor='white',
    )
    st.plotly_chart(fig, use_container_width=True)


# 5. Heatmap de corrélation classique

st.subheader(" Heatmap de corrélation")
num_cols = [c for c in ['rating', 'pages', 'reviews', 'totalratings'] if c in df.columns]
if len(num_cols) >= 2:
    sub = df[num_cols].copy()
    if 'pages' in sub.columns:
        sub.loc[sub['pages'] == 0, 'pages'] = float('nan')
    sub = sub.dropna()
    st.caption(f"Lignes utilisées pour la heatmap : {len(sub):,}")

    if len(sub) > 10:
        corr = sub.corr().round(2)
        labels = corr.columns.tolist()
        z = corr.values.tolist()                        # liste Python native
        text = [[str(v) for v in row] for row in z]

        fig = go.Figure(go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 16},
            colorscale=[
                [0.0,  "#d73027"],
                [0.25, "#f46d43"],
                [0.5,  "#ffffbf"],
                [0.75, "#74add1"],
                [1.0,  "#4575b4"],
            ],
            zmin=-1, zmax=1,
            showscale=True,
            colorbar=dict(title="r", tickvals=[-1, -0.5, 0, 0.5, 1]),
        ))
        fig.update_layout(
            title="Heatmap de corrélation entre variables numériques",
            plot_bgcolor='white', paper_bgcolor='white',
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"Pas assez de données ({len(sub)} lignes).")


# 6. Scatter Rating vs Reviews

st.subheader(" Rating vs Nombre de reviews")
if 'reviews' in df.columns and 'rating' in df.columns:
    sub = df[df['reviews'].notna() & df['rating'].notna() & df['rating'].between(0, 5)].copy()
    q99 = float(sub['reviews'].quantile(0.99))
    sub = sub[sub['reviews'] <= q99]
    st.caption(f"Livres affichés (reviews ≤ {int(q99):,}) : {len(sub):,}")

    # Convertit en listes Python natives → évite bdata
    x_vals = sub['reviews'].astype(float).tolist()
    y_vals = sub['rating'].astype(float).tolist()

    fig = go.Figure(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='markers',
        marker=dict(color='steelblue', opacity=0.2, size=3),
    ))
    fig.update_layout(
        title="Relation entre nombre de reviews et note",
        xaxis_title="Nombre de reviews",
        yaxis_title="Note",
        yaxis=dict(range=[0, 5]),
        plot_bgcolor='white', paper_bgcolor='white',
    )
    fig.update_xaxes(showgrid=True, gridcolor='#eeeeee')
    fig.update_yaxes(showgrid=True, gridcolor='#eeeeee')
    st.plotly_chart(fig, use_container_width=True)


# FIN

st.divider()
st.success(" Dashboard chargé correctement")
st.caption(" Lancer avec : streamlit run dashboard_books.py")