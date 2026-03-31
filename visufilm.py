import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go


# CONFIG

FILE_PATH = "Movies.csv"

st.set_page_config(page_title=" Dashboard Films", layout="wide")
st.title("🎬 Dashboard - Analyse des films")


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

for col in ['vote_average', 'vote_count', 'revenue', 'budget', 'popularity']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

if 'vote_average' in df.columns:
    df.loc[~df['vote_average'].between(0, 10), 'vote_average'] = float('nan')

for col in ['budget', 'revenue']:
    if col in df.columns:
        df[col] = df[col].replace(0, float('nan'))

if 'release_date' in df.columns:
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year

if 'genres' in df.columns:
    df['genre_primary'] = (
        df['genres'].astype(str)
        .str.split(',').str[0].str.strip()
        .replace('nan', float('nan'))
    )

if 'revenue' in df.columns and 'budget' in df.columns:
    df['profit'] = df['revenue'] - df['budget']

# Notes arrondies à l'entier
if 'vote_average' in df.columns:
    df['vote_int'] = df['vote_average'].round().astype('Int64')

# Tranches de budget
if 'budget' in df.columns:
    bins   = [0, 10e6, 50e6, 100e6, 200e6, np.inf]
    labels = ['< 10M$', '10–50M$', '50–100M$', '100–200M$', '> 200M$']
    df['budget_range'] = pd.cut(df['budget'], bins=bins, labels=labels, right=True)


# APERÇU & DIAGNOSTICS

with st.expander(" Aperçu des données"):
    st.dataframe(df.head(10))


# KPIs

st.subheader(" Résumé global")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Films total", f"{len(df):,}")
with c2:
    avg = df['vote_average'].mean() if 'vote_average' in df.columns else None
    st.metric("Note moyenne", f"{avg:.2f} / 10" if avg and pd.notna(avg) else "N/A")
with c3:
    rev = df['revenue'].median() if 'revenue' in df.columns else None
    st.metric("Revenu médian", f"${rev/1e6:.1f}M" if rev and pd.notna(rev) else "N/A")
with c4:
    if 'release_year' in df.columns:
        st.metric("Période", f"{int(df['release_year'].min())} – {int(df['release_year'].max())}")

st.divider()


# HELPER

LAYOUT = dict(plot_bgcolor='white', paper_bgcolor='white')
GRID   = dict(showgrid=True, gridcolor='#eeeeee')

def bar_chart(x, y, color, title, xlabel, ylabel="Nombre de films",
              orientation='v', text=None):
    fig = go.Figure(go.Bar(
        x=x, y=y, orientation=orientation,
        marker_color=color, opacity=0.88,
        text=text, textposition='outside' if text else None,
    ))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, **LAYOUT)
    fig.update_xaxes(**GRID)
    fig.update_yaxes(**GRID)
    return fig

def make_histogram(values, nbins, color, title, xlabel):
    counts, edges = np.histogram(values, bins=nbins)
    centers = ((edges[:-1] + edges[1:]) / 2).tolist()
    width   = float(edges[1] - edges[0]) * 0.9
    fig = go.Figure(go.Bar(
        x=centers, y=counts.tolist(),
        width=width, marker_color=color, opacity=0.85,
    ))
    fig.update_layout(title=title, xaxis_title=xlabel,
                      yaxis_title="Nombre de films", **LAYOUT, bargap=0.02)
    fig.update_xaxes(**GRID)
    fig.update_yaxes(**GRID)
    return fig


# 1. Notes agrégées en entiers

st.subheader(" Distribution des notes (entiers)")
if 'vote_int' in df.columns:
    counts = df['vote_int'].dropna().astype(int).value_counts().sort_index()
    fig = bar_chart(
        x=counts.index.tolist(), y=counts.values.tolist(),
        color='royalblue', title="Films par note arrondie",
        xlabel="Note (0–10)", text=counts.values.tolist(),
    )
    fig.update_xaxes(tickmode='linear', tick0=0, dtick=1, range=[-0.5, 10.5], showgrid=False)
    st.plotly_chart(fig, use_container_width=True)


# 2. Tranches de budget — nombre de films

st.subheader("💰 Répartition par tranche de budget")
if 'budget_range' in df.columns:
    order  = ['< 10M$', '10–50M$', '50–100M$', '100–200M$', '> 200M$']
    colors = ['#4575b4', '#74add1', '#ffffbf', '#f46d43', '#d73027']
    counts = df['budget_range'].value_counts().reindex(order).fillna(0)

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure(go.Bar(
            x=counts.index.tolist(), y=counts.values.tolist(),
            marker_color=colors, opacity=0.9,
            text=counts.values.astype(int).tolist(), textposition='outside',
        ))
        fig.update_layout(title="Nombre de films", xaxis_title="Tranche",
                          yaxis_title="Films", **LAYOUT)
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(**GRID)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if 'vote_average' in df.columns:
            avg_note = (
                df.dropna(subset=['budget_range', 'vote_average'])
                .groupby('budget_range', observed=True)['vote_average']
                .mean().reindex(order).round(2)
            )
            fig2 = go.Figure(go.Bar(
                x=avg_note.index.tolist(), y=avg_note.values.tolist(),
                marker_color=colors, opacity=0.9,
                text=[f"{v:.2f}" for v in avg_note.values], textposition='outside',
            ))
            fig2.update_layout(title="Note moyenne", xaxis_title="Tranche",
                               yaxis_title="Note moy.", yaxis=dict(range=[0, 10]), **LAYOUT)
            fig2.update_xaxes(showgrid=False)
            fig2.update_yaxes(**GRID)
            st.plotly_chart(fig2, use_container_width=True)


# 3. Revenus

st.subheader(" Distribution des revenus")
if 'revenue' in df.columns:
    vals = df['revenue'].dropna()
    vals = vals[(vals > 0) & (vals <= 3e9)].to_numpy().astype(float)
    st.caption(f"{len(vals):,} films avec revenu valide")
    st.plotly_chart(
        make_histogram(vals / 1e6, 50, 'mediumseagreen',
                       "Distribution des revenus", "Revenu (M$)"),
        use_container_width=True
    )


# 4. Films par année

st.subheader(" Films par année")
if 'release_year' in df.columns:
    yc = df['release_year'].dropna().astype(int).value_counts().sort_index()
    fig = bar_chart(yc.index.tolist(), yc.values.tolist(), 'steelblue',
                    "Films par année de sortie", "Année")
    st.plotly_chart(fig, use_container_width=True)


# 5. Top 15 genres + note moyenne

st.subheader("🎭 Genres")
if 'genre_primary' in df.columns:
    col1, col2 = st.columns(2)

    with col1:
        top = df['genre_primary'].dropna().value_counts().head(15)
        fig = bar_chart(
            x=top.values.tolist(), y=top.index.tolist(),
            color='mediumpurple', title="Top 15 genres",
            xlabel="Films", ylabel="Genre", orientation='h',
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if 'vote_average' in df.columns:
            gr = (
                df.dropna(subset=['genre_primary', 'vote_average'])
                .groupby('genre_primary')['vote_average']
                .agg(['mean', 'count']).reset_index()
            )
            gr = gr[gr['count'] >= 10].sort_values('mean', ascending=True)
            fig2 = bar_chart(
                x=gr['mean'].round(2).tolist(), y=gr['genre_primary'].tolist(),
                color='darkorange', title="Note moy. par genre",
                xlabel="Note", ylabel="Genre", orientation='h',
                text=[f"{v:.2f}" for v in gr['mean']],
            )
            fig2.update_layout(
                xaxis=dict(range=[0, 10]),
                yaxis={'categoryorder': 'total ascending'},
            )
            st.plotly_chart(fig2, use_container_width=True)


# 6. Heatmap — corrélation pairwise (pas de dropna global)

st.subheader(" Heatmap de corrélation")
num_cols = [c for c in ['vote_average', 'vote_count', 'budget', 'revenue', 'popularity'] if c in df.columns]
if len(num_cols) >= 2:
    # Corrélation pairwise : chaque paire utilise ses propres lignes valides
    n = len(num_cols)
    corr_matrix = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                pair = df[[num_cols[i], num_cols[j]]].dropna()
                if len(pair) > 10:
                    corr_matrix[i, j] = round(pair.corr().iloc[0, 1], 2)
                else:
                    corr_matrix[i, j] = 0.0

    z    = corr_matrix.tolist()
    text = [[str(round(v, 2)) for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z, x=num_cols, y=num_cols,
        text=text, texttemplate="%{text}",
        textfont={"size": 14},
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
        title="Corrélation entre variables (calcul pairwise)",
        height=420, **LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)


# 7. Scatter Budget vs Revenu — échantillonné à 3000 pts

st.subheader("💸 Budget vs Revenu")
if 'budget' in df.columns and 'revenue' in df.columns:
    sub = df[
        df['budget'].notna() & df['revenue'].notna() &
        (df['budget'] > 0) & (df['revenue'] > 0)
    ].copy()
    sub = sub[sub['budget']  <= sub['budget'].quantile(0.99)]
    sub = sub[sub['revenue'] <= sub['revenue'].quantile(0.99)]

    # Échantillon pour la perf
    if len(sub) > 3000:
        sub = sub.sample(3000, random_state=42)

    st.caption(f"{len(sub):,} films affichés")
    colors = ['mediumseagreen' if p > 0 else 'tomato'
              for p in (sub['revenue'] - sub['budget']).tolist()]
    fig = go.Figure(go.Scatter(
        x=(sub['budget'] / 1e6).astype(float).tolist(),
        y=(sub['revenue'] / 1e6).astype(float).tolist(),
        mode='markers',
        marker=dict(color=colors, opacity=0.55, size=5),
        text=sub['title'].tolist() if 'title' in sub.columns else None,
        hovertemplate="<b>%{text}</b><br>Budget: $%{x:.0f}M<br>Revenu: $%{y:.0f}M<extra></extra>",
    ))
    max_val = float(max(sub['budget'].max(), sub['revenue'].max()) / 1e6)
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val], mode='lines',
        line=dict(color='grey', dash='dash', width=1),
        name='Breakeven', hoverinfo='skip',
    ))
    fig.update_layout(
        title="Budget vs Revenu (vert = profit · rouge = perte)",
        xaxis_title="Budget (M$)", yaxis_title="Revenu (M$)",
        showlegend=True, **LAYOUT,
    )
    fig.update_xaxes(**GRID)
    fig.update_yaxes(**GRID)
    st.plotly_chart(fig, use_container_width=True)


# FIN

st.divider()
st.success(" Dashboard chargé correctement")
st.caption(" streamlit run dashboard_movies.py")