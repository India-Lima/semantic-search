"""Streamlit-демо поиска товаров: гибрид TF-IDF + e5-small (RRF).

Запуск: streamlit run app.py
Артефакты создаются ячейкой сохранения в semantic_search_project.ipynb.
"""
import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_DIR = Path("models/e5-small")
CATALOG_PATH = Path("data/catalog_app.parquet")
EMB_PATH = Path("data/emb_app.npy")

# Параметры финальной конфигурации hybrid_rrf_w08_labeled (см. ноутбук)
TFIDF_PARAMS = {"ngram_range": (1, 2), "max_features": 100_000, "sublinear_tf": True}
QUERY_PREFIX = "query: "
RRF_K, N_CANDIDATES = 60, 100
MODES = {"Гибрид (финальный)": 0.8, "Семантический": 1.0, "Лексический": 0.0}
SPECIAL_RE = re.compile(r'[^\w\s.,:;!?()\-–—«»"\'/%+№*°]')


def clean_text(text: str) -> str:
    """Та же очистка, что для каталога в ноутбуке."""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[“”„]", '"', text)
    text = SPECIAL_RE.sub(" ", text)
    text = re.sub(r"([!?.,*_=~-])\1+", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_resource
def load_model() -> SentenceTransformer:
    return SentenceTransformer(str(MODEL_DIR), device="cpu")


@st.cache_data
def load_data() -> tuple[pd.DataFrame, np.ndarray]:
    return pd.read_parquet(CATALOG_PATH), np.load(EMB_PATH)


@st.cache_resource
def build_tfidf(texts: tuple[str, ...]):
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    return vectorizer, vectorizer.fit_transform(texts)


def search(query: str, top_k: int, w_sem: float) -> pd.DataFrame:
    """RRF по рангам лексического и семантического поиска."""
    catalog, emb = load_data()
    vectorizer, tfidf = build_tfidf(tuple(catalog["product_text"]))
    query = clean_text(query)

    lex = (tfidf @ vectorizer.transform([query]).T).toarray().ravel()
    sem = emb @ load_model().encode(QUERY_PREFIX + query, normalize_embeddings=True)

    scores = {}
    for weight, engine_scores in ((1 - w_sem, lex), (w_sem, sem)):
        for rank, idx in enumerate(np.argsort(-engine_scores)[:N_CANDIDATES], 1):
            scores[idx] = scores.get(idx, 0) + weight / (RRF_K + rank)
    top = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return catalog.iloc[top].assign(similarity=sem[top])


st.set_page_config(page_title="Поиск одежды и обуви", page_icon="🔎")
st.title("🔎 Семантический поиск: одежда и обувь")
st.caption("Гибрид TF-IDF + multilingual-e5-small (RRF), каталог 48 376 карточек")

if not all(p.exists() for p in (MODEL_DIR, CATALOG_PATH, EMB_PATH)):
    st.error("Нет артефактов. Выполните ноутбук semantic_search_project.ipynb (см. README).")
    st.stop()

query = st.text_input("Запрос", placeholder="например: удобные кроссовки для бега")
col_mode, col_k = st.columns([2, 1])
mode = col_mode.radio("Метод", list(MODES), horizontal=True)
top_k = col_k.slider("Сколько товаров", 5, 30, 10)

if query.strip():
    for i, row in enumerate(search(query, top_k, MODES[mode]).itertuples(), 1):
        st.markdown(f"**{i}. {row.imt_name or row.subj_name}**")
        st.caption(
            f"{row.subj_name} · {row.brand_name or '—'} · {row.colors or '—'} · "
            f"imt_id {row.imt_id} · сходство e5: {row.similarity:.3f}"
        )
        if row.description:
            st.write(row.description[:200] + ("…" if len(row.description) > 200 else ""))
        st.divider()
