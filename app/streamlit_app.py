from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

try:
    import shap
except Exception:  # pragma: no cover - app fallback
    shap = None

try:
    from qdrant_client import QdrantClient
except Exception:  # pragma: no cover - app fallback
    QdrantClient = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "xgboost_optuna_tuned.ubj"
PREPROCESSOR_PATH = MODEL_DIR / "xgboost_optuna_tuned_preprocessor.pkl"
FEATURES_PATH = MODEL_DIR / "xgboost_optuna_tuned_features.json"
METRICS_PATH = MODEL_DIR / "xgboost_optuna_tuned_metrics.json"

TRAIN_TRANSACTION_PATH = DATA_DIR / "train_transaction.csv"
TRAIN_IDENTITY_PATH = DATA_DIR / "train_identity.csv"
QDRANT_COLLECTION = "fraud_transactions_shap"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DEFAULT_THRESHOLD = 0.25
SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 24 * SECONDS_PER_HOUR
SECONDS_PER_WEEK = 7 * SECONDS_PER_DAY

st.set_page_config(page_title="PayTrack Fraud Analyst", layout="wide")


@st.cache_resource(show_spinner=False)
def load_model_assets() -> tuple[xgb.XGBClassifier, Any, list[str], dict[str, Any]]:
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)

    feature_cols = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    return model, preprocessor, feature_cols, metrics


@st.cache_data(show_spinner="Chargement des transactions de demonstration...")
def load_demo_transactions(max_rows: int = 30_000) -> pd.DataFrame:
    transaction_cols = [
        "TransactionID", "TransactionDT", "TransactionAmt", "ProductCD", "card1", "card2", "card3", "card4",
        "card5", "card6", "addr1", "addr2", "dist1", "dist2", "P_emaildomain", "R_emaildomain",
        "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14",
        "D1", "D2", "D3", "D4", "D5", "D10", "D15", "isFraud",
    ]
    identity_cols = [
        "TransactionID", "DeviceType", "id_12", "id_15", "id_16", "id_28", "id_29", "id_35", "id_36", "id_37", "id_38",
    ]

    tx = pd.read_csv(TRAIN_TRANSACTION_PATH, usecols=lambda c: c in transaction_cols, nrows=max_rows)
    ident = pd.read_csv(TRAIN_IDENTITY_PATH, usecols=lambda c: c in identity_cols)
    df = tx.merge(ident, on="TransactionID", how="left").sort_values("TransactionDT").reset_index(drop=True)

    df["hour_rel"] = (df["TransactionDT"] // SECONDS_PER_HOUR).astype("int64")
    df["day_rel"] = (df["TransactionDT"] // SECONDS_PER_DAY).astype("int64")
    df["week_rel"] = (df["TransactionDT"] // SECONDS_PER_WEEK).astype("int64")
    df["hour_of_day_rel"] = ((df["TransactionDT"] // SECONDS_PER_HOUR) % 24).astype("int64")

    id_cols = [c for c in ["card1", "card2", "card3", "card5", "addr1"] if c in df.columns]
    merchant_cols = [c for c in ["ProductCD", "R_emaildomain"] if c in df.columns]
    df["customer_proxy"] = df[id_cols].astype("string").fillna("missing").agg("_".join, axis=1)
    df["merchant_proxy"] = df[merchant_cols].astype("string").fillna("missing").agg("_".join, axis=1)

    # Features historiques exactes pour les lignes gardees en demo. Pour une transaction saisie manuellement,
    # ces champs peuvent être modifiés dans l'interface ou laissés à 0 / manquants.
    df = df.sort_values(["customer_proxy", "TransactionDT", "TransactionID"]).copy()
    grp = df.groupby("customer_proxy", sort=False)
    df["customer_tx_count_prev"] = grp.cumcount()
    df["customer_amt_sum_prev"] = grp["TransactionAmt"].cumsum() - df["TransactionAmt"]
    df["customer_amt_mean_prev"] = np.where(
        df["customer_tx_count_prev"] > 0,
        df["customer_amt_sum_prev"] / df["customer_tx_count_prev"],
        np.nan,
    )
    df["amt_to_customer_mean_prev"] = np.where(
        df["customer_amt_mean_prev"] > 0,
        df["TransactionAmt"] / df["customer_amt_mean_prev"],
        np.nan,
    )
    df["merchant_seen_count_prev"] = df.groupby(["customer_proxy", "merchant_proxy"], sort=False).cumcount()
    df["is_new_merchant_for_customer"] = (df["merchant_seen_count_prev"] == 0).astype("int8")

    df = df.sort_values("TransactionDT").reset_index(drop=True)
    for label, seconds in {"1h": SECONDS_PER_HOUR, "24h": SECONDS_PER_DAY, "7d": SECONDS_PER_WEEK}.items():
        counts = np.zeros(len(df), dtype="int64")
        sums = np.zeros(len(df), dtype="float64")
        for _, idx in df.groupby("customer_proxy", sort=False).indices.items():
            idx = np.asarray(idx)
            times = df.loc[idx, "TransactionDT"].to_numpy()
            amounts = df.loc[idx, "TransactionAmt"].fillna(0).to_numpy(dtype="float64")
            cumsum = np.cumsum(amounts)
            left = np.searchsorted(times, times - seconds, side="left")
            pos = np.arange(len(idx))
            counts[idx] = pos - left
            sums[idx] = cumsum[pos] - amounts - np.where(left > 0, cumsum[left - 1], 0.0)
        df[f"customer_tx_count_prev_{label}"] = counts
        df[f"customer_amt_sum_prev_{label}"] = sums

    # Version demo rapide : on garde les premieres lignes chargees.
    # L objectif de l interface est de tester le scoring, SHAP et Qdrant, pas de recalculer tout l historique.
    return df.sort_values("TransactionDT").reset_index(drop=True)


def complete_feature_frame(row: pd.Series, feature_cols: list[str]) -> pd.DataFrame:
    values = {feature: row[feature] if feature in row.index else np.nan for feature in feature_cols}
    return pd.DataFrame([values], columns=feature_cols)


def top_shap_table(shap_values: np.ndarray, feature_names: list[str], n: int = 8) -> pd.DataFrame:
    row = np.asarray(shap_values).reshape(-1)
    order = np.argsort(np.abs(row))[::-1][:n]
    return pd.DataFrame(
        {
            "feature": [feature_names[i] for i in order],
            "contribution_shap": row[order],
            "effet": ["augmente le risque" if row[i] > 0 else "diminue le risque" for i in order],
        }
    )


def natural_language_explanation(score: float, decision: str, top_features: pd.DataFrame) -> str:
    risk_features = top_features[top_features["contribution_shap"] > 0]["feature"].head(4).tolist()
    safe_features = top_features[top_features["contribution_shap"] < 0]["feature"].head(2).tolist()
    risk_part = ", ".join(risk_features) if risk_features else "aucun facteur dominant"
    safe_part = ", ".join(safe_features) if safe_features else "aucun facteur protecteur dominant"
    return (
        f"Decision: {decision}. Le score fraude est {score:.3f}. "
        f"Les principaux facteurs qui poussent vers le risque sont: {risk_part}. "
        f"Les facteurs qui réduisent le score sont: {safe_part}. "
        "Cette explication sert d'aide à l'analyse, pas de justification réglementaire finale."
    )


def qdrant_search(query_vector: np.ndarray, limit: int = 5) -> tuple[pd.DataFrame, str | None]:
    if QdrantClient is None:
        return pd.DataFrame(), "qdrant-client n'est pas installé dans cet environnement."
    try:
        client = QdrantClient(url=QDRANT_URL, timeout=5)
        collections = [c.name for c in client.get_collections().collections]
        if QDRANT_COLLECTION not in collections:
            return pd.DataFrame(), f"Collection Qdrant introuvable: {QDRANT_COLLECTION}. Lance d'abord le notebook 08."
        if hasattr(client, "search"):
            hits = client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=query_vector.tolist(),
                limit=limit,
                with_payload=True,
            )
        else:
            response = client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=query_vector.tolist(),
                limit=limit,
                with_payload=True,
            )
            hits = response.points
    except Exception as exc:
        return pd.DataFrame(), f"Qdrant indisponible sur {QDRANT_URL}: {exc}"

    rows = []
    for rank, hit in enumerate(hits, start=1):
        payload = dict(hit.payload or {})
        payload["rank"] = rank
        payload["similarity_score"] = float(hit.score)
        rows.append(payload)
    return pd.DataFrame(rows), None


model, preprocessor, feature_cols, saved_metrics = load_model_assets()
demo_df = load_demo_transactions()

st.title("PayTrack Fraud Analyst")
st.caption("Score XGBoost, explication SHAP et recherche de transactions similaires dans Qdrant.")

with st.sidebar:
    st.header("Paramètres")
    threshold = st.slider("Seuil d'alerte", min_value=0.01, max_value=0.95, value=DEFAULT_THRESHOLD, step=0.01)
    top_k = st.slider("Transactions similaires", min_value=1, max_value=10, value=5, step=1)
    mode = st.radio("Mode de saisie", ["Transaction existante", "Saisie manuelle"], index=0)
    st.divider()
    st.subheader("Modèle chargé")
    st.write("`xgboost_optuna_tuned`")
    if saved_metrics:
        st.metric("AUPRC validation", f"{saved_metrics.get('AUPRC', np.nan):.4f}")
        st.metric("Recall@0.5", f"{saved_metrics.get('recall@0.5', np.nan):.2%}")

st.info(
    "XGBoost produit le score fraude. SHAP explique les facteurs du score. "
    "Qdrant retrouve des transactions historiques similaires si le service est lancé."
)

if mode == "Transaction existante":
    col_a, col_b = st.columns([2, 1])
    with col_a:
        fraud_only = st.checkbox("Afficher seulement des fraudes historiques", value=True)
        source_df = demo_df[demo_df["isFraud"] == 1] if fraud_only else demo_df
        selected_id = st.selectbox(
            "Transaction à analyser",
            options=source_df["TransactionID"].astype(int).tolist(),
            index=0,
        )
    selected = demo_df.loc[demo_df["TransactionID"] == selected_id].iloc[0].copy()
else:
    st.subheader("Saisie manuelle")
    defaults = demo_df[demo_df["isFraud"] == 1].iloc[-1].copy() if (demo_df["isFraud"] == 1).any() else demo_df.iloc[-1].copy()
    selected = defaults.copy()
    c1, c2, c3 = st.columns(3)
    with c1:
        selected["TransactionAmt"] = st.number_input("TransactionAmt", min_value=0.0, value=float(defaults.get("TransactionAmt", 100.0) or 100.0), step=10.0)
        selected["ProductCD"] = st.selectbox("ProductCD", ["W", "C", "R", "H", "S"], index=0)
        selected["card4"] = st.selectbox("card4", ["visa", "mastercard", "american express", "discover", "missing"], index=0)
        selected["card6"] = st.selectbox("card6", ["debit", "credit", "missing"], index=0)
    with c2:
        selected["P_emaildomain"] = st.text_input("P_emaildomain", value=str(defaults.get("P_emaildomain") or "gmail.com"))
        selected["R_emaildomain"] = st.text_input("R_emaildomain", value=str(defaults.get("R_emaildomain") or "gmail.com"))
        selected["DeviceType"] = st.selectbox("DeviceType", ["desktop", "mobile", "missing"], index=0)
        selected["hour_of_day_rel"] = st.number_input("hour_of_day_rel", min_value=0, max_value=23, value=int(defaults.get("hour_of_day_rel", 12) or 12))
    with c3:
        selected["customer_tx_count_prev_1h"] = st.number_input("tx client 1h", min_value=0, value=int(defaults.get("customer_tx_count_prev_1h", 0) or 0))
        selected["customer_tx_count_prev_24h"] = st.number_input("tx client 24h", min_value=0, value=int(defaults.get("customer_tx_count_prev_24h", 0) or 0))
        selected["customer_tx_count_prev_7d"] = st.number_input("tx client 7j", min_value=0, value=int(defaults.get("customer_tx_count_prev_7d", 0) or 0))
        selected["amt_to_customer_mean_prev"] = st.number_input("ratio montant / moyenne client", min_value=0.0, value=float(defaults.get("amt_to_customer_mean_prev", 1.0) or 1.0), step=0.1)

features_df = complete_feature_frame(selected, feature_cols)
prepared = preprocessor.transform(features_df)
score = float(model.predict_proba(prepared)[0, 1])
decision = "ALERTE FRAUDE" if score >= threshold else "Acceptée"

m1, m2, m3, m4 = st.columns(4)
m1.metric("Score fraude", f"{score:.3f}")
m2.metric("Décision", decision)
m3.metric("Seuil", f"{threshold:.2f}")
if "isFraud" in selected.index:
    m4.metric("Label historique", "Fraude" if int(selected.get("isFraud", 0)) == 1 else "Légitime")
else:
    m4.metric("Label historique", "Inconnu")

st.subheader("Transaction analysée")
preview_cols = [
    "TransactionID", "isFraud", "TransactionAmt", "ProductCD", "card4", "card6",
    "P_emaildomain", "R_emaildomain", "DeviceType", "day_rel", "hour_of_day_rel",
    "customer_tx_count_prev_1h", "customer_tx_count_prev_24h", "customer_tx_count_prev_7d",
    "amt_to_customer_mean_prev",
]
preview = pd.DataFrame([selected[[c for c in preview_cols if c in selected.index]]])
st.dataframe(preview, use_container_width=True)

st.subheader("Explication SHAP")
if shap is None:
    st.warning("SHAP n'est pas installé dans cet environnement. Installe `shap` pour afficher l'explication.")
    shap_row = None
else:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(prepared)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_row = np.asarray(shap_values)[0]
    top_shap = top_shap_table(shap_row, feature_cols, n=8)
    st.write(natural_language_explanation(score, decision, top_shap))
    st.dataframe(top_shap, use_container_width=True)
    st.bar_chart(top_shap.set_index("feature")["contribution_shap"])

st.subheader("Transactions similaires via Qdrant")
if shap_row is None:
    st.info("La recherche Qdrant utilise les SHAP values comme vecteur. Elle est disponible quand SHAP est chargé.")
else:
    norm = np.linalg.norm(shap_row)
    query_vector = shap_row / norm if norm > 0 else shap_row
    similar_df, error = qdrant_search(query_vector, limit=top_k)
    if error:
        st.warning(error)
        st.code("docker start <container_qdrant>\n# ou\ndocker run -p 6333:6333 -p 6334:6334 -v \"$(pwd)/qdrant_storage:/qdrant/storage\" qdrant/qdrant")
    elif similar_df.empty:
        st.info("Aucune transaction similaire retournee.")
    else:
        cols_first = [
            "rank", "similarity_score", "TransactionID", "isFraud", "TransactionAmt", "ProductCD",
            "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType", "day_rel", "top_shap_features",
        ]
        cols_first = [c for c in cols_first if c in similar_df.columns]
        st.dataframe(similar_df[cols_first + [c for c in similar_df.columns if c not in cols_first]], use_container_width=True)
        if "isFraud" in similar_df.columns:
            fraud_rate = pd.to_numeric(similar_df["isFraud"], errors="coerce").mean()
            st.caption(f"Taux de fraude parmi les voisins retournés : {fraud_rate:.1%}")

st.caption("Démo locale : les features historiques d'une saisie manuelle sont approximatives. En production, elles doivent être recalculées depuis l'historique réel avant scoring.")
