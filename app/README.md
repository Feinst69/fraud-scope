# Interface PayTrack Fraud Analyst

Cette app Streamlit permet de démontrer le pipeline analyste :

1. sélection ou saisie d'une transaction ;
2. score fraude avec `models/xgboost_optuna_tuned.ubj` ;
3. explication SHAP ;
4. recherche des transactions similaires dans Qdrant.

## Prérequis

Les artefacts suivants doivent exister :

- `models/xgboost_optuna_tuned.ubj`
- `models/xgboost_optuna_tuned_preprocessor.pkl`
- `models/xgboost_optuna_tuned_features.json`

Pour la recherche de similarité, Qdrant doit tourner et la collection `fraud_transactions_shap` doit avoir été créée par `notebooks/08_similarity_search_qdrant.ipynb`.

## Lancer Qdrant

Si le conteneur existe déjà :

```bash
docker ps -a

docker start <nom_ou_id_du_container_qdrant>
```

Sinon :

```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant
```

## Lancer l'app

```bash
streamlit run app/streamlit_app.py
```

Si Streamlit manque :

```bash
pip install streamlit xgboost shap qdrant-client pandas numpy scikit-learn
```
