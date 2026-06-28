# Fraud Scope - Détection de fraude PayTrack

Projet M2 Data-IA autour de la détection de fraude, avec une chaîne complète allant de l'EDA au monitoring en passant par XGBoost, SHAP, MLflow, NetworkX et GNN.

## Objectif

Construire un système de détection de fraude capable de :

- traiter un dataset fortement déséquilibré ;
- comparer plusieurs stratégies de modélisation ;
- expliquer les décisions du modèle ;
- tracer les expériences avec MLflow ;
- simuler une validation gate avant production ;
- surveiller drift et performance ;
- exploiter des features de graphe ;
- comparer GCN et GAT sur Elliptic Bitcoin Dataset.

## Structure du projet

```text
.
├── data/                         # Données locales, ignorées par git
├── docs/                         # Rapports, CSV de résultats et images
│   ├── assets/                   # Figures générées par les notebooks
│   ├── rapport_final.md          # Rapport global du projet
│   ├── rapport_shap_explicabilite.md
│   ├── rapport_monitoring_drift.md
│   ├── rapport_graph_features_networkx.md
│   └── rapport_gnn_elliptic_gcn_gat.md
├── evidently_reports/            # Rapports HTML Evidently
├── mlartifacts/                  # Artefacts MLflow
├── mlflow.db                     # Backend SQLite MLflow
└── notebooks/
    ├── 01_exploration_polars.ipynb
    ├── 02_modelling.ipynb
    ├── 03_explainability_shap.ipynb
    ├── 04_mlflow_tracking_registry.ipynb
    ├── 05_monitoring_drift_evidently.ipynb
    ├── 06_graph_features_networkx.ipynb
    └── 07_gnn_elliptic_gcn_gat.ipynb
```

## Données

Le dataset principal est **IEEE-CIS Fraud Detection**. Les fichiers attendus dans `data/` sont :

- `train_transaction.csv`
- `train_identity.csv`
- `test_transaction.csv`
- `test_identity.csv`
- `sample_submission.csv`

La partie GNN utilise **Elliptic Bitcoin Dataset**. Le notebook `07_gnn_elliptic_gcn_gat.ipynb` peut le charger via PyTorch Geometric ou depuis `data/elliptic/`.

## Ordre conseillé d'exécution

1. `notebooks/01_exploration_polars.ipynb`
2. `notebooks/02_modelling.ipynb`
3. `notebooks/03_explainability_shap.ipynb`
4. `notebooks/04_mlflow_tracking_registry.ipynb`
5. `notebooks/05_monitoring_drift_evidently.ipynb`
6. `notebooks/06_graph_features_networkx.ipynb`
7. `notebooks/07_gnn_elliptic_gcn_gat.ipynb`

## Résultats principaux

### XGBoost PayTrack

Le meilleur modèle principal est **XGBoost baseline**.

| Stratégie | AUPRC | Precision | Recall | F1 | Taux alertes |
|---|---:|---:|---:|---:|---:|
| XGBoost baseline | 0.4753 | 0.5843 | 0.4129 | 0.4839 | 2.43 % |
| XGBoost scale_pos_weight | 0.4525 | 0.0723 | 0.9252 | 0.1342 | 44.01 % |
| Random undersampling | 0.4523 | 0.0787 | 0.8999 | 0.1448 | 39.33 % |

### Compact vs Vxxx

Le notebook `02_modelling.ipynb` compare aussi le modèle compact sans `Vxxx` avec une version `wide_selected_v` ajoutant 50 colonnes anonymisées.

| Feature set | AUPRC | Recall | F1 | Gain AUPRC |
|---|---:|---:|---:|---:|
| compact | 0.4753 | 0.2904 | 0.4206 | 0.0000 |
| wide_selected_v | 0.4734 | 0.2923 | 0.4250 | -0.0019 |

Les `Vxxx` n'améliorent pas l'AUPRC dans ce run, mais restent une piste à tester avec une sélection plus robuste.

### Graph features NetworkX

Sur un sous-graphe de 10 000 transactions, l'ajout de graph features améliore l'AUPRC :

| Modèle | AUPRC | Recall | F1 | Gain AUPRC |
|---|---:|---:|---:|---:|
| XGBoost base | 0.5570 | 0.4805 | 0.5873 | 0.0000 |
| XGBoost + graph features | 0.6096 | 0.5065 | 0.6000 | +0.0526 |

### GNN Elliptic

| Modèle | AUPRC | Recall | Precision | Inférence ms/nœud |
|---|---:|---:|---:|---:|
| GCN | 0.2091 | 0.4778 | 0.2146 | 0.018188 |
| GAT | 0.1688 | 0.9197 | 0.0782 | 0.086605 |

Le GCN est meilleur selon l'AUPRC. Le GAT maximise le recall mais produit beaucoup plus de faux positifs.

## Rapports

Le rapport final est ici :

- `docs/rapport_final.md`

Rapports détaillés :

- `docs/analyse_resultats_reels_polars_modelling.md`
- `docs/rapport_shap_explicabilite.md`
- `docs/rapport_monitoring_drift.md`
- `docs/rapport_graph_features_networkx.md`
- `docs/rapport_gnn_elliptic_gcn_gat.md`

## MLflow

Le projet utilise MLflow avec un backend SQLite :

- base : `mlflow.db`
- artefacts : `mlartifacts/`
- expérience : `fraud-detection-paytrack`

Pour ouvrir l'interface MLflow :

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
```

Puis ouvrir l'URL affichée par MLflow, généralement `http://127.0.0.1:5000`.

## Conclusion

Le candidat recommandé pour PayTrack est **XGBoost baseline**, piloté par l'AUPRC et un seuil métier ajustable. SHAP fournit l'explicabilité analyste, MLflow structure le déploiement, le monitoring surveille les dérives, et les graph features sont la piste la plus prometteuse pour améliorer le modèle tabulaire.
