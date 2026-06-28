# Briefing presentation - Projet PayTrack Fraud Detection

Ce document sert de guide pour les personnes qui n'ont pas participe au projet mais qui doivent comprendre la presentation et pouvoir repondre aux questions. Il resume le contexte, les choix techniques, les resultats et les messages a faire passer pendant l'oral.

## 1. Idee generale du projet

PayTrack est une fintech fictive specialisee dans les paiements en ligne. Le but du projet est de remplacer ou completer un systeme de regles fixes par une chaine de detection fraude plus robuste : modele de scoring, explication des decisions, suivi des versions, monitoring de drift et aide aux analystes.

La phrase simple a retenir :

> XGBoost detecte le risque de fraude, SHAP explique la decision, MLflow trace les versions, le monitoring surveille la degradation, NetworkX/GNN exploitent les relations, et Qdrant aide l'analyste a retrouver des transactions similaires.

Le projet ne se limite donc pas a entrainer un modele. Il montre une chaine presque complete de mise en production et d'analyse fraude.

## 2. Pipeline global

Le pipeline presente est :

```text
Donnees brutes
-> EDA
-> Feature engineering
-> XGBoost + Optuna
-> SHAP
-> MLflow Registry
-> Validation gate
-> Monitoring drift
-> Analyste fraude
```

Deux briques complementaires sont ajoutees :

- **NetworkX / graph features** : exploiter les relations client-marchand.
- **Qdrant / recherche vectorielle** : retrouver les transactions historiques similaires.

Une partie GNN est aussi traitee sur Elliptic Bitcoin Dataset pour montrer l'usage de GCN/GAT sur des donnees naturellement graphes.

## 3. Donnees utilisees

### Dataset principal : IEEE-CIS Fraud Detection

Le dataset principal contient environ **590 540 transactions**. La cible est `isFraud`.

Repartition :

| Classe | Volume | Taux |
|---|---:|---:|
| Legitime | 569 877 | 96.50 % |
| Fraude | 20 663 | 3.50 % |

Point important : un modele qui predit toujours `legitime` obtient deja **96.5 % d'accuracy**, mais ne detecte aucune fraude. Donc l'accuracy est trompeuse.

Les metriques importantes sont :

- **AUPRC** : qualite du ranking fraude sur probleme desequilibre.
- **Recall fraude** : part des fraudes detectees.
- **Precision** : part des alertes qui sont vraiment frauduleuses.
- **F1** : compromis precision / recall.
- **Taux d'alertes** : volume envoye aux analystes.
- **Temps d'inference** : capacite a fonctionner en production.

### Dataset secondaire : Elliptic Bitcoin Dataset

Utilise pour la partie GNN : GCN vs GAT. Il ne remplace pas le dataset PayTrack principal. Il sert a montrer comment les graph neural networks peuvent detecter des activites illicites dans un reseau de transactions blockchain.

## 4. Feature engineering : comment on cree l'historique client

Le dataset IEEE-CIS ne fournit pas de vrai `customer_id`. On construit donc un proxy client :

```text
customer_proxy = card1 + card2 + card3 + card5 + addr1
```

Ce proxy est imparfait, mais il permet de regrouper des transactions qui ressemblent au meme client ou moyen de paiement.

On construit aussi un proxy marchand :

```text
merchant_proxy = ProductCD + R_emaildomain
```

Cela permet de calculer :

- nombre de transactions precedentes du client ;
- montant moyen historique du client ;
- ratio montant actuel / moyenne historique ;
- nouveau marchand ou marchand deja vu ;
- transactions sur les 1h, 24h et 7 jours precedents ;
- montants cumules sur les 1h, 24h et 7 jours precedents.

Message oral : ces features donnent au modele une memoire du comportement client, au lieu de traiter chaque transaction de facon isolee.

## 5. Resultats EDA importants

L'EDA montre plusieurs signaux :

- la fraude est rare et desequilibree ;
- le risque varie selon le temps ;
- `ProductCD`, type de carte, domaine email et device sont informatifs ;
- les cartes `credit` sont plus risquees que `debit` ;
- `DeviceType=mobile` presente un taux de fraude plus eleve ;
- certaines variables anonymisees `Vxxx` sont fortement correlees a `isFraud`.

Variables `Vxxx` importantes observees : `V257`, `V246`, `V244`, `V242`, `V201`, `V200`.

Nuance : les `Vxxx` ont du signal, mais elles sont anonymisees. Elles sont utiles pour predire, mais moins faciles a expliquer metier.

## 6. Compact vs Vxxx

Deux jeux de features ont ete compares :

| Feature set | Features | Vxxx | AUPRC | Recall@0.5 | F1@0.5 | Taux alertes |
|---|---:|---:|---:|---:|---:|---:|
| compact | 59 | 0 | 0.4753 | 0.2904 | 0.4206 | 1.31 % |
| wide_selected_v | 109 | 50 | 0.4734 | 0.2923 | 0.4250 | 1.29 % |

Conclusion : les `Vxxx` testees n'ameliorent pas l'AUPRC dans ce run. Le modele compact reste plus lisible et leger. Il faut dire :

> Les Vxxx contiennent du signal, mais la selection top 50 par correlation ne donne pas de gain d'AUPRC. On les garde comme piste, pas comme preuve d'amelioration.

## 7. Modeles tabulaires testes

Strategie principale : XGBoost sur donnees tabulaires.

Strategies comparees :

- `xgboost_baseline` ;
- `xgboost_scale_pos_weight` ;
- `random_undersampling` ;
- `SMOTE` ;
- `SMOTEENN`.

Resultats au seuil 0.5 dans le notebook 02 :

| Strategie | AUPRC | Precision@0.5 | Recall@0.5 | F1@0.5 | Taux alertes |
|---|---:|---:|---:|---:|---:|
| xgboost_baseline | 0.4753 | 0.7628 | 0.2904 | 0.4206 | 1.31 % |
| scale_pos_weight | 0.4525 | 0.1585 | 0.7576 | 0.2621 | 16.45 % |
| random_undersampling | 0.4523 | 0.1654 | 0.7544 | 0.2713 | 15.70 % |
| SMOTEENN | 0.4265 | 0.4097 | 0.4481 | 0.4280 | 3.76 % |
| SMOTE | 0.4137 | 0.4245 | 0.4240 | 0.4242 | 3.44 % |

Interpretation :

- XGBoost baseline a le meilleur ranking global en AUPRC.
- Les methodes de resampling augmentent le recall, mais generent beaucoup trop d'alertes ou baissent l'AUPRC.
- SMOTE/SMOTEENN ne battent pas la baseline en AUPRC.

## 8. Seuil de decision

Le modele donne un score de fraude entre 0 et 1. Ensuite il faut choisir un seuil.

Seuil 0.5 :

- precision elevee ;
- recall faible ;
- peu d'alertes.

Seuil 0.25 :

- precision 58.43 % ;
- recall 41.29 % ;
- F1 0.4839 ;
- taux d'alertes 2.43 %.

Message oral :

> Le seuil est une decision metier. Si PayTrack veut detecter plus de fraudes, 0.25 est plus interessant que 0.5, avec un volume d'alertes encore raisonnable.

## 9. Optuna : meilleur modele tabulaire

Optuna optimise les hyperparametres XGBoost. Le tuning a ete fait sur 25 essais.

Resultat :

| Modele | AUPRC | Precision@0.5 | Recall@0.5 | F1@0.5 | Taux alertes | Gain AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost baseline | 0.4753 | 0.7628 | 0.2904 | 0.4206 | 1.31 % | 0.0000 |
| XGBoost Optuna tuned | 0.5424 | 0.7721 | 0.3634 | 0.4942 | 1.62 % | +0.0671 |

Conclusion importante :

> Le meilleur modele tabulaire actuel est XGBoost compact tune avec Optuna.

Attention : les runs MLflow historiques peuvent encore montrer le XGBoost baseline non tune. Si on presente Optuna comme modele final, il faut preciser que c'est le meilleur resultat du notebook 02 et qu'il doit ensuite etre logge/promu dans MLflow.

## 10. SHAP : explicabilite

SHAP sert a expliquer les predictions du modele. Il repond a la question :

> Pourquoi cette transaction a-t-elle ete bloquee ?

Interpretation :

- valeur SHAP positive : la feature augmente le score de fraude ;
- valeur SHAP negative : la feature diminue le score de fraude.

Top features SHAP observees :

- `C5`, `C1` : compteurs transactionnels ;
- `D3`, `D2` : delais / historique temporel ;
- `TransactionAmt` : montant ;
- `card6` : type de carte ;
- `C14`, `C13`, `C11`, `C4`.

Template analyste :

> La transaction a ete signalee car plusieurs facteurs augmentent son risque : montant inhabituel, activite recente elevee, type de carte ou signaux transactionnels C/D eleves. Ces elements rapprochent la transaction de patterns observes dans des fraudes passees.

Message cle : SHAP ne remplace pas l'analyste, il l'aide a comprendre la decision.

## 11. MLflow : tracking et deploiement controle

MLflow sert a :

- enregistrer les runs ;
- comparer les modeles ;
- logguer parametres et metriques ;
- stocker les artefacts : courbes PR, matrices de confusion, preprocessor ;
- enregistrer un modele candidat dans le registry ;
- simuler une validation gate.

MLflow ne detecte pas directement le drift. C'est une erreur a eviter a l'oral.

Phrase claire :

> MLflow sert au cycle de vie du modele. Le monitoring sert a surveiller le comportement du modele apres deploiement.

Dans le projet, l'experience s'appelle :

```text
fraud-detection-paytrack
```

Le modele registry utilise :

```text
paytrack-fraud-xgboost
```

## 12. Validation gate

La validation gate est une regle automatique qui decide si un modele peut etre candidat au deploiement.

Criteres presentes :

- AUPRC > 0.50 ;
- Recall > 35 % ;
- Alert Rate < 3 % ;
- Inference Time < 50 ms.

Ces criteres correspondent bien au modele Optuna :

- AUPRC 0.5424 ;
- recall@0.5 36.34 % ;
- alert rate 1.62 % ;
- inference tres inferieure a 50 ms par ligne dans MLflow pour XGBoost baseline.

Nuance importante : dans le notebook MLflow, la gate historique etait plutot configuree sur le modele baseline avec un seuil 0.25 et AUPRC minimale 0.45. Si on presente la slide avec AUPRC > 0.50, il faut l'expliquer comme une gate cible adaptee au modele Optuna.

## 13. Monitoring drift : T1, T2 et PSI

Le monitoring simule deux periodes :

```text
T1 = periode de reference stable
T2 = periode recente / drifted
```

On compare T2 a T1 pour savoir si le modele reste fiable.

Ce qui est calcule :

1. **Performance drift** : AUPRC T1 vs AUPRC T2.
2. **Score drift** : evolution de la distribution des scores de fraude.
3. **Data drift** : evolution des distributions des features critiques via PSI.

Resultats :

| Periode | AUPRC | Precision | Recall | F1 | Taux alertes |
|---|---:|---:|---:|---:|---:|
| T1 stable | 0.4873 | 0.6024 | 0.3933 | 0.4759 | 2.23 % |
| T2 drifted | 0.4464 | 0.5666 | 0.3934 | 0.4644 | 2.41 % |

Baisse AUPRC : **8.40 %**.

PSI maximum : **0.0033**.

Interpretation :

- l'AUPRC baisse legerement ;
- la baisse reste sous le seuil d'alerte de 15 % ;
- le PSI est tres faible, donc pas de drift global fort sur les features surveillees ;
- decision : pas de retraining immediat.

### C'est quoi le PSI ?

PSI = **Population Stability Index**. Il mesure si une feature a change de distribution entre T1 et T2.

Regle simple :

| PSI | Interpretation |
|---:|---|
| < 0.10 | stable |
| 0.10 - 0.25 | changement modere |
| > 0.25 | changement fort |

Dans le projet, PSI max = 0.0033, donc tres stable.

## 14. NetworkX : graph features

Cette partie construit un graphe de 10 000 transactions :

- noeuds = clients et marchands ;
- aretes = transactions ;
- objectif = capturer les relations entre acteurs.

Resultats du graphe :

- 10 000 transactions ;
- 4 119 noeuds ;
- 4 501 aretes ;
- 14 composants ;
- cluster suspect principal : composant 0 ;
- taux de fraude dans ce cluster : 6.94 %.

Features extraites :

- degre du compte ;
- degre pondere ;
- betweenness centrality ;
- nombre de marchands distincts sur 7 jours.

Impact modele :

| Modele | AUPRC | Recall | F1 | Gain AUPRC |
|---|---:|---:|---:|---:|
| XGBoost base | 0.5570 | 0.4805 | 0.5873 | 0.0000 |
| XGBoost + graph features | 0.6096 | 0.5065 | 0.6000 | +0.0526 |

Conclusion : c'est une des ameliorations les plus fortes du projet. Les graph features capturent un signal relationnel que les variables tabulaires seules ne voient pas.

## 15. GNN : GCN vs GAT

Cette partie utilise Elliptic Bitcoin Dataset, pas IEEE-CIS.

Pourquoi c'est lie a la fraude ? Parce que certaines fraudes sont relationnelles : reseaux de comptes, transactions connectees, blanchiment, mule accounts.

Definitions :

- **GNN** : Graph Neural Network, famille de modeles qui apprend sur des graphes.
- **GCN** : Graph Convolutional Network, agrege les voisins de maniere plutot uniforme.
- **GAT** : Graph Attention Network, apprend quels voisins sont les plus importants avec un mecanisme d'attention.

Resultats :

| Modele | AUPRC | Recall | Precision | F1 | Inference ms/noeud |
|---|---:|---:|---:|---:|---:|
| GCN | 0.2091 | 0.4778 | 0.2146 | 0.2962 | 0.0182 |
| GAT | 0.1688 | 0.9197 | 0.0782 | 0.1441 | 0.0866 |

Interpretation :

- GCN est meilleur en AUPRC et plus rapide ;
- GAT detecte beaucoup plus de positifs, mais genere beaucoup de faux positifs ;
- GAT peut etre superieur en theorie quand certains voisins sont beaucoup plus informatifs que d'autres.

Message oral : la partie GNN est une extension graphe, pas le modele principal PayTrack.

## 16. Qdrant : recherche de transactions similaires

Qdrant n'est pas le modele de detection principal. Il sert a l'investigation.

Pipeline :

```text
XGBoost -> score de fraude
SHAP -> explication de la decision
Qdrant -> retrouve les 5 transactions historiques les plus similaires
```

Les embeddings peuvent etre les valeurs SHAP. `isFraud` n'est pas utilise dans l'embedding ; il est garde dans le payload pour analyser les voisins retrouves.

Phrase simple :

> XGBoost detecte, SHAP explique, Qdrant contextualise.

Utilite analyste : si une transaction suspecte ressemble a plusieurs fraudes passees, l'analyste peut reconstituer un pattern.

## 17. Interface Streamlit

Une interface demo a ete creee dans :

```text
app/streamlit_app.py
```

Elle permet :

- choisir une transaction historique ou saisir une transaction ;
- obtenir le score XGBoost ;
- voir la decision selon un seuil ;
- afficher les top contributions SHAP ;
- interroger Qdrant pour les transactions similaires.

Commande :

```bash
streamlit run app/streamlit_app.py
```

Si Qdrant tourne sur un autre port :

```bash
QDRANT_URL=http://localhost:6335 streamlit run app/streamlit_app.py
```

## 18. Ce qu'il faut dire slide par slide

### Slide 1 - Titre

Dire : on presente PayTrack, une chaine complete de detection de fraude bancaire par machine learning.

Message cle : le projet couvre detection, explication, industrialisation et monitoring.

### Slide 2 - Objectifs strategiques

Dire : PayTrack cherche a proteger les revenus, eviter les faux positifs et aider les analystes avec des explications.

Message cle : la fraude est un probleme metier, pas seulement un probleme de modele.

### Slide 3 - Desequilibre

Dire : seulement 3.5 % de fraudes, donc l'accuracy est piegeuse. On privilegie AUPRC, recall, precision.

Message cle : une accuracy de 96.5 % peut etre inutile.

### Slide 4 - Donnees

Dire : dataset IEEE-CIS, transactions + identity/device + variables Vxxx anonymisees.

Message cle : donnees riches mais partiellement anonymisees.

### Slide 5 - EDA

Dire : plusieurs signaux faibles apparaissent : temps, type de carte, device, email.

Message cle : la fraude depend du contexte.

### Slide 6 - Vxxx

Dire : les Vxxx sont correlees a la fraude mais moins explicables et potentiellement redondantes.

Message cle : on les teste separement.

### Slide 7 - Proxy client / marchand

Dire : pas de vrai customer_id, donc on reconstruit un proxy client et marchand.

Message cle : cela permet de calculer un historique comportemental.

### Slide 8 - Velocite transactionnelle et split temporel

Dire : on calcule des signaux sur 1h, 24h, 7j et on valide sur le futur.

Message cle : evaluer temporellement evite une validation trop optimiste.

### Slide 10 - Compact vs Vxxx

Dire : compact bat legerement wide_selected_v en AUPRC, donc on garde compact comme base lisible.

Message cle : les Vxxx ne prouvent pas de gain dans ce run.

### Slide 11 - Benchmarks modeles

Dire : XGBoost baseline a le meilleur ranking, resampling detecte plus mais alerte trop.

Message cle : recall seul ne suffit pas.

### Slide 12 - PR curves

Dire : l'AUPRC est la boussole car le dataset est desequilibre.

Message cle : on cherche un bon score de risque.

### Slide 13 - Seuil

Dire : seuil 0.25 detecte plus de fraudes que 0.5 avec un volume d'alertes acceptable.

Message cle : le seuil est un choix operationnel.

### Slide 14 - Optuna

Dire : le tuning donne le meilleur gain tabulaire, AUPRC 0.5424.

Message cle : meilleur modele tabulaire = XGBoost compact Optuna.

### Slide 15 - SHAP global

Dire : SHAP identifie les drivers globaux du risque : compteurs, delais, montant.

Message cle : le modele devient interpretable.

### Slide 16 - SHAP local

Dire : on peut expliquer une transaction bloquee avec une phrase lisible pour analyste.

Message cle : l'explication aide a traiter les alertes.

### Slide 17 - MLflow

Dire : MLflow trace les runs, les artefacts et les versions de modeles.

Message cle : MLflow = lifecycle modele, pas drift.

### Slide 18 - Validation gate

Dire : la gate evite de promouvoir un modele qui ne respecte pas les criteres metier.

Message cle : deploiement controle par seuils mesurables.

### Slide 19 - Monitoring

Dire : T1 est la reference, T2 la periode recente ; on compare AUPRC, scores et PSI.

Message cle : pas d'alerte retraining immediate dans la simulation.

### Slide 20 - NetworkX

Dire : la fraude peut etre visible dans les relations client-marchand.

Message cle : le graphe apporte une vision collective.

### Slide 21 - Gain graph

Dire : ajout des graph features : AUPRC 0.5570 -> 0.6096.

Message cle : les graph features sont une piste forte.

### Slide 22 - GNN

Dire : GCN plus equilibre, GAT plus sensible mais trop de faux positifs.

Message cle : GNN utile pour donnees naturellement graphes.

### Slide 23 - Qdrant

Dire : Qdrant retrouve les transactions similaires pour aider l'analyste.

Message cle : Qdrant contextualise, il ne remplace pas XGBoost.

### Slide 24 - Architecture

Dire : montrer la chaine complete de bout en bout.

Message cle : solution systeme, pas simple notebook.

### Slide 25 - Synthese

Dire : deploiement recommande = XGBoost Optuna + seuil 0.25 a calibrer + SHAP + MLflow + monitoring + evolution graph/Qdrant.

Message cle : modele performant, explicable et surveille.

## 19. Questions probables et reponses courtes

### Pourquoi pas l'accuracy ?

Parce que 96.5 % des transactions sont legitimes. Un modele qui ne detecte aucune fraude peut avoir 96.5 % d'accuracy.

### Pourquoi AUPRC ?

Parce que la fraude est rare. L'AUPRC mesure la qualite du classement des fraudes quand les classes sont desequilibrees.

### Pourquoi XGBoost ?

Il est tres performant sur donnees tabulaires, gere les non-linearites et fonctionne bien avec valeurs manquantes et features heterogenes.

### Pourquoi Optuna ?

Pour optimiser automatiquement les hyperparametres XGBoost et maximiser l'AUPRC validation.

### Pourquoi ne pas retenir les Vxxx ?

Elles ont du signal, mais dans le test `wide_selected_v`, elles n'ameliorent pas l'AUPRC. Elles augmentent la complexite et reduisent l'explicabilite.

### MLflow detecte-t-il le drift ?

Non. MLflow trace les modeles et runs. Le drift est surveille par le notebook de monitoring avec AUPRC T1/T2, score drift et PSI.

### C'est quoi T1 et T2 ?

T1 est la periode de reference. T2 est une periode recente ou simulee drifted. On compare les deux.

### C'est quoi PSI ?

Population Stability Index. Il mesure si la distribution d'une feature change entre T1 et T2.

### Qdrant detecte-t-il la fraude ?

Non. Qdrant retrouve les transactions similaires. Le detecteur principal reste XGBoost.

### GNN fait-il partie du modele PayTrack principal ?

Non. C'est une extension sur Elliptic Bitcoin pour repondre au sujet et montrer l'interet des graph neural networks.

### Quel est le meilleur resultat du projet ?

Pour le tabulaire : XGBoost compact Optuna, AUPRC 0.5424. Pour les graph features : gain AUPRC +0.0526 sur le sous-graphe.

## 20. Points de vigilance pendant la presentation

- Ne pas dire que MLflow detecte le drift.
- Ne pas dire que Qdrant est le modele de detection.
- Ne pas dire que les Vxxx ameliorent le modele : elles ont ete testees mais pas retenues comme gain AUPRC.
- Ne pas confondre IEEE-CIS et Elliptic : IEEE-CIS = modele PayTrack principal ; Elliptic = GNN.
- Ne pas presenter le seuil 0.5 comme optimal.
- Dire que le proxy client est une approximation, pas un vrai identifiant client.
- Dire que les graph features ont ete validees sur un sous-graphe de 10 000 transactions, pas sur toute la production.

## 21. Phrase de conclusion recommandee

> Le projet montre qu'une solution fraude efficace ne repose pas seulement sur un modele. PayTrack dispose d'un score XGBoost performant, d'explications SHAP pour les analystes, d'un suivi MLflow pour controler les versions, d'un monitoring pour detecter les derives, et de pistes d'amelioration fortes avec les graph features et Qdrant. La recommandation est de partir sur XGBoost compact tune avec Optuna, de calibrer le seuil metier autour de 0.25, puis d'industrialiser progressivement les briques SHAP, MLflow, monitoring et graph features.
