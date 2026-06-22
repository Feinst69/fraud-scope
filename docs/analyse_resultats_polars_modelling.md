# Analyse des resultats - Exploration et modelisation Polars

## Contexte

Ce document analyse les resultats produits par les notebooks :

- `notebooks/01_exploration_polars.ipynb` : exploration des donnees IEEE-CIS Fraud Detection avec Polars ;
- `notebooks/02_modelling.ipynb` : comparaison de plusieurs strategies XGBoost et resampling.

Les notebooks ne contiennent pas de sorties sauvegardees dans le fichier `.ipynb` au moment de la redaction. Les valeurs numeriques ci-dessous doivent donc etre completees avec les resultats obtenus apres execution locale.

## 1. Analyse de l'exploration

### Desequilibre de classes

Le dataset IEEE-CIS est fortement desequilibre : la classe fraude represente une faible proportion des transactions. Ce point est central pour tout le projet, car il rend l'accuracy peu informative.

Resultats a completer :

| Indicateur | Valeur |
|---|---:|
| Nombre total de transactions train | `...` |
| Nombre de transactions frauduleuses | `...` |
| Taux de fraude | `... %` |
| Accuracy d'un modele toujours legitime | `... %` |

Interpretation : si le taux de fraude est autour de quelques pourcents, un modele peut obtenir une accuracy tres elevee en predisant presque toujours la classe legitime. Ce comportement serait pourtant inutilisable en production, car il manquerait la majorite des fraudes.

Mon avis : l'EDA montre bien que la metrique principale ne doit pas etre l'accuracy. Pour PayTrack, il faut piloter la comparaison avec l'AUPRC, le recall fraude et le taux d'alertes generees.

### Analyse temporelle

L'utilisation de `TransactionDT` permet de reconstruire un ordre temporel relatif. Les variables `day_rel`, `week_rel` et `hour_of_day_rel` servent a observer l'evolution du volume et du taux de fraude dans le temps.

Resultats a completer :

| Observation | Conclusion |
|---|---|
| Le volume de transactions est-il stable dans le temps ? | `...` |
| Le taux de fraude varie-t-il selon les jours ? | `...` |
| Certaines heures relatives concentrent-elles plus de fraude ? | `...` |

Interpretation : toute variation temporelle du taux de fraude confirme qu'un split aleatoire serait trompeur. Un modele evalue avec un melange passe/futur peut obtenir de bons scores sans etre fiable en production.

Mon avis : le choix du split temporel est indispensable. C'est aussi un bon argument pour la soutenance : la fraude evolue, donc le protocole d'evaluation doit simuler le futur.

### Montants et comportement transactionnel

L'analyse de `TransactionAmt` permet de comparer les montants des fraudes et des transactions legitimes. La distribution est generalement asymetrique, avec quelques transactions tres elevees.

Resultats a completer :

| Indicateur | Legitime | Fraude |
|---|---:|---:|
| Montant moyen | `...` | `...` |
| Montant median | `...` | `...` |
| P95 | `...` | `...` |
| P99 | `...` | `...` |

Interpretation : si les montants frauduleux ont une distribution differente, `TransactionAmt` est une variable utile. Mais elle ne suffit probablement pas seule : les fraudes peuvent aussi prendre la forme de petits montants repetes ou de comportements inhabituels.

Mon avis : les features de deviation comportementale sont plus pertinentes que les montants bruts. Le ratio `amt_to_customer_mean_prev` est une bonne direction, meme si le `customer_proxy` reste une approximation.

### Valeurs manquantes

Le dataset contient beaucoup de valeurs manquantes, notamment dans les colonnes d'identite et de device.

Interpretation : dans ce contexte, une valeur manquante n'est pas seulement un probleme technique. Elle peut porter un signal de risque : absence d'information device, identite incomplete, email absent, etc.

Mon avis : il ne faut pas supprimer trop agressivement les colonnes avec valeurs manquantes. Pour XGBoost, on peut garder une partie de ces variables, surtout si elles sont disponibles en production.

### Categorie produit, carte, email et device

Les categories comme `ProductCD`, `card4`, `card6`, `P_emaildomain`, `R_emaildomain` et `DeviceType` permettent d'identifier des segments plus risques.

Resultats a completer :

| Variable | Modalites a risque observees | Commentaire |
|---|---|---|
| `ProductCD` | `...` | `...` |
| `card4` | `...` | `...` |
| `card6` | `...` | `...` |
| `P_emaildomain` | `...` | `...` |
| `R_emaildomain` | `...` | `...` |
| `DeviceType` | `...` | `...` |

Mon avis : ces variables sont utiles pour un premier modele, mais il faut rester prudent. Une modalite avec un taux de fraude eleve peut simplement avoir un faible volume. Il faut toujours regarder le nombre d'observations en plus du taux.

## 2. Analyse de la modelisation

### Baseline XGBoost

La baseline XGBoost sert de reference plus serieuse que la regression logistique naive. Elle doit normalement mieux capter les interactions non lineaires entre montant, carte, email, device et historique client.

Resultats a completer :

| Strategie | AUPRC | Precision@0.5 | Recall@0.5 | F1@0.5 | Taux d'alertes | Temps train |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost baseline | `...` | `...` | `...` | `...` | `...` | `...` |
| XGBoost scale_pos_weight | `...` | `...` | `...` | `...` | `...` | `...` |
| Random Undersampling | `...` | `...` | `...` | `...` | `...` | `...` |
| SMOTE | `...` | `...` | `...` | `...` | `...` | `...` |
| SMOTEENN | `...` | `...` | `...` | `...` | `...` | `...` |

### Lecture des strategies

#### XGBoost baseline

Cette strategie sert de point de depart. Elle ne corrige pas explicitement le desequilibre, mais XGBoost peut deja obtenir un score correct si les features sont informatives.

Mon avis : si cette baseline a une bonne AUPRC mais un recall faible au seuil 0.5, ce n'est pas forcement un mauvais modele. Cela signifie surtout que le seuil de decision doit etre ajuste.

#### XGBoost avec `scale_pos_weight`

Cette strategie force le modele a donner plus de poids aux fraudes pendant l'entrainement.

Effet attendu :

- recall fraude souvent meilleur ;
- plus d'alertes ;
- precision parfois plus faible ;
- bon compromis en production car simple a maintenir.

Mon avis : c'est souvent la strategie la plus propre pour un premier systeme industriel. Elle evite de fabriquer des exemples synthetiques et reste facile a expliquer a une equipe technique.

#### Random Undersampling

Cette strategie reduit le nombre d'exemples legitimes pour equilibrer le train.

Effet attendu :

- entrainement plus rapide ;
- recall parfois eleve ;
- perte d'information sur la classe legitime ;
- risque de faux positifs plus important.

Mon avis : utile comme experience comparative, mais risquee en production si elle degrade trop la precision. En fraude, perdre trop d'exemples legitimes peut empecher le modele d'apprendre la diversite du comportement normal.

#### SMOTE

SMOTE cree des exemples synthetiques de fraude dans l'espace des features.

Effet attendu :

- meilleur apprentissage de la classe minoritaire ;
- cout memoire et temps plus eleves ;
- risque d'exemples synthetiques peu realistes, surtout avec des variables categorielles encodees ordinalement.

Mon avis : SMOTE est pedagogiquement interessant, mais je serais prudent pour une recommandation production. Sur des donnees transactionnelles avec categories encodees, les exemples synthetiques peuvent etre moins coherents metier.

#### SMOTEENN

SMOTEENN combine sur-echantillonnage et nettoyage des frontieres de classe.

Effet attendu :

- peut ameliorer la separation fraude/legitime ;
- souvent couteux ;
- peut supprimer des exemples utiles ;
- resultats parfois instables.

Mon avis : a tester, mais pas mon premier choix pour un systeme MLOps simple. Si le gain AUPRC n'est pas net, la complexite supplementaire ne vaut probablement pas le cout.

## 3. Seuil de decision

Le seuil 0.5 est arbitraire. En detection de fraude, le modele produit un score de risque ; la decision de bloquer ou d'envoyer a un analyste doit dependre d'un seuil metier.

Resultats a completer :

| Seuil | Precision | Recall | F1 | Taux d'alertes | Nombre d'alertes |
|---:|---:|---:|---:|---:|---:|
| 0.05 | `...` | `...` | `...` | `...` | `...` |
| 0.10 | `...` | `...` | `...` | `...` | `...` |
| 0.20 | `...` | `...` | `...` | `...` | `...` |
| 0.50 | `...` | `...` | `...` | `...` | `...` |

Interpretation : un seuil plus bas augmente le recall mais cree plus d'alertes. Un seuil plus haut reduit la charge analyste mais laisse passer plus de fraudes.

Mon avis : la recommandation finale ne doit pas etre seulement "le meilleur modele". Elle doit etre "le meilleur modele + un seuil compatible avec la capacite operationnelle".

## 4. Recommandation provisoire

Sans les valeurs exactes sauvegardees, la recommandation doit rester conditionnelle.

Si `scale_pos_weight` obtient une AUPRC proche ou superieure aux autres strategies, je recommanderais cette approche pour PayTrack. Elle est simple, robuste, rapide a entrainer et facile a deployer.

Si SMOTE ou SMOTEENN gagnent legerement en recall mais augmentent fortement le taux d'alertes ou le temps d'entrainement, je ne les recommanderais pas en premiere version production.

Si Random Undersampling donne un tres haut recall mais une precision faible, il peut etre utile pour un systeme d'alerte large, mais risque de saturer les analystes et de bloquer trop de clients legitimes.

Recommandation a completer apres execution :

> Le modele recommande est `...`, avec un seuil initial de `...`, car il offre le meilleur compromis entre AUPRC, recall fraude, precision et taux d'alertes. Le modele devra etre surveille dans le temps via un monitoring de drift et re-entraine si l'AUPRC chute ou si les distributions transactionnelles evoluent fortement.

## 5. Limites actuelles

Les notebooks constituent une bonne base, mais plusieurs limites doivent etre explicitees :

- Le `customer_proxy` n'est pas un vrai identifiant client.
- Le mode `compact` exclut les colonnes `Vxxx`, donc il peut sous-exploiter le dataset complet.
- Les variables categorielles sont encodees ordinalement, ce qui est pratique pour XGBoost mais imparfait pour SMOTE.
- Les resultats dependent du split temporel choisi.
- Les couts metier exacts des faux positifs et faux negatifs ne sont pas encore quantifies.

## 6. Ce que je pense des resultats attendus

La demarche est solide : elle part d'une EDA qui montre le desequilibre, puis compare plusieurs strategies avec des metriques adaptees. C'est exactement ce qu'il faut pour repondre au CDO.

Le point le plus important est de ne pas survaloriser une strategie uniquement parce qu'elle augmente le recall. En fraude, un recall eleve est utile seulement si le nombre d'alertes reste exploitable. Sinon, le modele deplace le probleme vers les analystes.

Mon avis technique : pour une premiere version industrialisable, je privilegierais probablement **XGBoost avec `scale_pos_weight`**, puis un ajustement fin du seuil de decision. C'est generalement plus maintenable que SMOTE/SMOTEENN, plus simple a tracer dans MLflow, et plus facile a expliquer a une equipe MLOps.

## 7. Prochaines etapes

1. Executer les deux notebooks et sauvegarder les outputs.
2. Completer les tableaux de ce document avec les vraies valeurs.
3. Choisir un modele candidat selon AUPRC, recall et taux d'alertes.
4. Ajouter SHAP pour expliquer les decisions.
5. Passer au notebook MLOps : MLflow, registry, serving et monitoring Evidently.
6. Preparer la reponse aux quatre questions du CDO : performance, degradation, explicabilite, deploiement.
