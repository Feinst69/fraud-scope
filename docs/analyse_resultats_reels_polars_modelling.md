# Analyse des resultats reels - Exploration et modelisation Polars

## Contexte

Cette analyse s'appuie sur les sorties sauvegardees dans :

- `notebooks/01_exploration_polars.ipynb`
- `notebooks/02_modelling.ipynb`

Les resultats ont ete lus directement depuis les tableaux et plots des notebooks.

## 1. Resultats de l'exploration

### Volume et desequilibre de classes

Le dataset fusionne contient **590 540 transactions** et **434 colonnes** dans le notebook d'exploration.

| Classe | Nombre | Taux |
|---|---:|---:|
| Legitime | 569 877 | 96.50 % |
| Fraude | 20 663 | 3.50 % |

L'accuracy d'un modele qui predit toujours `legitime` est donc **96.50 %**.

Analyse : c'est le point le plus important de l'EDA. Une accuracy autour de 96 % ne signifie presque rien ici, car elle peut etre obtenue sans detecter aucune fraude. Le projet doit donc etre pilote avec l'AUPRC, le recall fraude, la precision et le taux d'alertes.

Mon avis : cette partie est bien demontree. Elle donne un argument solide pour expliquer au CDO pourquoi les metriques classiques peuvent mentir dans un contexte anti-fraude.

## 2. Analyse temporelle

Le dataset couvre les jours relatifs **1 a 182**. Le split temporel utilise dans les notebooks est coherent :

| Split | Lignes | Jours | Taux fraude |
|---|---:|---|---:|
| Train | 472 432 | 1 -> 141 | 3.5135 % |
| Validation | 118 108 | 141 -> 182 | 3.4409 % |

Le taux de fraude reste proche entre train et validation, ce qui est positif pour une premiere evaluation. Il y a quand meme une variabilite journaliere visible sur le plot : certains jours descendent autour de 1-2 %, tandis que d'autres montent vers 6-7 %.

Le plot horaire montre une concentration nette du risque sur certaines heures relatives :

| Heure relative | Taux de fraude approx. |
|---:|---:|
| 5 | 7.03 % |
| 6 | 7.77 % |
| 7 | 10.61 % |
| 8 | 9.30 % |
| 9 | 9.00 % |

Analyse : le comportement frauduleux n'est pas uniforme dans le temps. Les heures 5 a 9 ont un taux de fraude beaucoup plus eleve que la moyenne globale de 3.5 %. A l'inverse, les heures 13 a 15 sont plus faibles, autour de 2.3 % a 2.5 %.

Mon avis : les features temporelles sont utiles. Le split temporel est justifie, et `hour_of_day_rel`, `day_rel`, `week_rel` doivent rester dans les modeles.

## 3. Montants

| Classe | Moyenne | Mediane | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Legitime | 134.51 | 68.50 | 267.08 | 435.00 | 1 104.00 | 31 937.39 |
| Fraude | 149.24 | 75.00 | 335.00 | 500.00 | 994.00 | 5 191.00 |

Les fraudes ont une moyenne, une mediane, un P90 et un P95 plus eleves que les transactions legitimes. En revanche, le P99 legitime est plus eleve, probablement a cause de transactions legitimes tres grosses.

Le plot `log(1 + montant)` montre un fort chevauchement entre fraude et legitime. Les montants seuls ne suffisent donc pas a separer les classes.

Mon avis : `TransactionAmt` est utile, mais pas decisif seul. Les ratios comportementaux, comme montant actuel / moyenne historique client, sont plus interessants que le montant brut.

## 4. Variables categorielles

Certaines categories ont des taux de fraude nettement superieurs a la moyenne.

### ProductCD

| ProductCD | Transactions | Taux fraude |
|---|---:|---:|
| C | 68 519 | 11.69 % |
| S | 11 628 | 5.90 % |
| H | 33 024 | 4.77 % |
| R | 37 699 | 3.78 % |
| W | 439 670 | 2.04 % |

`ProductCD=C` est tres au-dessus de la moyenne globale. `W`, tres majoritaire, est beaucoup moins risque.

### Type de carte

| Variable | Modalite | Taux fraude |
|---|---|---:|
| card4 | discover | 7.73 % |
| card4 | visa | 3.48 % |
| card4 | mastercard | 3.43 % |
| card6 | credit | 6.68 % |
| card6 | debit | 2.43 % |

Les cartes credit sont nettement plus risquees que les cartes debit dans ce dataset.

### Email et device

| Variable | Modalite | Taux fraude |
|---|---|---:|
| P_emaildomain | mail.com | 18.96 % |
| P_emaildomain | outlook.com | 9.46 % |
| R_emaildomain | outlook.com | 16.51 % |
| R_emaildomain | icloud.com | 12.88 % |
| R_emaildomain | gmail.com | 11.92 % |
| DeviceType | mobile | 10.17 % |
| DeviceType | desktop | 6.52 % |
| DeviceType | manquant | 2.10 % |

Analyse : les variables email et device portent un signal fort. `DeviceType=mobile` est environ trois fois plus risque que la moyenne globale. Les domaines receveurs comme `outlook.com`, `icloud.com` et `gmail.com` sont aussi tres discriminants.

Mon avis : ces variables doivent rester dans le modele, mais il faut surveiller leur stabilite en production. Les fraudeurs peuvent changer rapidement de domaine email ou de device.

## 5. Valeurs manquantes

Les valeurs manquantes sont massives pour certaines colonnes :

| Feature | Taux manquant |
|---|---:|
| id_24 | 99.20 % |
| id_25 | 99.13 % |
| id_07 | 99.13 % |
| id_08 | 99.13 % |
| dist2 | 93.63 % |
| D7 | 93.41 % |
| id_18 | 92.36 % |
| D13 | 89.51 % |

Analyse : beaucoup de colonnes identity/device sont presque vides. Les supprimer toutes serait tentant, mais l'absence d'information peut elle-meme etre informative.

Mon avis : pour la version compact du notebook, le choix d'ecarter les colonnes tres manquantes est pragmatique afin d'eviter les crashes memoire. Pour une version plus performante, il faudrait tester un mode `wide` controle avec XGBoost, car XGBoost gere assez bien les valeurs manquantes.

## 6. Correlations avec la cible

Les plus fortes correlations avec `isFraud` sont principalement des variables anonymisees `Vxxx` :

| Feature | Correlation avec isFraud |
|---|---:|
| V257 | 0.3831 |
| V246 | 0.3669 |
| V244 | 0.3641 |
| V242 | 0.3606 |
| V201 | 0.3280 |
| V200 | 0.3188 |
| V189 | 0.3082 |
| V188 | 0.3036 |

Analyse : les colonnes `Vxxx` contiennent clairement du signal predictif. Le modele compact, qui les exclut, est donc probablement sous-optimal.

Mon avis : pour eviter les problemes memoire, le mode compact est raisonnable. Mais pour viser une meilleure performance finale, il faudra reintegrer une selection limitee des meilleures `Vxxx`, par exemple les 20 a 50 variables les plus correlees ou les plus importantes selon XGBoost.

## 7. Baseline logistique de l'EDA

La regression logistique naive donne :

| Metrique | Valeur |
|---|---:|
| Accuracy | 0.9662 |
| Recall fraude | 0.0529 |
| F1 | 0.0972 |
| AUPRC | 0.2269 |

Classification report :

| Classe | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Legitime | 0.97 | 1.00 | 0.98 | 114 044 |
| Fraude | 0.59 | 0.05 | 0.10 | 4 064 |

Analyse : c'est exactement le piege attendu. L'accuracy est tres haute, mais le modele ne detecte que **5.29 % des fraudes**.

Mon avis : cette baseline est utile pedagogiquement. Elle montre que l'accuracy est inutilisable seule et justifie le passage a XGBoost.

## 8. Resultats XGBoost et resampling

Le notebook modelling utilise le mode compact :

- 53 features au total ;
- 34 numeriques ;
- 15 categorielles ;
- 4 features comportementales ;
- train limite a 150 000 lignes pour les strategies de resampling ;
- `scale_pos_weight = 27.46`.

### Tableau comparatif

| Strategie | AUPRC | Precision@0.5 | Recall@0.5 | F1@0.5 | Taux alertes | Temps train |
|---|---:|---:|---:|---:|---:|---:|
| xgboost_baseline | 0.4827 | 0.7888 | 0.2913 | 0.4255 | 1.27 % | 17.72 s |
| xgboost_scale_pos_weight | 0.4522 | 0.1559 | 0.7618 | 0.2589 | 16.81 % | 9.30 s |
| random_undersampling | 0.4478 | 0.1588 | 0.7495 | 0.2621 | 16.24 % | 0.89 s |
| smote | 0.4219 | 0.3762 | 0.4719 | 0.4187 | 4.32 % | 5.98 s |
| smoteenn | 0.4186 | 0.3537 | 0.4946 | 0.4125 | 4.81 % | 183.98 s |

### Lecture des resultats

Le meilleur modele en AUPRC est **xgboost_baseline** avec **0.4827**. C'est aussi le meilleur F1 au seuil 0.5 avec **0.4255**.

Les strategies `scale_pos_weight` et `random_undersampling` augmentent fortement le recall, autour de 75-76 %, mais elles declenchent environ 16 % d'alertes. C'est beaucoup trop eleve pour un systeme operationnel si chaque alerte doit etre traitee par un analyste.

SMOTE et SMOTEENN sont plus equilibres que `scale_pos_weight` en taux d'alertes, mais leur AUPRC est inferieure a la baseline. SMOTEENN est surtout tres couteux : environ **184 secondes**, sans gain de performance.

Mon avis : sur ces resultats, je ne recommanderais ni SMOTE ni SMOTEENN pour la premiere version. Le gain metier n'est pas suffisant, et SMOTEENN ajoute une complexite forte.

## 9. Meilleur modele : matrice de confusion

Le meilleur modele par AUPRC est `xgboost_baseline`.

Matrice de confusion au seuil 0.5 :

|  | Pred legitime | Pred fraude |
|---|---:|---:|
| Vrai legitime | 113 727 | 317 |
| Vrai fraude | 2 880 | 1 184 |

Cela donne :

- vrais positifs fraude : 1 184 ;
- faux negatifs fraude : 2 880 ;
- faux positifs : 317 ;
- recall fraude : 29.13 % ;
- precision fraude : 78.88 %.

Analyse : au seuil 0.5, le modele est tres prudent. Quand il alerte, il se trompe peu, mais il laisse passer beaucoup de fraudes.

Mon avis : ce modele est bon comme score de risque, mais le seuil 0.5 est trop conservateur si l'objectif est de reduire fortement les chargebacks.

## 10. Analyse du seuil

Pour `xgboost_baseline`, le seuil change fortement le compromis precision / recall.

| Seuil | Precision | Recall | F1 | Taux alertes | Alertes |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.1719 | 0.7330 | 0.2785 | 14.67 % | 17 327 |
| 0.10 | 0.3173 | 0.5851 | 0.4115 | 6.35 % | 7 495 |
| 0.15 | 0.4263 | 0.5079 | 0.4635 | 4.10 % | 4 842 |
| 0.20 | 0.5148 | 0.4582 | 0.4848 | 3.06 % | 3 617 |
| 0.25 | 0.5870 | 0.4242 | 0.4925 | 2.49 % | 2 937 |
| 0.30 | 0.6401 | 0.3952 | 0.4887 | 2.12 % | 2 509 |
| 0.50 | 0.7888 | 0.2913 | 0.4255 | 1.27 % | 1 501 |

Le F1 est maximal autour du seuil **0.25**, avec **F1 = 0.4925**. Ce seuil detecte **42.42 %** des fraudes avec une precision de **58.70 %** et un taux d'alertes de **2.49 %**.

Analyse : le seuil 0.25 semble etre un meilleur compromis que 0.5. Il double presque le volume d'alertes par rapport a 0.5, mais augmente sensiblement le recall tout en gardant une precision correcte.

Mon avis : si PayTrack a une capacite analyste suffisante, je recommanderais de demarrer autour d'un seuil **0.20 a 0.25**, pas 0.5. Si la priorite est de limiter fortement les faux positifs, alors 0.5 reste defensible mais manque trop de fraudes.

## 11. Recommandation finale

Sur les resultats actuels, je recommande :

> **XGBoost baseline compact avec ajustement du seuil autour de 0.20-0.25.**

Raison :

- meilleure AUPRC globale : 0.4827 ;
- meilleur F1 parmi les strategies testees au seuil 0.5 ;
- precision elevee au seuil 0.5 ;
- courbe Precision-Recall meilleure que les autres strategies sur une grande partie des recalls utiles ;
- seuil ajustable pour obtenir plus de recall sans utiliser de resampling complexe.

Je ne recommande pas `scale_pos_weight` tel quel au seuil 0.5, meme si son recall est fort, car il alerte **16.81 %** des transactions. Pour 118 108 transactions de validation, cela represente presque 20 000 alertes. C'est probablement trop pour une equipe d'analystes.

Je ne recommande pas SMOTEENN : il est le plus lent et n'ameliore pas l'AUPRC.

## 12. Limites et ameliorations

Les resultats sont bons pour une premiere version, mais il y a trois limites importantes.

### 1. Mode compact

Le modele exclut les colonnes `Vxxx`, alors que l'EDA montre que plusieurs d'entre elles sont fortement correlees avec la fraude. Il faut tester une version intermediaire : compact + top 30 ou top 50 `Vxxx`.

### 2. Seuil metier

Le seuil optimal ne doit pas etre choisi uniquement par F1. Il doit dependre :

- du cout moyen d'un faux negatif ;
- du cout d'un faux positif ;
- de la capacite journaliere des analystes ;
- du niveau de friction client acceptable.

### 3. Features comportementales

Les features historiques montrent un signal :

| Classe | Tx precedentes moyenne | Ratio montant / moyenne historique |
|---|---:|---:|
| Legitime | 374.61 | 1.1429 |
| Fraude | 476.45 | 1.2750 |

Les fraudes ont en moyenne plus d'historique client et un ratio montant/historique plus eleve. Ces features sont pertinentes, mais le `customer_proxy` reste une approximation.

## 13. Conclusion CDO

Le modele ML fait mieux qu'une baseline naive : la regression logistique avait une AUPRC de **0.2269**, alors que XGBoost baseline atteint **0.4827**.

Le modele XGBoost baseline est le meilleur candidat actuel. Il ne faut cependant pas le deployer avec le seuil par defaut 0.5 sans discussion metier. Le seuil **0.20-0.25** semble plus pertinent si PayTrack veut detecter davantage de fraudes tout en gardant un taux d'alertes raisonnable.

La prochaine etape doit etre :

1. tester XGBoost avec une selection de colonnes `Vxxx` ;
2. ajouter SHAP pour expliquer les decisions ;
3. logger les runs dans MLflow ;
4. surveiller drift et degradation via Evidently ;
5. definir un seuil d'alerte avec les contraintes operationnelles de PayTrack.
