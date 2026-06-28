# Rapport - Graph features avec NetworkX

## Objectif

Cette partie répond à la demande du sujet sur l'utilisation de graphes pour détecter des comportements de fraude en réseau :

- construire un graphe de **10 000 transactions** ;
- représenter les **clients** et les **marchands** comme des nœuds ;
- représenter les transactions comme des **arêtes** ;
- visualiser le graphe ;
- identifier un cluster suspect ;
- extraire des features graphe ;
- intégrer ces features dans XGBoost ;
- commenter le gain d'AUPRC.

Le travail est fait dans `notebooks/06_graph_features_networkx.ipynb`.

## Construction du graphe

Le graphe a été construit sur les **10 000 dernières transactions du train temporel**, juste avant la période de validation. Ce choix est cohérent avec le projet, car il rapproche l'analyse graphe de la période utilisée pour mesurer la performance du modèle.

Résultats du graphe :

| Élément | Valeur |
|---|---:|
| Transactions utilisées | 10 000 |
| Nœuds | 4 119 |
| Arêtes | 4 501 |
| Composantes connexes | 14 |
| Taux de fraude du sous-graphe | 3.12 % |

Le graphe est biparti : les nœuds `customer_proxy` représentent les clients, les nœuds `merchant_proxy` représentent les marchands, et chaque arête correspond à une ou plusieurs transactions entre un client et un marchand.

## Construction des proxys client et marchand

Le dataset IEEE-CIS ne contient pas de vrai identifiant client ni de vrai identifiant marchand. Pour construire un graphe exploitable, le notebook utilise donc deux identifiants approximatifs.

Le nœud client est construit ainsi :

```text
customer_proxy = card1 + card2 + card3 + card5 + addr1
```

Ces champs décrivent une combinaison carte/adresse. L'hypothèse est que deux transactions avec la même combinaison ont de bonnes chances d'appartenir au même client ou au même moyen de paiement.

Le nœud marchand est construit ainsi :

```text
merchant_proxy = ProductCD + R_emaildomain
```

Ce proxy marchand est plus faible qu'un vrai `merchant_id`, mais il donne une approximation du contexte marchand ou receveur.

Limite importante : ces proxys peuvent fusionner des entités différentes ou séparer une même entité en plusieurs nœuds. Les features graphe sont donc utiles pour un POC, mais en production il faudrait remplacer ces proxys par de vrais identifiants compte, carte tokenisée et marchand.

## Visualisation

La visualisation sauvegardée montre la structure du graphe et met en évidence les nœuds liés à au moins une fraude.

![Visualisation du graphe NetworkX](assets/06_graph_networkx_visualization.png)

Cette visualisation permet de voir que la majorité des transactions appartiennent à une grande composante connexe, avec quelques composantes isolées. Les nœuds rouges sont les nœuds touchés par au moins une transaction frauduleuse.

## Cluster suspect

Le notebook identifie les composantes connexes avec leur nombre de nœuds, d'arêtes, d'arêtes frauduleuses, taux de fraude et montant total.

La composante la plus frauduleuse en taux est la **composante 8** :

| Composante | Nœuds | Arêtes | Arêtes frauduleuses | Taux fraude | Montant total |
|---:|---:|---:|---:|---:|---:|
| 8 | 2 | 1 | 1 | 100.0 % | 325.00 |

Cette composante est suspecte parce que toute l'activité observée dans ce mini-cluster est frauduleuse. En revanche, elle est très petite : elle correspond plutôt à un signal isolé qu'à un réseau de fraude organisé.

La grande composante principale est aussi intéressante :

| Composante | Nœuds | Arêtes | Arêtes frauduleuses | Taux fraude | Montant total |
|---:|---:|---:|---:|---:|---:|
| 0 | 4 088 | 4 483 | 311 | 6.94 % | 1 396 263.77 |

Mon interprétation : pour une analyse métier, la composante 8 est le cluster le plus suspect en taux, mais la composante 0 est plus importante opérationnellement car elle concentre beaucoup plus de fraudes. C'est dans cette composante qu'un analyste chercherait probablement des sous-réseaux de marchands ou de clients récurrents.

Un zoom dédié a aussi été généré sur la composante suspecte non triviale la plus importante : la **composante 0**, qui concentre **311 arêtes frauduleuses** et un montant total de **1 396 263.77**. Son taux de fraude est de **6.94 %**, ce qui est élevé à cette échelle.

![Zoom sur le cluster suspect](assets/06_graph_suspect_cluster_zoom.png)

## Features graphe extraites

Quatre features ont été ajoutées au niveau transaction :

| Feature | Interprétation |
|---|---|
| `graph_customer_degree` | Nombre de marchands distincts connectés au client dans le graphe |
| `graph_customer_weighted_degree` | Volume pondéré de transactions du client dans le graphe |
| `graph_customer_betweenness` | Centralité du client, c'est-à-dire sa position de pont dans le réseau |
| `graph_distinct_merchants_prev_7d` | Nombre de marchands distincts vus par ce client dans les 7 jours précédents |

Résumé observé sur les 10 000 transactions :

| Feature | Moyenne | Médiane | Max |
|---|---:|---:|---:|
| `graph_distinct_merchants_prev_7d` | 0.936 | 1.000 | 8 |
| `graph_customer_degree` | 1.675 | 1.000 | 8 |
| `graph_customer_weighted_degree` | 14.482 | 4.000 | non affiché complètement dans la sortie |
| `graph_customer_betweenness` | 0.0004 | proche de 0 | faible |

Analyse : la plupart des clients ont peu de connexions marchands, mais certains clients ont un comportement plus dispersé. Ce type de signal est utile pour détecter des schémas où un même compte interagit avec plusieurs marchands ou sert de point de passage dans un réseau.

## Impact sur XGBoost

Le modèle a été réentraîné sur le sous-échantillon graphe avec un split temporel local : 8 000 transactions en train et 2 000 en validation.

| Modèle | Features | AUPRC | Precision | Recall | F1 | Taux d'alertes | Gain AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost base | 44 | 0.5570 | 0.7551 | 0.4805 | 0.5873 | 2.45 % | 0.0000 |
| XGBoost + graph features | 48 | 0.6096 | 0.7358 | 0.5065 | 0.6000 | 2.65 % | +0.0526 |

![Comparaison XGBoost avec et sans graph features](assets/06_graph_features_model_comparison.png)

Les graph features améliorent l'AUPRC de **0.5570 à 0.6096**, soit un gain absolu de **+0.0526**. Le recall augmente aussi de **48.05 % à 50.65 %**, avec un taux d'alertes seulement légèrement plus élevé : **2.45 % -> 2.65 %**.

La précision baisse légèrement, de **75.51 % à 73.58 %**, mais le F1 progresse de **0.5873 à 0.6000**. Le compromis est donc favorable : le modèle détecte un peu plus de fraudes sans exploser le nombre d'alertes.

## Conclusion

Cette partie est suffisante pour répondre à la demande du sujet sur les graph features avec NetworkX.

Le résultat principal est que les features graphe apportent un signal complémentaire au modèle tabulaire. Sur le sous-graphe de 10 000 transactions, elles améliorent l'AUPRC de **+0.0526**, ce qui est un gain significatif pour un ajout de seulement quatre variables.

Mon avis : ces features méritent d'être conservées dans une version enrichie du modèle. Elles ne remplacent pas les variables transactionnelles classiques, mais elles capturent une information différente : la structure relationnelle client-marchand. Pour un système antifraude, c'est important, car certaines fraudes se détectent mieux par les connexions entre entités que par le montant ou l'heure d'une transaction seule.

Limite importante : l'expérience est faite sur un sous-graphe de 10 000 transactions, pas sur tout le dataset. Le gain est donc encourageant, mais il faudrait refaire une validation plus large avant de conclure que le même gain sera stable en production.

## Ce qu'il resterait à faire pour aller plus loin

Pour renforcer cette partie, je recommande :

1. recalculer les features graphe sur des fenêtres glissantes plus réalistes ;
2. éviter toute fuite temporelle en ne construisant les features qu'avec les transactions passées ;
3. tester ces features dans le notebook MLflow comme version candidate `v2_graph_features` ;
4. comparer cette approche avec un vrai GNN si le temps le permet.

Pour le projet actuel, la réponse attendue sur NetworkX est couverte.
