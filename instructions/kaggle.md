# Compétition Kaggle — Qui utilise mon logiciel ?

> **Objectif :** prédire l'utilisateur d'un logiciel à partir d'un jeu de traces d'utilisation.

| Début        | Clôture            |
| ------------ | ------------------ |
| il y a 1 jour | dans 3 mois        |

---

## Description

**Pouvez-vous prédire qui utilise mon logiciel ?**

La société **Infologic** édite et intègre des solutions logicielles pour l'agroalimentaire depuis 1982. L'une des clés de l'évolution et de la maintenance de ces logiciels repose sur les **tests continus** effectués par des équipes internes.

Ces testeurs utilisent divers profils pour tester différentes fonctionnalités. Pour chaque problème rencontré, ils doivent reporter leur utilisation du logiciel à partir de leur vrai profil, ce qui les oblige à effectuer des manipulations parfois longues et rébarbatives.

C'est pourquoi nous proposons d'étudier s'il est possible d'**identifier un utilisateur à partir de traces d'utilisation du logiciel** (actions effectuées sur le logiciel) en construisant un **modèle de classification** utilisant des techniques de machine learning.

D'autre part, un tel modèle permettrait de **diagnostiquer des tentatives d'intrusion logicielle**, par exemple lorsqu'un pirate tente de se faire passer pour un utilisateur réel en se connectant avec son compte.

### Ce qu'il faut faire

1. Partir du vaste ensemble de données de traces logicielles fourni.
2. Construire des *features* à partir de ces traces.
3. Entraîner un modèle de classification.
4. Appliquer le modèle aux données de test et **soumettre le résultat**.

Bonne chance et bon apprentissage !

---

## Évaluation

La métrique d'évaluation est le **F1 score moyen**.

Le F1 score, couramment utilisé en recherche d'information, combine la **précision** $p$ et le **rappel** $r$ :

- **Précision** — part des vrais positifs parmi tous les positifs prédits :

  $$p = \frac{tp}{tp + fp}$$

- **Rappel** (*recall*) — part des vrais positifs parmi tous les positifs réels :

  $$r = \frac{tp}{tp + fn}$$

- **Score F1** :

  $$F_1 = 2 \cdot \frac{p \cdot r}{p + r}$$

La métrique F1 pondère le rappel et la précision **de manière égale** : un bon algorithme maximisera les deux simultanément. Une performance modérément bonne sur les deux sera donc préférée à une performance excellente sur l'une et médiocre sur l'autre.
