# Étape 7 — Prédiction et soumission Kaggle

Code : [`src/07_soumission.py`](../src/07_soumission.py).
Lancement : `python src/07_soumission.py` (après l'étape 5, quelques secondes).

## 1. Objectif

Appliquer le modèle retenu aux 324 sessions du jeu de test et produire le fichier à déposer
sur Kaggle, en vérifiant qu'il respecte exactement le format attendu.

## 2. Entrées / sorties

| | Fichier | Contenu |
|---|---|---|
| Entrée | `outputs/modele_final.joblib` | Pipeline complet (TF-IDF + SVM linéaire sur séquences d'actions + contrainte navigateur) entraîné sur les 3 279 sessions du train |
| Entrée | `outputs/features_test.pkl` | Features du test (étape 4), avec la colonne `RowId` |
| Entrée | `data/sample_submission.csv` | Format de référence |
| Sortie | `outputs/submission.csv` | 324 lignes, colonnes `RowId` et `prediction` (trigramme), UTF-8 |

## 3. Démarche pas à pas

1. Chargement du modèle final et des features du test.
2. `predire_test` : le pipeline applique au test **exactement** les transformations apprises
   sur le train, puis le SVM et la contrainte navigateur prédisent un utilisateur par
   session. Les transformations concernées sont le one-hot du navigateur, ainsi que le
   vocabulaire et les poids IDF des deux TF-IDF.
3. `verifier_soumission` contrôle :
   - les noms de colonnes ;
   - l'ordre et les valeurs des `RowId` (identiques à `sample_submission.csv`) ;
   - que chaque utilisateur prédit existe dans le train.

   En cas de problème, le script s'arrête avec une `ValueError` explicite.
4. Écriture de `outputs/submission.csv`.

## 4. Choix et justifications

- **Le modèle est réentraîné sur tout le train avant de prédire le test.** La validation
  croisée a servi à *choisir* le modèle et ses réglages. Une fois ce choix fait, on utilise
  toutes les sessions disponibles. La courbe d'apprentissage (étape 6) montre que chaque
  session supplémentaire compte.
- **Un seul pipeline sauvegardé** : features et modèle sont ajustés ensemble à l'étape 5,
  puis rechargés ici. Il est ainsi impossible d'appliquer au test un pré-traitement
  différent de celui du train, ce qui serait une erreur classique.
- **Les étiquettes restent des trigrammes de bout en bout.** scikit-learn les gère
  directement, donc il n'y a pas de conversion codes ↔ trigrammes à refaire, alors que le
  notebook d'indications passe par `pd.Categorical.from_codes`.
- **Les features du test ne voient jamais le train**, à une exception près : les colonnes
  `part_action_*` reprennent la liste des libellés du train. Un libellé inconnu du train ne
  pourrait de toute façon pas être exploité par le modèle.

## 5. Résultats / observations

- **Contrôles** : 324 prédictions, au bon format, et tous les contrôles passent.
- **Diversité des prédictions** : 226 utilisateurs distincts sont prédits sur les 247 du
  train (218 avec la forêt de la première version). C'est cohérent avec un test de 324 sessions, puisque beaucoup d'utilisateurs n'y
  ont qu'une ou deux sessions, voire aucune.
- **Première version (forêt + contrainte navigateur)** : validation croisée 0,943, score
  Kaggle public **0,941**. L'accord est excellent : notre validation croisée est un bon
  estimateur du score sur le test, il n'y a ni sur-apprentissage du choix de modèle ni
  différence de distribution entre train et test (ce que l'étape 2 laissait attendre).
- **Version actuelle (SVM sur séquences + contrainte navigateur)** : validation croisée
  0,967 (4 plis), 0,974 (10 plis, plus proche du modèle final entraîné sur tout le train).
  Score attendu : environ 0,97. La nouvelle soumission diffère de la première sur
  12 sessions sur 324. Score Kaggle public : *à compléter après le dépôt*.
- **Ordre de grandeur du bruit** : le test ne compte que 324 sessions, et le classement
  public n'en utilise qu'une partie. Une seule erreur de plus ou de moins déplace le score
  d'environ 0,003 à 0,006 : des écarts de 0,01 entre deux soumissions ne sont pas
  significatifs.

## 6. Limites et pistes d'amélioration

- **Interprétation du « F1 moyen »** : la page Kaggle ne précise pas s'il est macro ou
  micro. Une soumission où toutes les lignes prédisent l'utilisateur le plus fréquent
  permettrait de trancher : environ 0,02 pour un F1 micro, environ 0,0001 pour un F1 macro.
- **Une seule soumission « finale »** : pour ne pas sur-ajuster nos choix au classement
  public, il vaut mieux limiter les allers-retours avec Kaggle et continuer à décider sur
  la validation croisée.
