# Étape 2 — Exploration des données

Code : [`src/02_exploration.py`](../src/02_exploration.py), qui s'appuie sur
[`src/traces.py`](../src/traces.py) pour lire les jetons.
Lancement : `python src/02_exploration.py` (après l'étape 1).

## 1. Objectif

Comprendre les données **avant** de construire des features et des modèles :

- à quoi ressemble la cible et à quel point les classes sont déséquilibrées ;
- quelles informations des traces semblent caractériser un utilisateur ;
- quelles anomalies devront être traitées au pré-traitement ;
- si le jeu de test ressemble au jeu d'entraînement, ce qui conditionne la fiabilité de la
  validation croisée.

## 2. Entrées / sorties

| | Fichier | Contenu |
|---|---|---|
| Entrée | `outputs/sessions_train.pkl`, `outputs/sessions_test.pkl` | Sessions de l'étape 1 |
| Sortie | `outputs/figures/fig01…fig07_*.png` | 7 figures pour le rapport (voir §5) |
| Sortie | `outputs/tableaux/*.csv` (UTF-8) | Sessions par utilisateur, fréquence des actions, utilisateurs par écran, statistiques descriptives des sessions |
| Sortie | Console | Chiffres clés repris ci-dessous |

## 3. Démarche pas à pas

1. **Lecture d'un jeton** (`traces.py`) :
   - `est_marqueur_temps` reconnaît les marqueurs `tXX`, et `valeur_marqueur` en extrait le
     nombre de secondes ;
   - `decomposer_action` découpe une action en `action` (libellé), `ecran` `(…)`,
     `conf` `<…>`, `chaine` `$…$` et `modif` (le `1` final), à l'aide d'une seule
     expression régulière.
2. **Résumé de chaque session** (`resumer_session`, `construire_profils`) :
   - nombre d'actions ;
   - durée (valeur du dernier marqueur) ;
   - vitesse (actions par minute) ;
   - nombre d'écrans distincts ;
   - part d'actions en mode modification ;
   - nombre de marqueurs répétés et de jetons illisibles.
3. **Agrégats** :
   - `compter_actions` : fréquence des 35 libellés ;
   - `compter_utilisateurs_par_ecran` : nombre d'utilisateurs distincts ayant visité chaque écran ;
   - `calculer_signatures` : part moyenne de chaque action par utilisateur.
4. **Figures** (`tracer_*`) et **tableaux** (`sauvegarder_tableaux`), puis affichage des
   chiffres clés (`afficher_chiffres_cles`).

## 4. Choix et justifications

- **Une regex unique pour décomposer les actions**, inspirée de celles du notebook
  d'indications mais ancrée sur tout le jeton (`fullmatch`). Les groupes non gourmands
  `(.*?)` permettent de lire correctement les 8 jetons dont l'écran contient des parenthèses
  imbriquées. Ce sont des traces d'erreur Java comme
  `Démarrage serveur…(…loadStatList (StatSelectionPart.java:514))`, sur lesquelles une regex
  naïve s'arrêtait trop tôt.
- **Le « 1 » final n'est retiré que s'il est en fin de jeton.** La fonction `filter_action`
  du notebook d'indications coupe au premier `1` rencontré, ce qui est fragile si un libellé
  ou un écran contient ce chiffre.
- **Cache (`lru_cache`) sur `decomposer_action`.** Il y a 2,3 millions d'actions, mais beaucoup
  de jetons se répètent. Le cache rend l'exploration rapide (quelques secondes) sans
  structure de données supplémentaire.
- **Indicateurs choisis.** On mesure ce qui peut distinguer des personnes :
  - le *rythme* de travail (vitesse) ;
  - le *contenu* du travail (écrans, actions) ;
  - les *habitudes* (mode modification).

  Ce sont les familles de features prévues à l'étape 4.
- **Échelles logarithmiques** pour le nombre d'actions et la durée, dont les distributions
  sont très étalées (de quelques secondes à plus de 3 h). On trace la *part des sessions* par
  classe, et non une densité : avec des classes de largeur variable, une densité gonflerait
  artificiellement les petites valeurs.
- **Boxplots sur les 6 utilisateurs les plus actifs** plutôt que sur des codes de classe
  arbitraires, pour avoir assez de sessions par boîte.

## 5. Résultats / observations

### Cible (fig01)

- 247 utilisateurs. Le nombre de sessions par utilisateur va de **4 à 75**, avec une médiane
  de 12, et **90 utilisateurs ont moins de 10 sessions**.
- Prédire toujours l'utilisateur le plus fréquent ne donnerait que **2,3 % d'accuracy**. C'est
  le plancher à battre.
- **Conséquences** :
  - les classes sont nombreuses, petites et déséquilibrées. On suivra le **F1 macro**, où
    chaque utilisateur pèse autant quel que soit son nombre de sessions (cf. plan d'évaluation) ;
  - la plus petite classe ayant 4 sessions, la validation croisée stratifiée sera limitée à
    **4 plis** (`N_SPLITS = 4`), pour que chaque utilisateur figure dans chaque pli de validation ;
  - **on ne supprimera aucune session** du train, car chaque exemple compte pour les petites classes.

### Navigateur (fig02)

- **Chacun des 247 utilisateurs n'utilise qu'un seul navigateur.**
- Les utilisateurs se répartissent ainsi : 114 sur Firefox, 93 sur Chrome, 39 sur Edge et
  **1 seul sur Opera**. Les 2 sessions Opera du test sont donc identifiables à coup sûr.
- **Conséquence** : le navigateur divise d'emblée le problème en sous-problèmes d'au plus
  114 classes. C'est une feature indispensable, à encoder en one-hot, puisqu'il n'y a pas
  d'ordre entre navigateurs.
- **Limite** : on suppose que cette règle reste vraie dans le test. Si un utilisateur
  changeait de navigateur, le modèle risquerait de se tromper systématiquement sur lui.
- La répartition des sessions par navigateur est quasi identique en train et en test
  (Firefox 44,7 % / 45,4 %, Chrome 40,8 % / 41,0 %).

### Sessions (fig03)

| Médiane | train | test |
|---|---|---|
| Nombre d'actions | 340 | 365 |
| Durée (s) | 395 (≈ 6,5 min) | 408 |
| Vitesse (actions / min) | 52,9 | 52,1 |
| Écrans distincts | 9 | 10 |
| Part en mode modification | 0,09 | 0,08 |

- Les distributions du train et du test **se superposent**. Le test semble tiré de la même
  population, donc les scores de validation croisée devraient bien prédire le score Kaggle.
- Les sessions durent de quelques secondes à **3,3 h**. Pour comparer des sessions de
  longueurs aussi différentes, il faudra des features **normalisées** (proportions, TF-IDF,
  vitesse) plutôt que des comptes bruts.

### Actions (fig04)

- **35 libellés** distincts, très inégalement fréquents. `Exécution d'un bouton` représente
  à lui seul 22,5 % des actions, suivi de `Saisie dans un champ` (11,7 %) et
  `Sélection d'un écran` (9,1 %). Certaines actions sont rares (`Erreur système grave` :
  15 occurrences).
- Les actions très fréquentes sont partagées par tout le monde et donc peu discriminantes,
  alors que les actions rares peuvent l'être beaucoup. C'est exactement ce que corrige la
  pondération **TF-IDF** prévue à l'étape 4.

### Écrans (fig05)

- 415 écrans distincts en train, dont **147 visités par un seul utilisateur** et 266 par
  5 utilisateurs au plus.
- Un écran correspond à un module métier (achats, comptabilité, CRM…), et chaque testeur
  travaille sur son périmètre. Notre **hypothèse** était donc que les écrans seraient la
  source d'information la plus discriminante. L'étape 4 l'a **démentie** : les écrans seuls
  donnent un F1 macro de 0,55, contre 0,82 pour la répartition des actions, car seules 13 %
  des actions portent un écran.
- Seuls 8 des 216 écrans présents dans le test n'apparaissent pas dans le train. Le
  vocabulaire appris en train couvre donc presque tout le test.

### Profils d'utilisateurs (fig06, fig07)

- Les **boîtes à moustaches** montrent des différences nettes entre utilisateurs. Par exemple,
  `cjr` est lent (≈ 33 actions/min) alors que `hjs` est rapide (≈ 75 actions/min). Mais la
  **variabilité intra-utilisateur est forte** : la part en mode modification de `skm` va de
  0 à 0,9. Ces indicateurs aideront, mais ne suffiront pas seuls.
- La **carte de chaleur** des actions fait apparaître des signatures. Quelques exemples :
  - `hjs` utilise beaucoup le `Chainage` et la grille d'historique de recherche ;
  - `bgn` déclenche beaucoup d'`Affichage d'un toast` ;
  - `muz` utilise beaucoup le `Retour sur un écran`.

  Cela justifie des features de **répartition des actions** par session.

### Anomalies à traiter à l'étape 3

- **Marqueurs répétés** (`t40,t40`) : 24 887 en train, présents dans 2 158 sessions (66 %), et
  2 636 en test. Une vérification complémentaire montre que deux marqueurs identiques sont
  **toujours séparés par des actions** (jamais `t40,t40` collés). Elles appartiennent donc à
  la même fenêtre de 5 s. Au pré-traitement, chaque action sera rattachée à la valeur du
  marqueur qui la suit, ce qui rend les répétitions sans effet.
- **Pas de temps morts dans les traces** : deux marqueurs distincts consécutifs sont
  **toujours** espacés de 5 s (sauf 53 cas sur 489 520), et les fenêtres sans action sont
  quasi inexistantes (5 sur 514 407). Les périodes d'inactivité ne sont donc pas enregistrées
  et la « durée » mesurée est une **durée d'activité**. Conséquence : il est inutile de
  construire des features de pause (on l'a d'abord tenté à l'étape 4, voir le carnet d'essais).
- **1 jeton illisible** en train, un marqueur collé à une action :
  `Création d'un écran(…AccueilController)t5`. Il sera séparé en deux jetons.
- **Sessions très courtes** : 3 en train et 1 en test comptent moins de 10 actions. On les
  conserve, car les supprimer du test est impossible (il faut une prédiction par ligne) et
  les supprimer du train retirerait des exemples à de petites classes.

## 6. Limites et pistes d'amélioration

- **La durée est estimée par défaut** : les actions situées après le dernier marqueur ont
  pu durer jusqu'à 5 s de plus.
- **Exploration limitée aux utilisateurs les plus actifs** pour les boxplots et la carte de
  chaleur. Les petites classes, les plus difficiles, sont moins bien décrites.
- **Information séquentielle non étudiée** : l'ordre des actions (enchaînements typiques,
  premier écran ouvert) n'a pas encore été regardé. On l'étudiera à l'étape 4 avec des
  bigrammes d'actions.
- **Configurations `<…>` et chaînes `$…$` non analysées en détail.** On pourrait refaire
  l'analyse de spécificité faite pour les écrans.
