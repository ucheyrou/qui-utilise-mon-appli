# Étape 6 — Évaluation

Code : [`src/06_evaluation.py`](../src/06_evaluation.py).
Lancement : `python src/06_evaluation.py` (après l'étape 5, environ 4 min).

## 1. Objectif

Mesurer la qualité des modèles **sous plusieurs angles**, et pas seulement avec la métrique
Kaggle :

- la qualité de la **décision** : quel utilisateur est prédit ;
- la qualité du **classement** : le vrai utilisateur est-il bien placé parmi les 247, ce
  que mesurent les courbes ROC ;
- une lecture **métier** : peut-on détecter une usurpation de compte ;
- un **diagnostic** du modèle retenu : quels utilisateurs posent problème, y a-t-il du
  sur-apprentissage, quelles variables comptent.

## 2. Entrées / sorties

| | Fichier | Contenu |
|---|---|---|
| Entrée | `outputs/scores_hors_pli.pkl` | Scores hors pli de chaque modèle (étape 5) |
| Entrée | `outputs/modele_final.joblib`, `outputs/features_train.pkl` | Modèle retenu et features (courbe d'apprentissage, importances) |
| Sortie | `outputs/tableaux/evaluation_modeles.csv` | 10 métriques par modèle |
| Sortie | `outputs/tableaux/points_de_fonctionnement.csv` | Lecture « détection d'intrusion » |
| Sortie | `outputs/tableaux/metriques_par_utilisateur.csv`, `confusions.csv` | F1 et AUC par utilisateur, couples confondus |
| Sortie | `outputs/tableaux/courbe_apprentissage.csv`, `importance_variables.csv` | Diagnostic du modèle retenu |
| Sortie | `outputs/figures/fig11` à `fig16` | ROC, ROC par utilisateur, métriques par utilisateur, précision-rappel, courbe d'apprentissage, importances |

## 3. Démarche pas à pas

1. `mesurer` calcule, pour chaque modèle, à partir de ses scores hors pli :
   - **métriques de décision**, sur l'utilisateur le plus probable : F1 macro, F1 micro,
     F1 pondéré, accuracy ;
   - **accuracy top-k** : le vrai utilisateur figure-t-il parmi les 3 ou 5 plus probables ?
   - **métriques de classement** : AUC ROC et précision moyenne, en moyennes macro et micro.
2. `tracer_roc_modeles` et `tracer_precision_rappel` superposent les courbes des 5 meilleurs
   modèles.
3. `points_de_fonctionnement` lit la ROC micro du modèle retenu à 0,1 % et 1 % de fausses
   alertes.
4. **Diagnostic par utilisateur** :
   - F1 et AUC de chaque utilisateur (`tracer_par_utilisateur`) ;
   - ROC des utilisateurs les plus difficiles (`tracer_roc_utilisateurs`) ;
   - couples les plus confondus (`lister_confusions`).
5. `tracer_courbe_apprentissage` réentraîne le modèle retenu sur 25, 50, 75 et 100 % des
   plis d'entraînement.
6. `extraire_importances` récupère l'importance de chaque variable dans le modèle final :
   diminution d'impureté pour une forêt, poids absolu moyen pour un modèle linéaire.

## 4. Choix et justifications

### Pourquoi évaluer sur des scores hors pli

Chaque session est notée par un modèle entraîné sans elle (validation croisée à 4 plis).
C'est la seule façon honnête d'évaluer sans sacrifier de données à un jeu de test interne,
précieux quand certains utilisateurs n'ont que 4 sessions.

Les métriques sont ici calculées **sur l'ensemble des 3 279 prédictions hors pli**, alors
que l'étape 5 fait la **moyenne des scores de chaque pli**. Le F1 macro « global » est
légèrement plus élevé (0,969 contre 0,967 pour le modèle retenu). Dans un pli, chaque utilisateur n'a que 1 à 4
sessions, ce qui rend son F1 très bruité et tire la moyenne vers le bas. Le score de
référence pour comparer les modèles reste celui de l'étape 5.

### Rappels de cours sur les métriques

| Métrique | Ce qu'elle mesure |
|---|---|
| **F1 d'un utilisateur** | Moyenne harmonique de la précision (parmi les sessions qu'on lui attribue, combien sont vraiment à lui) et du rappel (parmi ses sessions, combien on retrouve) |
| **F1 macro** | Moyenne des 247 F1 : chaque utilisateur pèse autant. Métrique Kaggle et critère de sélection |
| **F1 micro** | F1 sur toutes les décisions confondues ; en mono-étiquette, il est **égal à l'accuracy** |
| **F1 pondéré** | Moyenne des F1 pondérée par le nombre de sessions |
| **Accuracy top-k** | Le vrai utilisateur est-il parmi les k propositions les plus probables ? |
| **Précision moyenne** | Aire sous la courbe précision-rappel |

### La courbe ROC en multi-classes

Une courbe ROC est définie pour un problème **binaire**. On la construit en faisant varier
un seuil sur le score, puis en traçant :

- le **taux de vrais positifs** (rappel) ;
- en fonction du **taux de faux positifs** (part des négatifs acceptés à tort).

L'**AUC** est l'aire sous cette courbe : c'est la probabilité qu'une session positive
reçoive un score plus élevé qu'une session négative. Elle vaut 0,5 pour le hasard et 1 pour
un classement parfait.

Avec 247 utilisateurs, on se ramène au binaire par **« un contre tous »** : pour chaque
utilisateur k, le problème devient « session de k ou non ». Ensuite, deux façons de
résumer :

- **ROC macro-moyenne** : on trace la ROC de chaque utilisateur, on lit leur taux de vrais
  positifs sur une grille commune de taux de faux positifs, puis on fait la moyenne. Chaque
  utilisateur compte autant.
- **ROC micro-moyenne** : on met toutes les paires (session, utilisateur) dans un seul
  problème binaire, soit 3 279 × 247 ≈ 810 000 paires, dont 3 279 positives. Les scores de
  tous les utilisateurs sont comparés entre eux.

### L'axe logarithmique des faux positifs

Le problème « un contre tous » est très déséquilibré : environ 1 positif pour 246 négatifs.
Même un modèle moyen classe correctement l'immense majorité des négatifs, et **toutes les
AUC sont supérieures à 0,99**. Toute la différence entre modèles se joue aux très faibles
taux de faux positifs, qu'une échelle logarithmique permet de voir.

### Compléter la ROC par la précision-rappel

Pour la même raison, on trace aussi des courbes **précision-rappel**. La précision rapporte
les faux positifs au nombre de positifs prédits, et non à l'énorme nombre de négatifs :
elle est beaucoup plus sévère quand les positifs sont rares.

### Une lecture métier : la détection d'intrusion

La page Kaggle évoque l'usage du modèle pour repérer quelqu'un qui se connecte avec le
compte d'un autre. Le scénario est le suivant :

- une session prétend venir de l'utilisateur k ;
- on l'accepte si le score de k dépasse un seuil ;
- les paires « vraie session de k » sont les positifs, et les paires « session d'un autre
  qui se fait passer pour k » sont les négatifs.

C'est **exactement la ROC micro**. Choisir un seuil revient à arbitrer entre :

- les usurpations acceptées, c'est-à-dire les faux positifs ;
- les vrais utilisateurs bloqués à tort, c'est-à-dire 1 − le rappel.

### Courbe d'apprentissage et importances

- **Courbe d'apprentissage** : un écart persistant entre score d'entraînement et score de
  validation signale de la variance (sur-apprentissage). Une courbe de validation qui monte
  encore indique que plus de données aideraient.
- **Importances** : pour le SVM linéaire retenu, chaque utilisateur a un vecteur de poids ;
  l'importance d'une variable est la **moyenne sur les utilisateurs de la valeur absolue de
  son poids**. C'est légitime ici parce que les variables ont la même échelle (TF-IDF
  normés, navigateur en 0/1). Pour une forêt, on utiliserait la diminution moyenne
  d'impureté de Gini. L'importance par permutation serait plus fiable dans les deux cas,
  mais elle demanderait des centaines de prédictions sur 46 000 variables.

## 5. Résultats / observations

### Toutes les métriques (scores hors pli, ensemble des 3 279 sessions)

| Modèle | F1 macro | Accuracy (= F1 micro) | Top-3 | Top-5 | AUC macro | AUC micro | Préc. moy. micro |
|---|---|---|---|---|---|---|---|
| **SVM sur séquences + contrainte navigateur** | **0,969** | **0,971** | **0,992** | **0,995** | **0,9993** | **0,9996** | **0,985** |
| SVM sur séquences | 0,967 | 0,970 | 0,992 | 0,995 | 0,9992 | 0,9996 | 0,984 |
| Forêt + contrainte navigateur | 0,947 | 0,948 | 0,987 | 0,993 | 0,9992 | 0,9995 | 0,974 |
| Extra-Trees | 0,943 | 0,951 | 0,988 | 0,993 | 0,9985 | 0,9990 | 0,980 |
| Forêt aléatoire | 0,942 | 0,945 | 0,986 | 0,991 | 0,9989 | 0,9992 | 0,966 |
| Perceptron multicouche | 0,911 | 0,921 | 0,969 | 0,980 | 0,9979 | 0,9982 | 0,965 |
| SVM linéaire | 0,900 | 0,910 | 0,963 | 0,973 | 0,9945 | 0,9953 | 0,790 |
| Régression logistique | 0,863 | 0,871 | 0,939 | 0,959 | 0,9936 | 0,9942 | 0,917 |
| Gradient boosting | 0,793 | 0,821 | 0,913 | 0,934 | 0,9895 | 0,9919 | 0,874 |
| k plus proches voisins | 0,721 | 0,735 | 0,736 | 0,738 | 0,858 | 0,867 | 0,541 |
| Référence | 0,000 | 0,023 | 0,033 | 0,041 | 0,500 | 0,510 | 0,005 |

- **Le vrai utilisateur est parmi les 3 premières propositions dans 99,2 % des cas.** Les
  erreurs restantes sont donc « de justesse » : le bon utilisateur est presque toujours
  en deuxième ou troisième position.
- **Les AUC sont toutes saturées** (> 0,99 sauf pour le k-NN). Elles classent les modèles
  dans le même ordre que le F1, mais avec des écarts minuscules. C'est la limite annoncée
  de la ROC en « un contre tous » très déséquilibré. La **précision moyenne** sépare mieux
  les modèles (0,79 pour le SVM contre 0,97 pour la forêt).
- **Le SVM sur séquences est en tête sur toutes les métriques**, décision comme
  classement. Dans la première version (sans ce modèle), le meilleur modèle dépendait de
  la métrique : les Extra-Trees avaient la meilleure accuracy, la forêt contrainte le
  meilleur F1 macro. Le changement de représentation a donc plus apporté que le choix entre
  modèles.

### Courbes ROC (fig11) et précision-rappel (fig14)

- À **1 faux positif pour 10 000 paires** (10⁻⁴), le SVM sur séquences contraint retrouve
  **93,2 %** des vraies paires (ROC micro), contre 92,7 % sans contrainte et 86,5 % pour la
  forêt contrainte. La contrainte navigateur décale donc la courbe vers le haut.
- **Cas particulier du SVM sur les features E8** : sa ROC micro s'effondre aux faibles
  taux de faux positifs.
  Ses scores de décision ne sont **pas calibrés d'un utilisateur à l'autre** : un score de
  0,5 n'a pas le même sens pour deux utilisateurs différents. Or la ROC micro compare les
  scores de tous les utilisateurs entre eux. Sa ROC macro, qui traite chaque utilisateur
  séparément, est bien meilleure. Les probabilités des forêts et du MLP sont comparables
  entre utilisateurs et n'ont pas ce problème. Le SVM sur séquences a pourtant aussi des
  scores non calibrés, mais il sépare si nettement les utilisateurs que sa ROC micro reste
  la meilleure : l'effet existe, mais il est masqué.
- **Les paliers de la ROC macro** en dessous de 10⁻³ viennent du faible nombre de
  négatifs par utilisateur (≈ 3 200) : on ne peut pas mesurer un taux de faux positifs plus
  fin que 1/3 200 ≈ 3·10⁻⁴.

### Lecture « détection d'intrusion » (modèle retenu)

| Usurpations acceptées (faux positifs) | Sessions légitimes acceptées (rappel) | Seuil sur le score SVM |
|---|---|---|
| 0,1 % | **98,3 %** | −0,68 |
| 1 % | **99,5 %** | −0,87 |

Si l'on tolère qu'une tentative d'usurpation sur 1 000 passe, seules 1,7 % des vraies
sessions déclencheraient une fausse alerte (3,6 % avec la forêt de la première version).
Le seuil porte sur un score de décision et non sur une probabilité : il n'a de sens que
pour ce modèle. Ce chiffre n'est qu'indicatif : de vraies
usurpations, faites par quelqu'un qui *imite* l'utilisateur, seraient plus difficiles à
détecter que des sessions d'autres testeurs qui ne cherchent pas à tromper le modèle.

### Diagnostic par utilisateur (fig12, fig13)

- **153 utilisateurs sur 247 ont un F1 parfait (1,0)**, et **aucun** n'est sous 0,5. Les
  moins bien reconnus sont `lra` (F1 0,60, 4 sessions), `hll` (0,71, 9 sessions) et `itt`
  (0,75, 4 sessions). Avec la forêt, `hll` était à 0,11 et `zok` à 0,47.
- **Le F1 dépend peu du nombre de sessions** : le F1 moyen vaut 0,96 pour les utilisateurs à
  4-6 sessions et entre 0,97 et 0,98 au-delà. L'équilibrage des classes
  (`class_weight="balanced"`) protège les petits utilisateurs.
- **AUC et F1 ne désignent pas les mêmes utilisateurs difficiles.**
  - `itt` a une AUC de 1,000 mais un F1 de 0,75 : ses sessions sont toujours classées
    au-dessus de celles des autres *pour lui*, mais une de ses sessions a un autre
    utilisateur juste devant.
  - À l'inverse, `jkc` (AUC la plus basse, 0,950) a un F1 correct (0,80).

  L'AUC juge le *classement*, le F1 juge la *décision* : les deux se complètent.
- **Confusions principales** (79 couples, dont la plupart une seule fois) :
  - `skm` ↔ `slq` (5 + 3) et `flj` → `skm` / `slq` (2 + 2) : ce sont trois des utilisateurs
    les plus actifs, dont les boxplots de l'étape 2 montraient des profils proches (même
    vitesse, part en mode modification très variable). Ce trio reste le principal point
    faible, déjà présent avec la forêt ;
  - `esu` ↔ `lgd` (2 + 2), `ald` → `fuy`, `xkr` → `oqa`, `gvp` → `hrj` (2 chacun).
- **Les erreurs portent surtout sur des sessions courtes** : 130 actions en médiane pour
  les sessions mal classées, contre 340 pour l'ensemble. Une session courte contient peu de
  jetons caractéristiques.

  Il s'agit vraisemblablement de testeurs au périmètre similaire.

### Courbe d'apprentissage (fig15)

| Sessions d'entraînement | F1 macro entraînement | F1 macro validation |
|---|---|---|
| 614 | 1,00 | 0,723 |
| 1 229 | 1,00 | 0,889 |
| 1 844 | 1,00 | 0,946 |
| 2 459 | 1,00 | 0,967 |

- **Score d'entraînement de 1** : avec 46 000 variables pour 2 459 sessions, le SVM peut
  séparer parfaitement les sessions d'entraînement (il y a plus de dimensions que de
  points). Il généralise quand même grâce à la régularisation (C = 0,3 : marge large plutôt
  que zéro erreur à tout prix).
- **Le score de validation monte encore entre 75 % et 100 % des données** (+0,02). Une
  validation à 10 plis (90 % des données pour apprendre) le confirme : 0,974. **Plus de
  sessions par utilisateur amélioreraient le modèle** : c'est la limite principale, plus
  que le choix du modèle.

### Variables importantes (fig16)

- **Importances très « plates »** : la variable la plus importante ne pèse que 0,3 % du
  total. L'information est répartie sur des dizaines de milliers de jetons.
- **Par bloc** : les séquences d'actions complètes portent **71 %** de l'importance totale,
  les jetons par famille 28 % et le navigateur 1 %. Le bloc ajouté dans cette version est
  donc bien celui que le modèle utilise le plus.
- **En tête, variable par variable** : les trois navigateurs (normal, chacun sépare un
  groupe entier d'utilisateurs), puis des jetons par famille déjà repérés par l'étape 4 et
  par la forêt (`conf=INSTALLATEUR`, `Lancement d'un tableau de bord`, `Raccourci`,
  `Sélection d'un flag`, chaînes `GP`, `SV`, `EDI`, écran de planning). Les jetons de
  séquences, eux, sont pris un par un trop rares pour apparaître dans les 20 premiers : leur
  force vient de leur nombre.
- **Biais de la mesure** : un poids moyen élevé ne dit pas *pour quels* utilisateurs la
  variable compte, et deux jetons corrélés se partagent le poids. Il faut donc la lire avec
  prudence.

## 6. Limites et pistes d'amélioration

- **Pas de jeu de test interne indépendant** : le seul contrôle vraiment externe est le
  score Kaggle. Les scores hors pli restent légèrement optimistes, car les hyperparamètres
  ont été choisis sur les mêmes plis (voir l'étape 5).
- **Choix d'un seuil de rejet** : pour un usage réel en détection d'intrusion, il faudrait
  fixer le seuil sur un jeu dédié et tester de vraies tentatives d'usurpation, et non des
  sessions d'autres testeurs.
- **Importance par permutation** : la calculer par bloc de variables (séquences, jetons par
  famille, navigateur) serait plus fiable que les poids moyens, et moins coûteux qu'une
  permutation variable par variable.
- **Exploiter le top-3 (99,2 %)** : un second modèle spécialisé pour départager les 2 ou 3
  candidats les plus probables pourrait récupérer une partie des erreurs.
