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
6. `extraire_importances` récupère l'importance de chaque variable dans la forêt finale.

## 4. Choix et justifications

### Pourquoi évaluer sur des scores hors pli

Chaque session est notée par un modèle entraîné sans elle (validation croisée à 4 plis).
C'est la seule façon honnête d'évaluer sans sacrifier de données à un jeu de test interne,
précieux quand certains utilisateurs n'ont que 4 sessions.

Les métriques sont ici calculées **sur l'ensemble des 3 279 prédictions hors pli**, alors
que l'étape 5 fait la **moyenne des scores de chaque pli**. Le F1 macro « global » est
légèrement plus élevé (0,947 contre 0,943). Dans un pli, chaque utilisateur n'a que 1 à 4
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
- **Importances** : on utilise la diminution moyenne d'impureté de Gini, gratuite à
  calculer pour une forêt. L'importance par permutation serait plus fiable, mais elle
  demanderait des centaines de prédictions sur 1 800 variables.

## 5. Résultats / observations

### Toutes les métriques (scores hors pli, ensemble des 3 279 sessions)

| Modèle | F1 macro | Accuracy (= F1 micro) | Top-3 | Top-5 | AUC macro | AUC micro | Préc. moy. micro |
|---|---|---|---|---|---|---|---|
| **Forêt + contrainte navigateur** | **0,947** | 0,948 | 0,987 | 0,993 | **0,9992** | **0,9995** | 0,974 |
| Extra-Trees | 0,943 | **0,951** | **0,988** | 0,993 | 0,9985 | 0,9990 | **0,980** |
| Forêt aléatoire | 0,942 | 0,945 | 0,986 | 0,991 | 0,9989 | 0,9992 | 0,966 |
| Perceptron multicouche | 0,911 | 0,921 | 0,969 | 0,980 | 0,9979 | 0,9982 | 0,965 |
| SVM linéaire | 0,900 | 0,910 | 0,963 | 0,973 | 0,9945 | 0,9953 | 0,790 |
| Régression logistique | 0,863 | 0,871 | 0,939 | 0,959 | 0,9936 | 0,9942 | 0,917 |
| Gradient boosting | 0,793 | 0,821 | 0,913 | 0,934 | 0,9895 | 0,9919 | 0,874 |
| k plus proches voisins | 0,721 | 0,735 | 0,736 | 0,738 | 0,858 | 0,867 | 0,541 |
| Référence | 0,000 | 0,023 | 0,033 | 0,041 | 0,500 | 0,510 | 0,005 |

- **Le vrai utilisateur est parmi les 3 premières propositions dans 98,7 % des cas.** Les
  erreurs restantes sont donc « de justesse » : le bon utilisateur est presque toujours
  en deuxième ou troisième position.
- **Les AUC sont toutes saturées** (> 0,99 sauf pour le k-NN). Elles classent les modèles
  dans le même ordre que le F1, mais avec des écarts minuscules. C'est la limite annoncée
  de la ROC en « un contre tous » très déséquilibré. La **précision moyenne** sépare mieux
  les modèles (0,79 pour le SVM contre 0,97 pour la forêt).
- **Selon la métrique, le meilleur modèle n'est pas le même.** Les Extra-Trees ont la
  meilleure accuracy et la meilleure précision moyenne : ils *classent* mieux en moyenne.
  La forêt contrainte a le meilleur F1 macro : elle *décide* mieux pour les petits
  utilisateurs. Notre critère étant le F1 macro, on garde la forêt.

### Courbes ROC (fig11) et précision-rappel (fig14)

- À **1 faux positif pour 10 000 paires** (10⁻⁴), la forêt contrainte retrouve **86,5 %**
  des vraies paires (ROC micro), contre 84,2 % pour la forêt seule. La contrainte
  navigateur décale donc la courbe vers le haut.
- **Cas particulier du SVM** : sa ROC micro s'effondre aux faibles taux de faux positifs.
  Ses scores de décision ne sont **pas calibrés d'un utilisateur à l'autre** : un score de
  0,5 n'a pas le même sens pour deux utilisateurs différents. Or la ROC micro compare les
  scores de tous les utilisateurs entre eux. Sa ROC macro, qui traite chaque utilisateur
  séparément, est bien meilleure. Les probabilités des forêts et du MLP sont comparables
  entre utilisateurs et n'ont pas ce problème.
- **Les paliers de la ROC macro** en dessous de 10⁻³ viennent du faible nombre de
  négatifs par utilisateur (≈ 3 200) : on ne peut pas mesurer un taux de faux positifs plus
  fin que 1/3 200 ≈ 3·10⁻⁴.

### Lecture « détection d'intrusion » (modèle retenu)

| Usurpations acceptées (faux positifs) | Sessions légitimes acceptées (rappel) | Seuil sur la probabilité |
|---|---|---|
| 0,1 % | **96,4 %** | 0,147 |
| 1 % | **99,5 %** | 0,035 |

Si l'on tolère qu'une tentative d'usurpation sur 1 000 passe, seules 3,6 % des vraies
sessions déclencheraient une fausse alerte. Ce chiffre n'est qu'indicatif : de vraies
usurpations, faites par quelqu'un qui *imite* l'utilisateur, seraient plus difficiles à
détecter que des sessions d'autres testeurs qui ne cherchent pas à tromper le modèle.

### Diagnostic par utilisateur (fig12, fig13)

- **141 utilisateurs sur 247 ont un F1 parfait (1,0)**, et seuls 2 sont sous 0,5 :
  - `hll` : F1 0,11, 9 sessions ;
  - `zok` : F1 0,47, 11 sessions.
- **Le F1 ne dépend presque pas du nombre de sessions** : le F1 moyen vaut 0,93 pour les
  utilisateurs à 4-6 sessions et entre 0,93 et 0,97 au-delà. L'équilibrage des classes
  (`class_weight="balanced"`) a bien protégé les petits utilisateurs.
- **AUC et F1 ne désignent pas les mêmes utilisateurs difficiles.**
  - `hll` a une AUC de 0,993 : ses sessions sont bien classées *près du haut*, mais
    l'utilisateur `guh` passe juste devant (3 confusions `hll` → `guh`).
  - À l'inverse, `zus` (AUC la plus basse, 0,956) a un F1 correct. Une de ses sessions
    reçoit un score nul, ce qui crée un long palier sur sa ROC.

  L'AUC juge le *classement*, le F1 juge la *décision* : les deux se complètent.
- **Confusions principales** :
  - `flj` → `skm` (8) et `skm` ↔ `slq` (5 + 5) : ce sont trois des utilisateurs les plus
    actifs, dont les boxplots de l'étape 2 montraient des profils proches (même vitesse,
    part en mode modification très variable) ;
  - `xkr` → `oqa` (6) ;
  - `zok` → `sps` (5).

  Il s'agit vraisemblablement de testeurs au périmètre similaire.

### Courbe d'apprentissage (fig15)

| Sessions d'entraînement | F1 macro entraînement | F1 macro validation |
|---|---|---|
| 614 | 1,00 | 0,626 |
| 1 229 | 1,00 | 0,837 |
| 1 844 | 1,00 | 0,919 |
| 2 459 | 1,00 | 0,939 |

- **Score d'entraînement de 1** : la forêt apprend parfaitement ses données d'entraînement.
  C'est normal pour une forêt aléatoire (arbres complets). Elle généralise quand même grâce
  au moyennage de 500 arbres construits sur des échantillons différents.
- **Le score de validation monte encore entre 75 % et 100 % des données** (+0,02). **Plus de
  sessions par utilisateur amélioreraient le modèle** : c'est la limite principale, plus
  que le choix du modèle.

### Variables importantes (fig16)

- **Importances très « plates »** : la première variable ne pèse que 0,7 % du total.
  L'information est répartie sur des centaines de variables, sans « variable magique ».
- **En tête** :
  - des proportions d'actions (`part_action_Raccourci`, grille d'historique, retour sur un
    écran, double-clic, toasts, filtrage) ;
  - le navigateur ;
  - une configuration (`conf=INSTALLATEUR`) ;
  - quelques écrans (planning CRM) ;
  - un enchaînement (`Exécution d'un bouton > Chainage`).

  C'est cohérent avec l'étape 4 : les habitudes d'action dominent, et le contexte (écran,
  configuration) affine.
- **Biais de la mesure** : l'importance par impureté favorise les variables continues
  (proportions) par rapport aux variables binaires ou rares. Il faut donc la lire avec
  prudence.

## 6. Limites et pistes d'amélioration

- **Pas de jeu de test interne indépendant** : le seul contrôle vraiment externe est le
  score Kaggle. Les scores hors pli restent légèrement optimistes, car les hyperparamètres
  ont été choisis sur les mêmes plis (voir l'étape 5).
- **Choix d'un seuil de rejet** : pour un usage réel en détection d'intrusion, il faudrait
  fixer le seuil sur un jeu dédié et tester de vraies tentatives d'usurpation, et non des
  sessions d'autres testeurs.
- **Importance par permutation** : la calculer sur les familles de variables (toutes les
  parts d'actions, tous les écrans…) serait plus fiable que l'importance par impureté, et
  moins coûteux qu'une permutation variable par variable.
- **Exploiter le top-3 (98,7 %)** : un second modèle spécialisé pour départager les 2 ou 3
  candidats les plus probables pourrait récupérer une partie des erreurs.
