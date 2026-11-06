# Étape 5 — Modélisation

Code :

- [`src/modeles.py`](../src/modeles.py) : catalogue des modèles, pipeline et contrainte navigateur ;
- [`src/05_modele.py`](../src/05_modele.py) : comparaison des modèles et entraînement du modèle final.

Lancement : `python src/05_modele.py` (après l'étape 4, environ 20 min).

## 1. Objectif

Choisir le modèle de classification et ses hyperparamètres. On en compare plusieurs, chacun
réglé au mieux, sur les **mêmes features** (essai E8 de l'étape 4 : features simples +
TF-IDF complet) et avec la **même validation croisée**. Un modèle supplémentaire, le SVM
linéaire, est aussi évalué sur la représentation « séquences d'actions complètes » (essai E9),
qui s'est révélée bien mieux adaptée aux modèles linéaires. On teste ensuite une règle métier
issue de l'exploration, « un utilisateur n'utilise qu'un seul navigateur », puis on
entraîne le modèle retenu sur tout le jeu d'entraînement.

## 2. Entrées / sorties

| | Fichier | Contenu |
|---|---|---|
| Entrée | `outputs/features_train.pkl` | Features de l'étape 4 |
| Sortie | `outputs/tableaux/comparaison_modeles.csv` | Une ligne par modèle : meilleurs hyperparamètres, F1 macro, F1 pondéré, accuracy (moyenne ± écart-type sur les plis), durée |
| Sortie | `outputs/tableaux/grilles_hyperparametres.csv` | Score de chaque combinaison d'hyperparamètres testée |
| Sortie | `outputs/scores_hors_pli.pkl` | Scores hors pli de chaque modèle (matrices 3 279 × 247), utilisés par l'étape 6 |
| Sortie | `outputs/modele_final.joblib` | Modèle retenu, entraîné sur tout le train (39 Mo compressé) |
| Sortie | `outputs/figures/fig10_comparaison_modeles.png` | Comparaison des modèles |

## 3. Démarche pas à pas

1. **Découpage** : `StratifiedKFold(n_splits=4, shuffle=True, random_state=42)`, identique
   pour tous les modèles. La stratification garantit que chaque utilisateur est représenté
   dans chaque pli (au moins 1 session en validation et 3 en entraînement).
2. **Pour chaque modèle du catalogue** (`catalogue_modeles`) :
   - `construire_pipeline` assemble les features (normalisation, one-hot, TF-IDF) et le
     modèle dans un seul `Pipeline` ;
   - `regler_hyperparametres` : `GridSearchCV` teste toutes les combinaisons de la grille et
     garde celle qui a le meilleur **F1 macro** moyen sur les 4 plis ;
   - `calculer_scores_hors_pli` : `cross_val_predict` produit, pour chaque session, les
     probabilités (ou scores de décision pour le SVM) données par le modèle entraîné **sur
     les 3 autres plis** ;
   - `mesurer_par_pli` en déduit le F1 macro, le F1 pondéré et l'accuracy de chaque pli.
3. `evaluer_contrainte_navigateur` applique la classe `ContrainteNavigateur` aux **deux
   meilleurs** modèles (`NB_MODELES_CONTRAINTS`) et les évalue de la même façon.
4. Le meilleur modèle (F1 macro) est réentraîné sur les 3 279 sessions et sauvegardé.

## 4. Choix et justifications

### Les modèles comparés

| Modèle | Pourquoi l'essayer | Hyperparamètres explorés |
|---|---|---|
| Référence (classe majoritaire) | Plancher : ce que fait un modèle qui n'apprend rien | — |
| k plus proches voisins (cosinus) | Le plus simple : « cette session ressemble à celle de… » ; le cosinus est la distance naturelle entre vecteurs TF-IDF | k ∈ {1, 3, 5, 10} |
| Régression logistique | Modèle linéaire de référence, probabiliste, interprétable | C ∈ {1, 10, 100}, poids des classes |
| SVM linéaire | Souvent le meilleur modèle linéaire sur du texte en TF-IDF | C ∈ {0,03 ; 0,1 ; 0,3} |
| Forêt aléatoire | Non linéaire, robuste, peu de réglages | `max_features` ∈ {√p, 5 %}, poids des classes |
| Extra-Trees | Variante plus aléatoire de la forêt (seuils tirés au hasard) | idem |
| Perceptron multicouche | Réseau de neurones simple (demandé dans les consignes) | 256 ou 512 neurones, régularisation α |
| Gradient boosting | Méthode d'ensemble séquentielle (type XGBoost) | réglages fixés (voir ci-dessous) |
| SVM linéaire sur séquences d'actions | Même modèle, mais sur navigateur + TF-IDF par famille + TF-IDF des actions complètes et de leurs bigrammes (E9), sans features simples | C ∈ {0,3 ; 1 ; 3}, `class_weight=balanced` |

### Pourquoi une représentation à part pour le SVM

Avec les features de E8, le SVM plafonnait à 0,890. Deux raisons :

- **Le contexte des actions était perdu.** Les jetons par famille séparent l'action de son
  écran ; les jetons « action complète » (étape 4, E9) les gardent ensemble, et leurs
  bigrammes décrivent un geste précis. Ces jetons sont très nombreux (≈ 46 000) et chacun
  est rare, mais très spécifique à un utilisateur : c'est exactement le cas où un modèle
  linéaire sur TF-IDF excelle, car il additionne les petites contributions de tous les
  jetons présents.
- **Les features simples centrées-réduites gênaient le SVM.** Mêlées aux TF-IDF (valeurs
  entre 0 et 1, normées par session), elles n'ont pas la même échelle et dominent la marge :
  sur la représentation séquences, les ajouter faisait tomber le F1 macro de 0,95 à 0,93.

À l'inverse, la forêt n'en profite pas (E9 : 0,897 contre 0,913 pour E8) : à chaque nœud,
elle ne regarde qu'un tirage de √p ≈ 200 colonnes parmi 46 000 et rate la plupart des jetons
utiles. **La meilleure représentation dépend du modèle**, d'où ce couple « features + modèle »
évalué à part.

Essais complémentaires, sans gain (F1 macro avec contrainte, 4 plis) : jetons vus une seule
fois conservés (`min_df=1`, 0,969), trigrammes d'actions complètes (0,965), jetons des 10
premières actions (0,968), dates masquées dans les configurations (0,969), régression
logistique (0,959), classifieur ridge (0,966), moyenne des scores SVM et forêt (0,971, gain
dans le bruit et modèle bien plus lourd).

### Critère de sélection : le F1 macro

C'est la métrique de la compétition (« F1 moyen »), et la plus exigeante : chaque
utilisateur compte autant, qu'il ait 4 ou 75 sessions. On affiche aussi l'accuracy, qui
correspond au F1 micro en classification mono-étiquette.

### Le poids des classes (`class_weight="balanced"`)

Ce réglage donne plus de poids aux sessions des utilisateurs rares pendant l'apprentissage.
Il a été retenu par la recherche de grille pour la régression logistique, la forêt et les
Extra-Trees, ce qui est cohérent avec un critère macro.

### Tâtonnements sur le gradient boosting

Avec les réglages par défaut, le gradient boosting obtenait un F1 macro de 0,20 sur un pli.

- **Première hypothèse, fausse** : les feuilles exigent au moins 20 sessions
  (`min_samples_leaf=20`) alors que beaucoup d'utilisateurs n'en ont que 3 par pli. Avec
  `min_samples_leaf=2`, le score restait à 0,16.
- **Diagnostic** : le F1 **sur le pli d'entraînement lui-même** n'était que de 0,44, et le
  score de validation passait de 0,31 après 1 itération à 0,07 après 5. Ce n'est pas du
  sur-apprentissage : **l'optimisation diverge**. Avec 247 classes et très peu d'exemples
  par classe, les premières mises à jour sont trop grandes.
- **Correction** : pas d'apprentissage réduit (0,05), pénalité L2 sur les feuilles, arbres
  de profondeur 4. Le score de validation monte alors régulièrement (0,45 → 0,61 → 0,72 →
  0,77). Le modèle reste toutefois loin des forêts et coûte environ 10 minutes. Nous n'avons
  pas poussé le réglage plus loin.

### La contrainte navigateur (`ContrainteNavigateur`)

L'exploration a montré que chacun des 247 utilisateurs n'utilise qu'un seul navigateur. La
forêt aléatoire prédisait pourtant, pour 17 de ses 182 erreurs, un utilisateur qui
n'a **jamais** utilisé le navigateur de la session.

- **Principe** : on met à zéro la probabilité des utilisateurs incompatibles avec le
  navigateur, puis on renormalise. Pour un modèle qui ne donne que des scores de décision
  (SVM), on ramène le score des utilisateurs incompatibles sous le plus petit score de la
  session : ils passent en dernière position, sans changer l'ordre des autres. La contrainte
  s'applique donc à n'importe quel modèle (méthodes exposées avec `available_if` selon ce
  que sait faire le modèle encapsulé).
- **Apprentissage sans fuite** : la correspondance utilisateur → navigateur est apprise à
  l'entraînement, donc pli par pli pendant la validation croisée.
- **Cas de repli** : si le modèle n'accordait aucune probabilité aux utilisateurs
  compatibles, on leur donne une probabilité uniforme. Une prédiction incompatible serait
  fausse à coup sûr. Ce cas de repli a été ajouté après un premier test avec le k-NN, où
  10 prédictions restaient incompatibles.
- **Pourquoi elle ne peut pas dégrader le modèle** : si la meilleure proposition du modèle
  est compatible, elle ne change pas. Si elle est incompatible, elle était fausse. Le gain
  observé (+0,004) est donc petit mais **systématique**, et non du bruit, tant que la règle
  reste vraie.

### Prévention des fuites et honnêteté de l'évaluation

- Tout le pré-traitement est dans le pipeline et réappris à chaque pli.
- Les scores rapportés sont **hors pli** : aucune session n'est jamais prédite par un modèle
  qui l'a vue.
- **Biais de sélection** : les hyperparamètres sont choisis sur les mêmes plis que ceux qui
  servent à rapporter le score, ce qui rend ce score légèrement optimiste. Une validation
  croisée imbriquée corrigerait ce biais, mais multiplierait le temps de calcul par 4.
  Comme les grilles sont petites (au plus 6 combinaisons), le biais l'est aussi, et le score
  Kaggle public sert de contrôle indépendant.

### Mémoire

Une forêt de 500 arbres occupe environ 1,1 Go en mémoire, car chaque nœud stocke 247
probabilités. Au-delà de 2 forêts entraînées en parallèle, une machine de 8 Go sature et
utilise le disque à la place de la RAM. Le parallélisme est donc limité à 2 pour ces
modèles (`N_JOBS_FORETS`), et le modèle final est sauvegardé compressé : 39 Mo au lieu de
1,5 Go.

## 5. Résultats / observations

> **Mise à jour (modèle séquences)** : le SVM linéaire sur séquences d'actions obtient
> **0,966 ± 0,010** de F1 macro (accuracy 0,970, C = 0,3) sans contrainte, et environ
> **0,969** avec la contrainte navigateur. Il devient le modèle retenu, devant la forêt
> contrainte (0,943), tout en s'entraînant en quelques secondes. Avec 10 plis (90 % du
> train pour apprendre, plus proche du modèle final entraîné sur tout le train), il atteint
> 0,974. Les erreurs restantes portent surtout sur des sessions courtes (130 actions en
> médiane contre 340). Le tableau ci-dessous est celui de la première version ; il sera
> régénéré par `python src/05_modele.py`.

Validation croisée stratifiée à 4 plis, moyenne ± écart-type sur les plis (fig10) :

| Modèle | Hyperparamètres retenus | F1 macro | Accuracy |
|---|---|---|---|
| **Forêt aléatoire + contrainte navigateur** | `class_weight=balanced`, `max_features=sqrt` | **0,943** ± 0,005 | 0,948 |
| Forêt aléatoire | idem | 0,939 ± 0,006 | 0,944 |
| Extra-Trees | `class_weight=balanced`, `max_features=0.05` | 0,935 ± 0,005 | 0,951 |
| Perceptron multicouche | 256 neurones, α = 10⁻⁴ | 0,903 ± 0,003 | 0,921 |
| SVM linéaire | C = 0,3 | 0,890 ± 0,004 | 0,910 |
| Régression logistique | C = 10, `class_weight=balanced` | 0,855 ± 0,010 | 0,871 |
| Gradient boosting | réglages fixés | 0,774 ± 0,011 | 0,821 |
| k plus proches voisins | k = 1 | 0,707 ± 0,017 | 0,735 |
| Référence (classe majoritaire) | — | 0,000 | 0,023 |

Ce qu'on en retient :

- **Les forêts dominent.** Notre interprétation : la signature d'un utilisateur est une
  *combinaison* de proportions d'actions (« beaucoup de flags **et** peu de raccourcis
  **et** tel écran »). Un arbre exprime naturellement ces combinaisons, alors qu'un modèle
  linéaire additionne des effets indépendants. Le moyennage de 500 arbres limite le
  sur-apprentissage d'un arbre seul.
- **L'équilibrage des classes est le réglage le plus utile.** À hyperparamètres égaux
  (500 arbres, `max_features=sqrt`), la forêt passe de 0,910 sans pondération à **0,939**
  avec `class_weight="balanced"`, soit +0,029, bien au-delà de l'écart-type entre plis. Le
  détail de toutes les combinaisons est dans `grilles_hyperparametres.csv`.
- **Les Extra-Trees ont la meilleure accuracy (0,951) mais un F1 macro plus faible.** Ils
  reconnaissent mieux les gros utilisateurs et moins bien les petits. C'est un bon exemple
  où le choix de la métrique change le choix du modèle.
- **Les modèles linéaires sont nettement en dessous**, contrairement à ce qu'on attendait
  pour du TF-IDF (voir l'étape 4). Le SVM retient C = 0,3, en bord de grille. Un essai
  préalable à C = 1 donnait le même score avec des problèmes de convergence, d'où la grille
  limitée à 0,3.
- **Le k-NN est le moins bon modèle « appris »** : avec k = 1, il copie l'utilisateur de la
  session la plus proche, ce qui est fragile quand les sessions d'un même utilisateur
  varient beaucoup.
- **Modèle retenu (première version) : forêt aléatoire équilibrée + contrainte navigateur**,
  remplacée ensuite par le SVM linéaire sur séquences d'actions + contrainte navigateur
  (voir l'encadré ci-dessus).

## 6. Limites et pistes d'amélioration

- **Grilles volontairement petites**, à cause du temps de calcul (20 min pour l'étape). Le
  nombre d'arbres, `min_samples_leaf` ou les paramètres du TF-IDF (`min_df`, familles de
  jetons) n'ont pas été réglés.
- **Écarts proches du bruit** : un seul découpage en 4 plis ne permet pas de distinguer
  sûrement des modèles séparés de moins de 0,01 (forêt, Extra-Trees). Une validation
  croisée répétée le permettrait.
- **Pistes de modèles** :
  - combiner les probabilités de la forêt et des Extra-Trees (vote souple) ;
  - régler davantage le boosting (plus d'itérations, puisque le score montait encore).
- **Hypothèse forte de la contrainte navigateur** : un utilisateur qui change de navigateur
  dans le test serait systématiquement mal classé.
