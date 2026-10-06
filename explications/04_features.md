# Étape 4 — Construction des features et carnet d'essais

Code :

- [`src/features.py`](../src/features.py) : fonctions de construction des features,
  réutilisées par les étapes 5 et 7 ;
- [`src/04_features.py`](../src/04_features.py) : calcul, inspection et carnet d'essais.

Lancement : `python src/04_features.py` (après l'étape 3, environ 1 min).

## 1. Objectif

Transformer chaque session (une liste d'actions de longueur variable) en un **vecteur de
taille fixe** qu'un modèle de classification peut utiliser. Plutôt que de tout construire
d'un coup, on procède **par essais successifs**. On commence par des features très simples
qu'on a imaginées nous-mêmes, comme compter une seule action, et on ajoute une famille de
features à la fois en mesurant à chaque fois ce qu'elle apporte. Le carnet d'essais qui en
résulte justifie les features retenues.

## 2. Entrées / sorties

| | Fichier | Contenu |
|---|---|---|
| Entrée | `outputs/sessions_pretraitees_{train,test}.pkl` | Sessions de l'étape 3 |
| Sortie | `outputs/features_train.pkl`, `outputs/features_test.pkl` | 3 279 et 324 lignes : identifiant, `navigateur`, 47 features numériques, 3 catégorielles, `actions` (pour le TF-IDF) |
| Sortie | `outputs/tableaux/classement_features_anova.csv` | Score F de chaque feature numérique |
| Sortie | `outputs/tableaux/carnet_essais_features.csv` | Résultats des 8 essais |
| Sortie | `outputs/figures/fig08_…`, `fig09_…` | Classement des features, carnet d'essais |

## 3. Démarche pas à pas

### 3.1 Les features « simples » (`features.py`)

Elles sont calculées session par session, sans regarder les autres sessions.

| Famille | Fonction | Features |
|---|---|---|
| Navigateur | (encodé dans le pipeline) | one-hot des 4 navigateurs |
| Comptage simple | `features_contenu` | `nb_raccourci` : nombre de raccourcis clavier |
| Rythme | `features_rythme` | `log_nb_actions`, `log_duree_s`, `vitesse_apm`, actions par fenêtre de 5 s (moyenne, écart-type, max) |
| Répartition | `features_contenu` | `part_action_<libellé>` : part de chacun des 35 libellés dans la session |
| Contenu | `features_contenu` | écrans, configurations et chaînes distincts, part en mode modification, part d'actions portant un écran |
| Catégorielles | `features_contenu` | `ecran_principal`, `premier_ecran`, `chaine_principale` (one-hot dans le pipeline) |

`calculer_features_simples` assemble le tout. Pour le test, elle reprend **exactement les
colonnes du train**, un libellé absent valant 0.

### 3.2 Le sac de jetons TF-IDF (`features.py`)

`jetons_session` transforme une session en « document ». Chaque action produit des jetons
préfixés par leur famille, par exemple :

- `action=Saisie dans un champ|modif`
- `ecran=…AnnuaireController`
- `conf=ACCUEIL_INST`
- `chaine=GP`
- `bigramme=Raccourci>Chainage` (deux actions consécutives)

Un `TfidfVectorizer` (`sublinear_tf=True`, `min_df=2`) en fait une matrice creuse. Il apprend
son vocabulaire et ses poids IDF **à l'intérieur du pipeline** (`construire_preprocesseur`),
donc uniquement sur les plis d'entraînement.

### 3.3 Inspection : classement univarié (`04_features.py`)

`classer_features_simples` calcule le **score F de l'ANOVA** de chaque feature numérique. Ce
score est le rapport entre la variance *entre* utilisateurs et la variance *au sein* d'un même
utilisateur. Un score élevé signifie que la feature sépare bien les utilisateurs (fig08).

### 3.4 Le carnet d'essais (`04_features.py`)

`definir_essais` liste 8 essais, et `evaluer_essai` évalue chacun avec :

- **le même modèle** : forêt aléatoire de 300 arbres, hyperparamètres par défaut ;
- **le même découpage** : `StratifiedKFold` à 4 plis, `random_state=42`.

Seules les features changent d'un essai à l'autre, ce qui rend la comparaison équitable.

## 4. Choix et justifications

- **Des proportions plutôt que des comptes.** Les sessions vont de 0 à 14 000 actions. Un
  compte brut mesure surtout la *longueur* de la session, alors qu'une proportion mesure
  l'*habitude*. Le classement ANOVA le confirme sur notre comptage de départ : `nb_raccourci`
  a un score F de 10,6 (16e sur 47), alors que la **même information normalisée**,
  `part_action_Raccourci`, atteint 50,2 (9e).
- **Logarithme du nombre d'actions et de la durée.** Ces grandeurs sont très étalées
  (fig03). Le log les ramène sur une échelle comparable, ce qui aide les modèles linéaires
  de l'étape 5.
- **Pas de features de pause.** On avait prévu de compter les pauses (sauts entre
  marqueurs). En les calculant, on a constaté qu'elles valaient **toujours 0** : les
  marqueurs se suivent de 5 en 5, et les temps morts ne sont pas enregistrés (voir
  étape 2). Ces features ont été retirées.
- **TF-IDF pour les écrans, configurations et chaînes.** Il y a des centaines de valeurs
  possibles, trop pour un one-hot « à la main ». Le TF-IDF pondère chaque jeton par sa
  rareté : un écran que tout le monde ouvre (l'accueil) pèse peu, un écran propre à un
  utilisateur pèse beaucoup. `sublinear_tf` (1 + log du nombre d'occurrences) évite qu'une
  action répétée 2 000 fois écrase tout. `min_df=2` ignore les jetons vus dans une seule
  session, qui ne peuvent pas aider à généraliser.
- **Bigrammes d'actions.** Ils capturent une partie de l'*ordre* des actions (habitudes
  d'enchaînement), que le simple sac de mots perd.
- **Prévention des fuites.** Tout ce qui « apprend » sur plusieurs sessions (normalisation,
  one-hot, vocabulaire et IDF du TF-IDF) est dans le `Pipeline`. La validation croisée le
  réapprend donc à chaque pli sur les seules données d'entraînement. Les features simples
  peuvent être calculées une fois pour toutes, car elles ne dépendent que de la session
  elle-même.
- **Une forêt aléatoire pour le carnet.**
  - Elle n'a pas besoin de normalisation et gère aussi bien des features denses que creuses.
  - Elle capte des effets non linéaires.
  - Elle donne de bons résultats sans réglage.

  Le choix du meilleur modèle et de ses hyperparamètres est l'objet de l'étape 5.
- **Vérification anti-fuite.** Le saut de l'essai E4 étant spectaculaire, on l'a relancé avec
  des utilisateurs **mélangés au hasard**. Le F1 macro tombe à 0,003, soit le niveau du
  hasard : le score de E4 vient bien d'un vrai signal et pas d'une fuite.

## 5. Résultats / observations

### Carnet d'essais (fig09)

Validation croisée stratifiée à 4 plis, forêt aléatoire de 300 arbres. Les scores sont des
moyennes sur les plis ± l'écart-type.

| Essai | Features | Nb | F1 macro | Accuracy |
|---|---|---|---|---|
| E1 | Navigateur seul | 4 | 0,005 ± 0,000 | 0,051 |
| E2 | + comptage d'une action (Raccourci) | 5 | 0,021 ± 0,002 | 0,070 |
| E3 | + rythme (volume, durée, vitesse) | 11 | 0,106 ± 0,008 | 0,147 |
| E4 | + répartition des 35 actions | 46 | **0,823** ± 0,005 | 0,845 |
| E5 | + contenu et écran / chaîne principaux | 241 | 0,857 ± 0,013 | 0,881 |
| E6 | Navigateur + TF-IDF des écrans seuls | 304 | 0,545 ± 0,008 | 0,644 |
| E7 | Navigateur + TF-IDF complet | 1 557 | 0,907 ± 0,008 | 0,930 |
| E8 | Features simples (E5) + TF-IDF complet | 1 794 | **0,913** ± 0,010 | 0,935 |

Ce qu'on en retient :

- **E1 : le navigateur seul ne suffit pas** (accuracy 5 %). Il réduit le nombre de candidats,
  mais il faut d'autres informations pour départager les dizaines d'utilisateurs d'un même
  navigateur.
- **E2 : notre premier comptage simple** quadruple le F1 macro (0,005 → 0,021), mais il
  reste très faible. Une seule action ne caractérise pas une personne.
- **E3 : le rythme** apporte un gain réel (F1 macro 0,106), mais les boxplots de l'étape 2
  l'annonçaient : chaque utilisateur varie beaucoup d'une session à l'autre.
- **E4 : la répartition des 35 actions fait passer le F1 macro de 0,11 à 0,82.** C'est le
  résultat central. La « signature » d'un utilisateur est sa *façon* de travailler : la part
  de tableaux de bord, de flags, de filtres, de raccourcis… Le classement ANOVA (fig08) va
  dans le même sens, puisque les 10 features les plus discriminantes sont toutes des parts
  d'actions, en tête `Lancement d'un tableau de bord`, `Sélection d'un flag` et
  `Entrée en saisie dans un formulaire`.
- **E6 : les écrans seuls sont décevants** (0,545). À l'étape 2, on pensait qu'ils seraient
  la meilleure source, puisque 147 écrans ne sont vus que par un seul utilisateur. Deux
  raisons expliquent ce résultat :
  - seules 13 % des actions portent un écran ;
  - les écrans spécifiques à un utilisateur n'apparaissent que dans peu de ses sessions.

  Le *où* compte moins que le *comment*.
- **E7 : le TF-IDF complet** (actions, écrans, configurations, chaînes, bigrammes) donne
  0,907 à lui seul. Il contient l'information de E4, enrichie du contexte.
- **E8 : ajouter les features simples au TF-IDF** n'apporte que +0,006, ce qui reste dans
  l'écart-type entre plis (± 0,01). Le gain n'est donc pas significatif. On garde quand
  même cette combinaison comme point de départ de l'étape 5 : les features simples sont
  peu nombreuses, interprétables, et pourraient aider d'autres modèles.
- **F1 macro systématiquement inférieur à l'accuracy** : les utilisateurs qui ont peu de
  sessions sont moins bien reconnus. C'est exactement ce que le F1 macro est censé révéler.

### Ce que les tâtonnements nous ont appris

Cette étape s'est construite par essais et erreurs, à garder pour la discussion du rapport :

1. **Normaliser** : un comptage brut (`nb_raccourci`) mesure la longueur de la session. Sa
   version normalisée (`part_action_Raccourci`) mesure l'habitude, et elle est 5 fois plus
   discriminante.
2. **Vérifier une idée sur les données avant de la garder** : les features de pause
   semblaient pertinentes, mais elles étaient toujours nulles.
3. **Tester les intuitions** : l'intuition « les écrans suffisent » a été démentie par E6.
4. **Douter des bons résultats** : le saut de E4 a été vérifié avec des étiquettes mélangées.

## 6. Limites et pistes d'amélioration

- **Une seule graine et un seul découpage.** Des écarts de l'ordre de 0,01 (E7 contre E8)
  ne sont pas significatifs. Une validation croisée répétée avec plusieurs graines
  permettrait de trancher.
- **Le carnet n'utilise qu'un modèle** (forêt aléatoire). On s'attendait à ce qu'un modèle
  linéaire exploite mieux un TF-IDF creux. L'étape 5 a démenti cette intuition : sur les
  features de E8, la régression logistique (0,855) et le SVM linéaire (0,890) restent sous
  la forêt (0,939).
- **Le classement ANOVA est univarié.** Il ne voit ni les redondances entre features ni
  leurs interactions. L'importance par permutation, à l'étape 6, complétera cette analyse.
- **Pistes de features** :
  - jetons combinant action et écran (`Exécution d'un bouton@MajFactureAchForm`) ;
  - trigrammes d'actions ;
  - premières actions de la session (rituel de démarrage) ;
  - normalisation des configurations contenant des dates.
- **Le one-hot du navigateur suppose qu'un utilisateur garde son navigateur.** C'est vrai
  pour les 247 utilisateurs du train, mais ce n'est pas garanti en général.
