# CLAUDE.md — Consignes pour Claude sur ce projet

## Contexte : un projet de cours

Ce dépôt est un **projet étudiant** réalisé dans le cadre du BE **MOD 7.2 – Introduction à la science des données**
(4A, groupe 1, 2026-2027 ; enseignants : Julien Velcin et Antoine Becquet).

- **Sujet** : identifier l'utilisateur du logiciel Copilote (Infologic) à partir de ses traces d'utilisation
  (classification supervisée multi-classes).
- **Compétition Kaggle** : https://www.kaggle.com/competitions/qui-utilise-mon-appli-challenge/
- **Métrique** : F1 score moyen.
- **Consignes officielles** : voir [instructions/](instructions/) (notebooks `BE_CS1_*.ipynb`, `kaggle.md`,
  `dataset_descriiption.md`). Les relire avant toute décision structurante.
- **Rendu attendu** : code Python ré-exécutable (Python 3.14 + `requirements.txt`), rapport PDF de 5 à 8 pages,
  prédictions déposées sur Kaggle.
- **Critères de notation** : chaîne de traitement complète, **capacité à expliquer et motiver chaque choix**,
  performance de la solution.

Comme le but est pédagogique, **comprendre et justifier compte autant que le score**. Chaque choix
(nettoyage, features, modèle, hyperparamètres) doit pouvoir être expliqué à l'oral par les étudiants.

## Règle principale : une explication par étape

Chaque étape du pipeline a **son propre fichier d'explication**, séparé du code, dans le dossier `explications/`.

| Étape | Code | Explication |
| ----- | ---- | ----------- |
| 1. Chargement des données brutes | `src/01_chargement.py` | `explications/01_chargement.md` |
| 2. Exploration / premières analyses | `src/02_exploration.py` | `explications/02_exploration.md` |
| 3. Nettoyage et pré-traitement | `src/03_pretraitement.py` | `explications/03_pretraitement.md` |
| 4. Construction des features | `src/04_features.py` | `explications/04_features.md` |
| 5. Modélisation | `src/05_modele.py` | `explications/05_modele.md` |
| 6. Évaluation | `src/06_evaluation.py` | `explications/06_evaluation.md` |
| 7. Prédiction et soumission Kaggle | `src/07_soumission.py` | `explications/07_soumission.md` |

Ce découpage peut évoluer (ajout d'une étape, notebooks au lieu de scripts…), mais le principe reste :
**un fichier de code ↔ un fichier d'explication portant le même numéro et le même nom**.
Toute modification du code d'une étape implique de **mettre à jour son fichier d'explication** dans le même temps.

### Contenu attendu d'un fichier d'explication

Chaque `explications/XX_nom.md` suit ce plan :

1. **Objectif** — ce que fait l'étape et pourquoi elle est nécessaire.
2. **Entrées / sorties** — fichiers ou variables lus et produits (formes, types, colonnes).
3. **Démarche pas à pas** — les traitements dans l'ordre, en citant les fonctions concernées.
4. **Choix et justifications** — pourquoi cette méthode plutôt qu'une autre, alternatives envisagées.
5. **Résultats / observations** — chiffres clés, graphiques, ce qu'on en conclut.
6. **Limites et pistes d'amélioration**.

Le ton est celui d'un étudiant qui explique son travail à un enseignant : clair, précis, sans jargon inutile,
avec les notions de cours rappelées brièvement quand elles servent (ex. : pourquoi le F1 macro, ce qu'est le
sur-apprentissage, etc.). Ces fichiers servent de base à la rédaction du rapport PDF.

## Qualité du code

- **Encodage UTF-8 partout** : fichiers sources, Markdown, CSV produits. Toujours préciser l'encodage
  explicitement en lecture/écriture (`open(..., encoding="utf-8")`, `pd.read_csv(..., encoding="utf-8")`,
  `df.to_csv(..., encoding="utf-8", index=False)`). Les accents français doivent s'afficher correctement.
- **Code propre et lisible** :
  - respecter PEP 8, noms de variables et fonctions explicites (en français ou en anglais, mais de façon cohérente) ;
  - petites fonctions à responsabilité unique, avec **docstring** (rôle, paramètres, valeur de retour) ;
  - pas de code mort, pas de cellules ou blocs commentés laissés « au cas où » ;
  - constantes (chemins, graine aléatoire, hyperparamètres) regroupées en haut du fichier ;
  - imports en tête de fichier, regroupés (bibliothèque standard, tiers, local).
- **Commentaires en français**, utiles et non redondants : expliquer le *pourquoi* plutôt que répéter le *quoi*.
- **Reproductibilité** : graine aléatoire fixée (`RANDOM_STATE = 42`), chemins relatifs à la racine du projet,
  toute nouvelle dépendance ajoutée à `requirements.txt`.
- **Évaluation honnête** : validation croisée ou jeu de validation séparé, jamais d'évaluation sur les données
  d'entraînement, attention aux fuites de données entre train et test.

## Organisation du dépôt

```
.
├── CLAUDE.md              # ce fichier
├── README.md              # présentation et installation
├── requirements.txt       # dépendances Python
├── instructions/          # consignes du BE et description Kaggle (ne pas modifier)
├── data/                  # données Kaggle (non versionnées, cf. .gitignore)
├── src/                   # code, un fichier par étape
│   ├── config.py          # chemins et constantes partagés
│   ├── traces.py          # lecture des jetons (marqueurs tXX, décomposition des actions)
│   ├── features.py        # construction des features (réutilisé par les étapes 5 et 7)
│   ├── modeles.py         # catalogue des modèles, pipeline, contrainte navigateur
│   └── figures.py         # style graphique et tracés communs
├── outputs/               # sorties régénérables : pickles, figures, tableaux (non versionnées)
└── explications/          # un fichier Markdown d'explication par étape
```

`config.py` et `traces.py` sont des modules utilitaires partagés, sans étape propre : leurs choix
sont expliqués dans le fichier d'explication de la première étape qui les utilise.

- Ne jamais versionner les données (`data/`, `*.csv`) : elles sont exclues par le `.gitignore`.
- Ne pas modifier le contenu de `instructions/` : ce sont les documents fournis par les enseignants.

## Façon de travailler avec Claude

- Rédiger les réponses, commentaires et explications **en français**.
- Avant d'écrire du code, expliquer brièvement l'approche ; après, résumer ce qui a été fait et ce qu'il faut
  retenir pour le rapport.
- Privilégier des solutions que les étudiants peuvent comprendre et défendre plutôt que des astuces opaques.
  Si une méthode avancée est utilisée, l'expliquer dans le fichier d'explication correspondant.
