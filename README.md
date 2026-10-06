# Qui utilise mon appli ? — Identification des utilisateurs de Copilote

Projet du BE **MOD 7.2 – Introduction à la science des données** (groupe 1, 2026-2027).

Objectif : identifier l'utilisateur à partir de ses traces d'utilisation du logiciel Copilote (Infologic).
Compétition Kaggle : https://www.kaggle.com/competitions/qui-utilise-mon-appli-challenge/

## Structure

```
.
├── instructions/           # consignes du BE et description Kaggle
├── data/                   # données Kaggle (non versionnées)
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── src/                    # une étape du pipeline par fichier
│   ├── config.py           # chemins et constantes
│   ├── traces.py           # lecture des jetons de trace
│   ├── features.py         # construction des features
│   ├── modeles.py          # catalogue des modèles, contrainte navigateur
│   ├── figures.py          # style graphique et tracés communs
│   ├── 01_chargement.py
│   ├── 02_exploration.py
│   ├── 03_pretraitement.py
│   ├── 04_features.py
│   ├── 05_modele.py
│   ├── 06_evaluation.py
│   └── 07_soumission.py
├── explications/           # un fichier d'explication par étape (base du rapport)
├── outputs/                # sorties régénérables (non versionnées)
└── requirements.txt
```

## Installation

Python 3.14 :

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Télécharger les données depuis l'onglet *Data* de la compétition Kaggle et les placer dans `data/`.

## Exécution

Les étapes se lancent dans l'ordre depuis la racine du projet ; chacune lit les sorties de la précédente
dans `outputs/`.

```bash
python src/01_chargement.py    # -> outputs/sessions_*.pkl
python src/02_exploration.py   # -> outputs/figures/, outputs/tableaux/
python src/03_pretraitement.py # -> outputs/sessions_pretraitees_*.pkl
python src/04_features.py      # -> outputs/features_*.pkl + carnet d'essais (~1 min)
python src/05_modele.py        # -> comparaison des modèles, outputs/modele_final.joblib (~20 min)
python src/06_evaluation.py    # -> métriques, courbes ROC, diagnostic (~4 min)
python src/07_soumission.py    # -> outputs/submission.csv (à déposer sur Kaggle)
```

Les forêts aléatoires occupent environ 1 Go de mémoire chacune : le parallélisme est limité à
2 entraînements simultanés pour tenir sur une machine de 8 Go.

## Résultats

Modèle retenu : forêt aléatoire (500 arbres, classes équilibrées) sur features simples + TF-IDF des
traces, avec la contrainte « un utilisateur = un navigateur ».
F1 macro en validation croisée stratifiée (4 plis) : **0,943 ± 0,005** ; accuracy 0,948 ;
le vrai utilisateur est parmi les 3 premières propositions dans 98,7 % des cas.

## Rendu

- Code Python ré-exécutable (+ `requirements.txt`)
- Rapport PDF de 5 à 8 pages
- Prédictions déposées sur Kaggle (`submission.csv`)
