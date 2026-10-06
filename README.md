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
│   ├── figures.py          # style graphique commun
│   ├── 01_chargement.py
│   ├── 02_exploration.py
│   ├── 03_pretraitement.py
│   └── 04_features.py
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
```

## Rendu

- Code Python ré-exécutable (+ `requirements.txt`)
- Rapport PDF de 5 à 8 pages
- Prédictions déposées sur Kaggle (`submission.csv`)
