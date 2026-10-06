# Qui utilise mon appli ? — Identification des utilisateurs de Copilote

Projet du BE **MOD 7.2 – Introduction à la science des données** (groupe 1, 2026-2027).

Objectif : identifier l'utilisateur à partir de ses traces d'utilisation du logiciel Copilote (Infologic).
Compétition Kaggle : https://www.kaggle.com/competitions/qui-utilise-mon-appli-challenge/

## Structure

```
.
├── BE_CS1_instructions.ipynb   # consignes du BE
├── BE_CS1_indications.ipynb    # notebook de travail
├── data/                       # données Kaggle (non versionnées)
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
└── requirements.txt
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Télécharger les données depuis l'onglet *Data* de la compétition Kaggle et les placer dans `data/`.

## Rendu

- Code Python ré-exécutable (+ `requirements.txt`)
- Rapport PDF de 5 à 8 pages
- Prédictions déposées sur Kaggle (`submission.csv`)
