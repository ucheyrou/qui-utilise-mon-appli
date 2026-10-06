"""Constantes partagées par toutes les étapes du pipeline.

Tous les chemins sont construits à partir de la racine du projet : les scripts
peuvent donc être lancés depuis n'importe quel dossier.
"""

from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent

# Données brutes fournies par Kaggle
DOSSIER_DONNEES = RACINE / "data"
FICHIER_TRAIN = DOSSIER_DONNEES / "train.csv"
FICHIER_TEST = DOSSIER_DONNEES / "test.csv"
FICHIER_EXEMPLE_SOUMISSION = DOSSIER_DONNEES / "sample_submission.csv"

# Résultats produits par le pipeline (non versionnés)
DOSSIER_SORTIES = RACINE / "outputs"
DOSSIER_FIGURES = DOSSIER_SORTIES / "figures"
DOSSIER_TABLEAUX = DOSSIER_SORTIES / "tableaux"
SESSIONS_TRAIN = DOSSIER_SORTIES / "sessions_train.pkl"
SESSIONS_TEST = DOSSIER_SORTIES / "sessions_test.pkl"
SESSIONS_PRETRAITEES_TRAIN = DOSSIER_SORTIES / "sessions_pretraitees_train.pkl"
SESSIONS_PRETRAITEES_TEST = DOSSIER_SORTIES / "sessions_pretraitees_test.pkl"
FEATURES_TRAIN = DOSSIER_SORTIES / "features_train.pkl"
FEATURES_TEST = DOSSIER_SORTIES / "features_test.pkl"
SCORES_HORS_PLI = DOSSIER_SORTIES / "scores_hors_pli.pkl"
MODELE_FINAL = DOSSIER_SORTIES / "modele_final.joblib"
SOUMISSION = DOSSIER_SORTIES / "submission.csv"

ENCODAGE = "utf-8"
RANDOM_STATE = 42
# La plus petite classe compte 4 sessions : au-delà de 4 plis, une validation
# croisée stratifiée ne pourrait plus placer chaque utilisateur dans chaque pli.
N_SPLITS = 4
