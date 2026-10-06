"""Étape 7 — Prédiction sur le jeu de test et fichier de soumission Kaggle.

Charge le modèle retenu à l'étape 5 (déjà entraîné sur tout le train), prédit
l'utilisateur de chaque session de test et écrit outputs/submission.csv au
format de sample_submission.csv, après vérification.

Usage : python src/07_soumission.py (après src/05_modele.py)
"""

import joblib
import pandas as pd

from config import (
    ENCODAGE,
    FEATURES_TEST,
    FEATURES_TRAIN,
    FICHIER_EXEMPLE_SOUMISSION,
    MODELE_FINAL,
    SOUMISSION,
)


def predire_test(modele, features_test: pd.DataFrame) -> pd.DataFrame:
    """Prédit l'utilisateur de chaque session de test.

    Paramètres
    ----------
    modele : estimateur ajusté
        Pipeline complet (features + modèle) entraîné sur tout le train.
    features_test : pd.DataFrame
        Features du test calculées à l'étape 4 (colonne « RowId » incluse).

    Renvoie
    -------
    pd.DataFrame
        Colonnes « RowId » et « prediction », dans l'ordre du fichier de test.
    """
    return pd.DataFrame({
        "RowId": features_test["RowId"],
        "prediction": modele.predict(features_test),
    })


def verifier_soumission(soumission: pd.DataFrame, utilisateurs_connus: set[str]) -> None:
    """Contrôle que la soumission respecte le format attendu par Kaggle.

    Lève une ValueError si les colonnes, les RowId ou les prédictions ne
    correspondent pas à sample_submission.csv et aux utilisateurs du train.
    """
    exemple = pd.read_csv(FICHIER_EXEMPLE_SOUMISSION, encoding=ENCODAGE)
    if list(soumission.columns) != list(exemple.columns):
        raise ValueError(f"colonnes {list(soumission.columns)} au lieu de {list(exemple.columns)}")
    if soumission["RowId"].tolist() != exemple["RowId"].tolist():
        raise ValueError("les RowId ne correspondent pas à sample_submission.csv")
    inconnus = set(soumission["prediction"]) - utilisateurs_connus
    if inconnus:
        raise ValueError(f"utilisateurs prédits absents du train : {inconnus}")


def main() -> None:
    """Prédit le test, vérifie et écrit le fichier de soumission."""
    final = joblib.load(MODELE_FINAL)
    features_test = pd.read_pickle(FEATURES_TEST)
    utilisateurs_connus = set(pd.read_pickle(FEATURES_TRAIN)["util"])

    soumission = predire_test(final["modele"], features_test)
    verifier_soumission(soumission, utilisateurs_connus)
    soumission.to_csv(SOUMISSION, encoding=ENCODAGE, index=False)

    print(f"Modèle : {final['nom']}")
    print(f"{len(soumission)} prédictions, {soumission['prediction'].nunique()} utilisateurs "
          f"distincts prédits")
    print(f"Fichier écrit : {SOUMISSION}")


if __name__ == "__main__":
    main()
