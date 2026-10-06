"""Étape 1 — Chargement des données brutes.

Lit train.csv et test.csv ligne par ligne, range chaque session dans un
DataFrame (une ligne = une session, la trace étant gardée sous forme de liste
de jetons), vérifie la cohérence des fichiers puis sauvegarde le résultat
dans outputs/ pour les étapes suivantes.

Usage : python src/01_chargement.py
"""

import pandas as pd

from config import (
    DOSSIER_SORTIES,
    ENCODAGE,
    FICHIER_EXEMPLE_SOUMISSION,
    FICHIER_TEST,
    FICHIER_TRAIN,
    SESSIONS_TEST,
    SESSIONS_TRAIN,
)


def lire_sessions(chemin, avec_cible: bool) -> pd.DataFrame:
    """Lit un fichier de traces et renvoie une ligne par session.

    On lit le fichier à la main plutôt qu'avec pd.read_csv : les lignes ont des
    longueurs très variables (de 3 à plus de 14 000 champs) et read_csv créerait
    autant de colonnes que la plus longue ligne, presque toutes vides.

    Paramètres
    ----------
    chemin : Path
        Fichier CSV à lire.
    avec_cible : bool
        True pour le jeu d'entraînement, dont le premier champ est l'utilisateur.

    Renvoie
    -------
    pd.DataFrame
        Colonnes « util » (train) ou « RowId » (test), « navigateur » et
        « traces » (liste des jetons de la session, dans l'ordre).
    """
    enregistrements = []
    with open(chemin, encoding=ENCODAGE) as fichier:
        for ligne in fichier:
            champs = ligne.rstrip("\n").split(",")
            if avec_cible:
                util, navigateur, *traces = champs
                enregistrements.append(
                    {"util": util, "navigateur": navigateur, "traces": traces}
                )
            else:
                navigateur, *traces = champs
                enregistrements.append({"navigateur": navigateur, "traces": traces})

    sessions = pd.DataFrame(enregistrements)
    if not avec_cible:
        # Kaggle numérote les sessions de test à partir de 1, dans l'ordre du fichier
        sessions.insert(0, "RowId", range(1, len(sessions) + 1))
    return sessions


def verifier(condition: bool, message: str) -> None:
    """Lève une ValueError avec le message donné si la condition est fausse."""
    if not condition:
        raise ValueError(message)


def verifier_coherence(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Contrôle que les données chargées sont exploitables.

    Vérifie l'absence de champs vides, la correspondance entre le test et le
    fichier d'exemple de soumission, et que le test ne contient pas de
    navigateur inconnu du train.

    Paramètres
    ----------
    train, test : pd.DataFrame
        Sessions renvoyées par lire_sessions.
    """
    exemple = pd.read_csv(FICHIER_EXEMPLE_SOUMISSION, encoding=ENCODAGE)

    verifier(train["util"].str.len().gt(0).all(), "utilisateur vide dans le train")
    for nom, sessions in [("train", train), ("test", test)]:
        verifier(sessions["traces"].map(len).gt(0).all(), f"session vide ({nom})")
        jetons_vides = sessions["traces"].map(lambda traces: "" in traces).sum()
        verifier(jetons_vides == 0, f"{jetons_vides} sessions avec un jeton vide ({nom})")

    verifier(
        test["RowId"].tolist() == exemple["RowId"].tolist(),
        "les RowId du test ne correspondent pas à sample_submission.csv",
    )
    navigateurs_inconnus = set(test["navigateur"]) - set(train["navigateur"])
    verifier(
        not navigateurs_inconnus,
        f"navigateurs absents du train : {navigateurs_inconnus}",
    )


def afficher_resume(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Affiche les dimensions et quelques chiffres clés des données chargées."""
    longueurs_train = train["traces"].map(len)
    longueurs_test = test["traces"].map(len)
    print(f"Sessions : {len(train)} en train, {len(test)} en test")
    print(f"Utilisateurs distincts (classes) : {train['util'].nunique()}")
    print(f"Navigateurs : {sorted(train['navigateur'].unique())}")
    print(
        "Jetons par session (min / médiane / max) : "
        f"train {longueurs_train.min()} / {longueurs_train.median():.0f} / "
        f"{longueurs_train.max()}, "
        f"test {longueurs_test.min()} / {longueurs_test.median():.0f} / "
        f"{longueurs_test.max()}"
    )


def main() -> None:
    """Charge, vérifie et sauvegarde les sessions d'entraînement et de test."""
    train = lire_sessions(FICHIER_TRAIN, avec_cible=True)
    test = lire_sessions(FICHIER_TEST, avec_cible=False)

    verifier_coherence(train, test)
    afficher_resume(train, test)

    DOSSIER_SORTIES.mkdir(parents=True, exist_ok=True)
    train.to_pickle(SESSIONS_TRAIN)
    test.to_pickle(SESSIONS_TEST)
    print(f"Sessions sauvegardées dans {DOSSIER_SORTIES}")


if __name__ == "__main__":
    main()
