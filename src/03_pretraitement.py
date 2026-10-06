"""Étape 3 — Nettoyage et pré-traitement des traces.

Transforme la liste brute de jetons de chaque session en deux listes alignées :
- « actions » : les actions décomposées (libellé, écran, configuration, chaîne,
  mode modification), dans l'ordre ;
- « temps » : pour chaque action, la fin de la fenêtre de 5 s où elle a eu lieu.

Corrige au passage les marqueurs collés à une action. Aucune session n'est
supprimée (voir explications/03_pretraitement.md).

Usage : python src/03_pretraitement.py (après src/01_chargement.py)
"""

import re

import numpy as np
import pandas as pd

from config import (
    SESSIONS_PRETRAITEES_TEST,
    SESSIONS_PRETRAITEES_TRAIN,
    SESSIONS_TEST,
    SESSIONS_TRAIN,
)
from traces import decomposer_action, est_marqueur_temps, valeur_marqueur

PAS_FENETRE_S = 5
MOTIF_MARQUEUR_COLLE = re.compile(r"(?P<action>.+?)(?P<marqueur>t\d+)")


def separer_marqueurs_colles(traces: list[str]) -> tuple[list[str], int]:
    """Sépare les marqueurs « tXX » collés à la fin d'une action.

    On ne découpe que les jetons illisibles tels quels : un jeton comme
    « Sélection d'un onglet1 » se termine aussi par « t1 » mais est une action
    valide (le « 1 » est le mode modification), il ne faut donc pas y toucher.

    Paramètres
    ----------
    traces : list[str]
        Jetons bruts d'une session.

    Renvoie
    -------
    tuple[list[str], int]
        Jetons corrigés et nombre de jetons séparés.
    """
    corriges = []
    nb_separes = 0
    for jeton in traces:
        colle = MOTIF_MARQUEUR_COLLE.fullmatch(jeton)
        if (
            not est_marqueur_temps(jeton)
            and decomposer_action(jeton) is None
            and colle is not None
            and decomposer_action(colle["action"]) is not None
        ):
            corriges.extend([colle["action"], colle["marqueur"]])
            nb_separes += 1
        else:
            corriges.append(jeton)
    return corriges, nb_separes


def rattacher_actions_aux_fenetres(traces: list[str]) -> tuple[list, np.ndarray, int]:
    """Décompose les actions et associe à chacune la fin de sa fenêtre temporelle.

    Un marqueur « tXX » clôt la fenêtre : les actions qui le précèdent (depuis
    le marqueur précédent) ont eu lieu avant XX secondes. Les actions situées
    après le dernier marqueur sont rattachées à la fenêtre suivante (+5 s).
    Les marqueurs répétés (« t40 … t40 ») donnent simplement la même valeur.

    Paramètres
    ----------
    traces : list[str]
        Jetons d'une session (marqueurs collés déjà séparés).

    Renvoie
    -------
    tuple[list[Action], np.ndarray, int]
        Actions décomposées, temps (en s) de chaque action, et nombre de jetons
        illisibles ignorés.
    """
    actions = []
    temps = []
    en_attente = 0  # actions vues depuis le dernier marqueur
    dernier_marqueur = 0
    nb_illisibles = 0

    for jeton in traces:
        if est_marqueur_temps(jeton):
            dernier_marqueur = valeur_marqueur(jeton)
            temps.extend([dernier_marqueur] * en_attente)
            en_attente = 0
            continue
        action = decomposer_action(jeton)
        if action is None:
            nb_illisibles += 1
            continue
        actions.append(action)
        en_attente += 1

    temps.extend([dernier_marqueur + PAS_FENETRE_S] * en_attente)
    return actions, np.array(temps, dtype=np.int32), nb_illisibles


def pretraiter_sessions(sessions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Applique le pré-traitement à toutes les sessions d'un jeu.

    Paramètres
    ----------
    sessions : pd.DataFrame
        Sessions de l'étape 1 (colonne « traces »).

    Renvoie
    -------
    tuple[pd.DataFrame, dict]
        Sessions avec les colonnes « actions » et « temps » à la place de
        « traces », et compteurs des corrections effectuées.
    """
    toutes_actions, tous_temps = [], []
    bilan = {"marqueurs_colles_separes": 0, "jetons_illisibles_ignores": 0}

    for traces in sessions["traces"]:
        traces_corrigees, nb_separes = separer_marqueurs_colles(traces)
        actions, temps, nb_illisibles = rattacher_actions_aux_fenetres(traces_corrigees)
        toutes_actions.append(actions)
        tous_temps.append(temps)
        bilan["marqueurs_colles_separes"] += nb_separes
        bilan["jetons_illisibles_ignores"] += nb_illisibles

    pretraitees = sessions.drop(columns="traces")
    pretraitees["actions"] = toutes_actions
    pretraitees["temps"] = tous_temps
    return pretraitees, bilan


def main() -> None:
    """Pré-traite les sessions d'entraînement et de test et les sauvegarde."""
    for nom, entree, sortie in [
        ("train", SESSIONS_TRAIN, SESSIONS_PRETRAITEES_TRAIN),
        ("test", SESSIONS_TEST, SESSIONS_PRETRAITEES_TEST),
    ]:
        sessions = pd.read_pickle(entree)
        pretraitees, bilan = pretraiter_sessions(sessions)
        nb_actions = pretraitees["actions"].map(len)
        print(
            f"{nom} : {len(pretraitees)} sessions, {nb_actions.sum()} actions "
            f"(min {nb_actions.min()} par session) ; "
            f"{bilan['marqueurs_colles_separes']} marqueur(s) collé(s) séparé(s), "
            f"{bilan['jetons_illisibles_ignores']} jeton(s) illisible(s) ignoré(s)"
        )
        pretraitees.to_pickle(sortie)
    print("Sessions pré-traitées sauvegardées.")


if __name__ == "__main__":
    main()
