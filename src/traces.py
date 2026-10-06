"""Lecture des jetons qui composent une trace de session.

Une trace est une suite de jetons de deux sortes :
- les marqueurs temporels « tXX » (XX = nombre de secondes écoulées) ;
- les actions, de la forme « libellé(écran)<configuration>$chaîne$1 », où
  chaque partie après le libellé est facultative et le « 1 » final signale
  le mode modification.

Ce module est partagé par l'exploration (étape 2) et le pré-traitement (étape 3).
"""

import re
from functools import lru_cache
from typing import NamedTuple

MOTIF_MARQUEUR = re.compile(r"t(\d+)")

# Les groupes non gourmands « .*? » combinés à l'ancrage de fullmatch permettent
# de lire les écrans contenant des parenthèses imbriquées (traces d'erreur Java
# comme « loadStatList (StatSelectionPart.java:514) »).
MOTIF_ACTION = re.compile(
    r"(?P<action>[^(<$]*?)"
    r"(?:\((?P<ecran>.*?)\))?"
    r"(?:<(?P<conf>.*?)>)?"
    r"(?:\$(?P<chaine>.*?)\$)?"
    r"(?P<modif>1)?"
)


class Action(NamedTuple):
    """Action décomposée ; les parties absentes valent None."""

    action: str
    ecran: str | None
    conf: str | None
    chaine: str | None
    modif: bool


def est_marqueur_temps(jeton: str) -> bool:
    """Indique si le jeton est un marqueur temporel « tXX »."""
    return MOTIF_MARQUEUR.fullmatch(jeton) is not None


def valeur_marqueur(jeton: str) -> int:
    """Renvoie le nombre de secondes d'un marqueur « tXX » (ex. « t35 » -> 35)."""
    return int(MOTIF_MARQUEUR.fullmatch(jeton).group(1))


@lru_cache(maxsize=None)
def decomposer_action(jeton: str) -> Action | None:
    """Découpe un jeton d'action en libellé, écran, configuration, chaîne et mode.

    Le cache évite de réappliquer la regex aux mêmes jetons : il y a environ
    2,3 millions d'actions mais beaucoup moins de jetons distincts.

    Paramètres
    ----------
    jeton : str
        Jeton d'action brut (pas un marqueur temporel).

    Renvoie
    -------
    Action | None
        L'action décomposée, ou None si le jeton ne respecte pas la syntaxe
        décrite par Kaggle (cas isolé, traité au pré-traitement).
    """
    correspondance = MOTIF_ACTION.fullmatch(jeton)
    if correspondance is None:
        return None
    return Action(
        action=correspondance["action"],
        ecran=correspondance["ecran"],
        conf=correspondance["conf"],
        chaine=correspondance["chaine"],
        modif=correspondance["modif"] is not None,
    )
