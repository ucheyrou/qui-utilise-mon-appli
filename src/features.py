"""Construction des features (module partagé par les étapes 4, 5 et 7).

Deux sortes de features :

1. Les features « simples », construites à la main et calculées session par
   session (comptages, proportions, rythme, écrans principaux). Elles ne
   dépendent d'aucune autre session : on peut les calculer une fois pour toutes.
2. Les sacs de jetons pondérés par TF-IDF :
   - jetons par famille (actions, écrans, configurations, chaînes,
     enchaînements de libellés) ;
   - « séquences d'actions complètes » : chaque action gardée entière
     (libellé + écran + configuration + chaîne + mode) et chaque couple
     d'actions complètes consécutives.
   Les poids IDF dépendent du jeu sur lequel on les apprend : ils sont donc
   calculés dans le pipeline scikit-learn, uniquement sur les plis
   d'entraînement (pas de fuite).

Voir explications/04_features.md.
"""

from collections import Counter
from functools import partial

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

AUCUN = "aucun"  # valeur des colonnes catégorielles quand la session n'a pas d'écran

# Action comptée seule dans le premier essai : l'usage des raccourcis clavier
# est une habitude personnelle, indépendante du métier de l'utilisateur.
ACTION_TEST_SIMPLE = "Raccourci"
COLONNE_COMPTAGE_SIMPLE = f"nb_{ACTION_TEST_SIMPLE.lower()}"

COLONNES_RYTHME = [
    "log_nb_actions",
    "log_duree_s",
    "vitesse_apm",
    "actions_par_fenetre_moy",
    "actions_par_fenetre_ecart_type",
    "actions_par_fenetre_max",
]
COLONNES_CONTENU = [
    "nb_ecrans_distincts",
    "nb_conf_distinctes",
    "nb_chaines_distinctes",
    "part_modif",
    "part_actions_avec_ecran",
]
COLONNES_CATEGORIELLES = ["ecran_principal", "premier_ecran", "chaine_principale"]
PREFIXE_PART_ACTION = "part_action_"

FAMILLES_JETONS_COMPLETES = ("action", "ecran", "conf", "chaine", "bigramme")


def features_rythme(temps: np.ndarray) -> dict:
    """Décrit le rythme de travail à partir des temps des actions.

    Paramètres
    ----------
    temps : np.ndarray
        Pour chaque action, fin (en s) de la fenêtre de 5 s où elle a eu lieu.

    Renvoie
    -------
    dict
        Volume, durée d'activité, vitesse et nombre d'actions par fenêtre de
        5 s. Les grandeurs très étalées (nombre d'actions, durée) sont passées
        au logarithme. Les marqueurs se suivant toujours de 5 en 5, les temps
        morts ne sont pas visibles : aucune feature de pause n'est possible.
    """
    nb_actions = len(temps)
    if nb_actions == 0:
        return {colonne: 0.0 for colonne in COLONNES_RYTHME}

    duree_s = int(temps.max())
    actions_par_fenetre = np.unique(temps, return_counts=True)[1]
    return {
        "log_nb_actions": np.log1p(nb_actions),
        "log_duree_s": np.log1p(duree_s),
        "vitesse_apm": nb_actions / duree_s * 60,
        "actions_par_fenetre_moy": actions_par_fenetre.mean(),
        "actions_par_fenetre_ecart_type": actions_par_fenetre.std(),
        "actions_par_fenetre_max": actions_par_fenetre.max(),
    }


def features_contenu(actions: list) -> dict:
    """Décrit ce que fait l'utilisateur : actions, écrans, mode modification.

    Paramètres
    ----------
    actions : list[Action]
        Actions décomposées de la session.

    Renvoie
    -------
    dict
        Comptage de l'action test, part de chaque libellé d'action, diversité
        des écrans / configurations / chaînes, part en mode modification et
        écrans ou chaînes les plus caractéristiques (catégoriels).
    """
    nb_actions = max(len(actions), 1)
    libelles = Counter(action.action for action in actions)
    ecrans = Counter(action.ecran for action in actions if action.ecran)
    confs = {action.conf for action in actions if action.conf}
    chaines = Counter(action.chaine for action in actions if action.chaine)

    features = {
        COLONNE_COMPTAGE_SIMPLE: libelles[ACTION_TEST_SIMPLE],
        "nb_ecrans_distincts": len(ecrans),
        "nb_conf_distinctes": len(confs),
        "nb_chaines_distinctes": len(chaines),
        "part_modif": sum(action.modif for action in actions) / nb_actions,
        "part_actions_avec_ecran": sum(ecrans.values()) / nb_actions,
        "ecran_principal": ecrans.most_common(1)[0][0] if ecrans else AUCUN,
        "premier_ecran": next((a.ecran for a in actions if a.ecran), AUCUN),
        "chaine_principale": chaines.most_common(1)[0][0] if chaines else AUCUN,
    }
    for libelle, nombre in libelles.items():
        features[f"{PREFIXE_PART_ACTION}{libelle}"] = nombre / nb_actions
    return features


def calculer_features_simples(
    sessions: pd.DataFrame, colonnes_parts: list[str] | None = None
) -> pd.DataFrame:
    """Calcule les features simples de toutes les sessions d'un jeu.

    Paramètres
    ----------
    sessions : pd.DataFrame
        Sessions pré-traitées (colonnes « navigateur », « actions », « temps »).
    colonnes_parts : list[str] | None
        Colonnes « part_action_<libellé> » à produire. None pour le train (on garde
        les libellés rencontrés) ; pour le test, on passe celles du train afin
        que les deux tableaux aient exactement les mêmes colonnes.

    Renvoie
    -------
    pd.DataFrame
        Une ligne par session : navigateur, features numériques, features
        catégorielles et la colonne « actions » (utilisée par le TF-IDF).
    """
    lignes = [
        features_rythme(temps) | features_contenu(actions)
        for actions, temps in zip(sessions["actions"], sessions["temps"])
    ]
    features = pd.DataFrame(lignes, index=sessions.index)

    parts = colonnes_parts_actions(features)
    if colonnes_parts is None:
        colonnes_parts = sorted(parts)
    features = features.drop(columns=parts).join(
        features[parts].reindex(columns=colonnes_parts).fillna(0.0)
    )

    colonnes_identite = [c for c in ("util", "RowId") if c in sessions]
    return pd.concat(
        [sessions[colonnes_identite + ["navigateur"]], features, sessions[["actions"]]],
        axis=1,
    )


def colonnes_parts_actions(features: pd.DataFrame) -> list[str]:
    """Renvoie les colonnes « part_action_<libellé> » (répartition des libellés)."""
    return [colonne for colonne in features if colonne.startswith(PREFIXE_PART_ACTION)]


def colonnes_numeriques(features: pd.DataFrame) -> list[str]:
    """Renvoie toutes les features simples numériques (essai E8 de l'étape 4)."""
    return (
        [COLONNE_COMPTAGE_SIMPLE] + COLONNES_RYTHME + colonnes_parts_actions(features)
        + COLONNES_CONTENU
    )


def jetons_session(actions: list, familles=FAMILLES_JETONS_COMPLETES) -> list[str]:
    """Transforme une session en « document » de jetons pour le TF-IDF.

    Chaque jeton est préfixé par sa famille pour qu'un écran et une
    configuration de même nom ne soient pas confondus.

    Paramètres
    ----------
    actions : list[Action]
        Actions décomposées de la session.
    familles : tuple[str, ...]
        Familles de jetons à produire parmi « action » (libellé, suffixé de
        « |modif » en mode modification), « ecran », « conf », « chaine » et
        « bigramme » (deux libellés d'action consécutifs).

    Renvoie
    -------
    list[str]
        Jetons de la session, avec répétitions (le TF compte les occurrences).
    """
    jetons = []
    for action in actions:
        if "action" in familles:
            jetons.append(f"action={action.action}{'|modif' if action.modif else ''}")
        if "ecran" in familles and action.ecran:
            jetons.append(f"ecran={action.ecran}")
        if "conf" in familles and action.conf:
            jetons.append(f"conf={action.conf}")
        if "chaine" in familles and action.chaine:
            jetons.append(f"chaine={action.chaine}")
    if "bigramme" in familles:
        jetons.extend(
            f"bigramme={avant.action}>{apres.action}"
            for avant, apres in zip(actions, actions[1:])
        )
    return jetons


def action_complete(action) -> str:
    """Écrit une action entière en un seul jeton (« libellé|écran|conf|chaîne|mode »).

    Contrairement aux jetons par famille, on ne sépare pas l'action de son
    contexte : « Saisie dans un champ » sur l'écran des commandes et sur
    l'écran des tiers donnent deux jetons différents.
    """
    return "|".join([
        action.action,
        action.ecran or "",
        action.conf or "",
        action.chaine or "",
        "modif" if action.modif else "",
    ])


def jetons_actions_completes(actions: list) -> list[str]:
    """Transforme une session en jetons « actions complètes » et leurs bigrammes.

    Paramètres
    ----------
    actions : list[Action]
        Actions décomposées de la session.

    Renvoie
    -------
    list[str]
        Une action complète par action de la session, puis un jeton
        « A >> B » pour chaque couple d'actions complètes consécutives (le
        geste précis enchaîné par l'utilisateur, dans son contexte).
    """
    completes = [action_complete(action) for action in actions]
    return completes + [f"{avant} >> {apres}" for avant, apres in zip(completes, completes[1:])]


def construire_preprocesseur(
    numeriques: list[str] | tuple = (),
    categorielles: list[str] | tuple = (),
    familles_jetons: tuple[str, ...] | None = None,
    actions_completes: bool = False,
) -> ColumnTransformer:
    """Assemble la transformation des colonnes en matrice de features.

    - navigateur : toujours inclus, encodé en one-hot (pas d'ordre entre navigateurs) ;
    - numériques : centrées-réduites (utile aux modèles linéaires, neutre pour les arbres) ;
    - catégorielles : one-hot, les modalités vues une seule fois sont regroupées ;
    - traces : TF-IDF sur les jetons de jetons_session, si familles_jetons est donné ;
    - sequences : TF-IDF sur les jetons de jetons_actions_completes, si
      actions_completes est vrai. C'est un second TF-IDF (et non une famille de
      plus) pour que chaque bloc soit normalisé séparément : les très nombreux
      bigrammes d'actions complètes n'écrasent pas les jetons par famille.

    Paramètres
    ----------
    numeriques, categorielles : list[str]
        Colonnes du tableau de features à utiliser.
    familles_jetons : tuple[str, ...] | None
        Familles de jetons pour le TF-IDF ; None pour ne pas l'utiliser.
    actions_completes : bool
        True pour ajouter le TF-IDF des séquences d'actions complètes.

    Renvoie
    -------
    ColumnTransformer
        Transformation non ajustée, à placer en tête d'un Pipeline.
    """
    transformations = [
        ("navigateur", OneHotEncoder(handle_unknown="ignore"), ["navigateur"])
    ]
    if numeriques:
        transformations.append(("numeriques", StandardScaler(), list(numeriques)))
    if categorielles:
        transformations.append(
            (
                "categorielles",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=2),
                list(categorielles),
            )
        )
    if familles_jetons:
        vectoriseur = TfidfVectorizer(
            analyzer=partial(jetons_session, familles=familles_jetons),
            sublinear_tf=True,
            min_df=2,
        )
        transformations.append(("traces", vectoriseur, "actions"))
    if actions_completes:
        vectoriseur = TfidfVectorizer(
            analyzer=jetons_actions_completes, sublinear_tf=True, min_df=2
        )
        transformations.append(("sequences", vectoriseur, "actions"))
    return ColumnTransformer(transformations)
