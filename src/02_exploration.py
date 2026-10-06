"""Étape 2 — Exploration des données.

Résume chaque session par quelques indicateurs (nombre d'actions, durée,
vitesse, écrans visités...), puis étudie la cible, les navigateurs, les
actions et les écrans. Produit les figures (outputs/figures/) et tableaux
(outputs/tableaux/) utilisés dans le rapport, et affiche les chiffres clés.

Usage : python src/02_exploration.py (après src/01_chargement.py)
"""

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from config import (
    DOSSIER_FIGURES,
    DOSSIER_TABLEAUX,
    ENCODAGE,
    SESSIONS_TEST,
    SESSIONS_TRAIN,
)
from figures import (
    COULEUR_BOITE,
    COULEUR_PRINCIPALE as COULEUR_TRAIN,
    COULEUR_SECONDAIRE as COULEUR_TEST,
    ENCRE_SECONDAIRE,
    RAMPE_BLEUE,
    configurer_style,
    sauvegarder_figure,
)
from traces import decomposer_action, est_marqueur_temps, valeur_marqueur

# Les marqueurs sont espacés de 5 s : c'est la durée minimale qu'on attribue
# à une session pour éviter une division par zéro dans le calcul de la vitesse.
DUREE_MINIMALE_S = 5
SEUIL_SESSION_COURTE = 10  # en nombre d'actions
NB_UTILISATEURS_PROFILS = 6
NB_UTILISATEURS_SIGNATURES = 15
NB_ACTIONS_SIGNATURES = 15
TRANCHES_UTILISATEURS_PAR_ECRAN = [0, 1, 2, 5, 10, 25, 50, 100, 250]


def charger_sessions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge les sessions produites par l'étape 1.

    Renvoie
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Sessions d'entraînement et de test.
    """
    return pd.read_pickle(SESSIONS_TRAIN), pd.read_pickle(SESSIONS_TEST)


def resumer_session(traces: list[str]) -> dict:
    """Calcule les indicateurs descriptifs d'une session.

    Paramètres
    ----------
    traces : list[str]
        Jetons de la session, dans l'ordre.

    Renvoie
    -------
    dict
        Indicateurs numériques (nombre d'actions, durée en secondes, vitesse en
        actions par minute, etc.), ainsi que le compteur des libellés d'action
        et l'ensemble des écrans visités, réutilisés pour les agrégats.
    """
    marqueurs = []
    compte_actions = Counter()
    ecrans = set()
    nb_modif = 0
    nb_non_decomposables = 0

    for jeton in traces:
        if est_marqueur_temps(jeton):
            marqueurs.append(valeur_marqueur(jeton))
            continue
        action = decomposer_action(jeton)
        if action is None:
            nb_non_decomposables += 1
            continue
        compte_actions[action.action] += 1
        nb_modif += action.modif
        if action.ecran:
            ecrans.add(action.ecran)

    nb_actions = sum(compte_actions.values()) + nb_non_decomposables
    # Les actions qui suivent le dernier marqueur ont eu lieu dans les 5 s
    # suivantes : la dernière valeur est donc une estimation par défaut.
    duree_s = max(marqueurs, default=0)
    nb_marqueurs_repetes = sum(a == b for a, b in zip(marqueurs, marqueurs[1:]))

    return {
        "nb_actions": nb_actions,
        "nb_marqueurs": len(marqueurs),
        "nb_marqueurs_repetes": nb_marqueurs_repetes,
        "duree_s": duree_s,
        "vitesse_apm": nb_actions / max(duree_s, DUREE_MINIMALE_S) * 60,
        "nb_ecrans_distincts": len(ecrans),
        "part_modif": nb_modif / nb_actions if nb_actions else 0.0,
        "nb_non_decomposables": nb_non_decomposables,
        "compte_actions": compte_actions,
        "ecrans": ecrans,
    }


def construire_profils(sessions: pd.DataFrame) -> pd.DataFrame:
    """Applique resumer_session à chaque session.

    Paramètres
    ----------
    sessions : pd.DataFrame
        Sessions de l'étape 1 (colonne « traces »).

    Renvoie
    -------
    pd.DataFrame
        Une ligne par session : colonnes d'origine (hors traces) + indicateurs.
    """
    indicateurs = pd.DataFrame(sessions["traces"].map(resumer_session).tolist())
    return pd.concat([sessions.drop(columns="traces"), indicateurs], axis=1)


def compter_actions(profils: pd.DataFrame) -> pd.DataFrame:
    """Compte chaque libellé d'action sur l'ensemble des sessions.

    Renvoie
    -------
    pd.DataFrame
        Colonnes action, occurrences, part (en %) et nb_sessions (sessions
        contenant l'action au moins une fois), triées par occurrences.
    """
    occurrences = Counter()
    presence = Counter()
    for compte in profils["compte_actions"]:
        occurrences.update(compte)
        presence.update(compte.keys())

    tableau = pd.DataFrame(
        {
            "action": list(occurrences),
            "occurrences": list(occurrences.values()),
            "nb_sessions": [presence[action] for action in occurrences],
        }
    )
    tableau["part"] = 100 * tableau["occurrences"] / tableau["occurrences"].sum()
    return tableau.sort_values("occurrences", ascending=False, ignore_index=True)


def compter_utilisateurs_par_ecran(profils_train: pd.DataFrame) -> pd.Series:
    """Compte, pour chaque écran, le nombre d'utilisateurs distincts qui l'ont visité.

    Un écran visité par un seul utilisateur est un indice très fort de son
    identité : cette mesure guide le choix des features à l'étape 4.

    Renvoie
    -------
    pd.Series
        Nombre d'utilisateurs distincts, indexé par écran, trié décroissant.
    """
    utilisateurs = {}
    for util, ecrans in zip(profils_train["util"], profils_train["ecrans"]):
        for ecran in ecrans:
            utilisateurs.setdefault(ecran, set()).add(util)
    nombres = {ecran: len(utils) for ecran, utils in utilisateurs.items()}
    return pd.Series(nombres, name="nb_utilisateurs").sort_values(ascending=False)


def calculer_signatures(
    profils_train: pd.DataFrame, utilisateurs: list[str], actions: list[str]
) -> pd.DataFrame:
    """Calcule la part moyenne (en %) de chaque action dans les sessions d'un utilisateur.

    Paramètres
    ----------
    profils_train : pd.DataFrame
        Profils des sessions d'entraînement.
    utilisateurs : list[str]
        Utilisateurs à afficher (lignes).
    actions : list[str]
        Actions à afficher (colonnes).

    Renvoie
    -------
    pd.DataFrame
        Matrice utilisateurs x actions.
    """
    lignes = {}
    for util in utilisateurs:
        comptes = profils_train.loc[profils_train["util"] == util, "compte_actions"]
        parts = [
            [compte[action] / max(sum(compte.values()), 1) for action in actions]
            for compte in comptes
        ]
        lignes[util] = 100 * np.mean(parts, axis=0)
    return pd.DataFrame.from_dict(lignes, orient="index", columns=actions)


def tracer_sessions_par_utilisateur(profils_train: pd.DataFrame) -> None:
    """Histogramme du nombre de sessions par utilisateur (déséquilibre des classes)."""
    sessions_par_util = profils_train["util"].value_counts()
    figure, axe = plt.subplots(figsize=(7, 3.5))
    axe.hist(
        sessions_par_util,
        bins=np.arange(0, sessions_par_util.max() + 3, 2),
        color=COULEUR_TRAIN,
        edgecolor="white",
        linewidth=1,
    )
    mediane = sessions_par_util.median()
    axe.axvline(mediane, color=ENCRE_SECONDAIRE, linestyle="--", linewidth=1)
    axe.annotate(
        f"médiane = {mediane:.0f}",
        xy=(mediane, axe.get_ylim()[1] * 0.92),
        xytext=(-4, 0),
        textcoords="offset points",
        ha="right",
        color=ENCRE_SECONDAIRE,
    )
    axe.set_title("Nombre de sessions par utilisateur (train)")
    axe.set_xlabel("Sessions d'un utilisateur")
    axe.set_ylabel("Nombre d'utilisateurs")
    sauvegarder_figure(figure, "fig01_sessions_par_utilisateur")


def tracer_navigateurs(profils_train: pd.DataFrame, profils_test: pd.DataFrame) -> None:
    """Répartition des navigateurs (train vs test) et nombre d'utilisateurs par navigateur."""
    ordre = profils_train["navigateur"].value_counts().index
    parts_train = 100 * profils_train["navigateur"].value_counts(normalize=True)[ordre]
    parts_test = 100 * profils_test["navigateur"].value_counts(normalize=True).reindex(
        ordre, fill_value=0
    )
    utils_par_nav = profils_train.groupby("navigateur")["util"].nunique()[ordre]

    figure, (axe_parts, axe_utils) = plt.subplots(1, 2, figsize=(10, 3.5))
    positions = np.arange(len(ordre))
    hauteur = 0.38
    for decalage, parts, couleur, libelle in [
        (-hauteur / 2, parts_train, COULEUR_TRAIN, "train"),
        (hauteur / 2, parts_test, COULEUR_TEST, "test"),
    ]:
        barres = axe_parts.barh(
            positions + decalage, parts, height=hauteur - 0.04, color=couleur, label=libelle
        )
        axe_parts.bar_label(barres, fmt="%.1f %%", padding=3, fontsize=7, color=ENCRE_SECONDAIRE)
    axe_parts.set_yticks(positions, ordre)
    axe_parts.invert_yaxis()
    axe_parts.set_xlabel("Part des sessions (%)")
    axe_parts.set_title("Navigateur des sessions")
    axe_parts.legend(loc="lower right")
    axe_parts.grid(axis="y", visible=False)

    barres = axe_utils.barh(positions, utils_par_nav, height=0.6, color=COULEUR_TRAIN)
    axe_utils.bar_label(barres, padding=3, fontsize=7, color=ENCRE_SECONDAIRE)
    axe_utils.set_yticks(positions, ordre)
    axe_utils.invert_yaxis()
    axe_utils.set_xlabel("Nombre d'utilisateurs distincts")
    axe_utils.set_title("Utilisateurs par navigateur (train)")
    axe_utils.grid(axis="y", visible=False)
    sauvegarder_figure(figure, "fig02_navigateurs")


def tracer_distributions_sessions(
    profils_train: pd.DataFrame, profils_test: pd.DataFrame
) -> None:
    """Distributions du nombre d'actions, de la durée et de la vitesse (train vs test).

    Les classes sont logarithmiques : on trace la part des sessions par classe
    (et non une densité, qui gonflerait artificiellement les classes étroites).
    """
    indicateurs = [
        ("nb_actions", "Nombre d'actions", 1, 1),
        ("duree_s", "Durée (minutes)", 1 / 60, DUREE_MINIMALE_S),
        ("vitesse_apm", "Vitesse (actions / minute)", 1, 1),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for axe, (colonne, libelle, echelle, minimum) in zip(axes, indicateurs):
        valeurs_train = profils_train[colonne].clip(lower=minimum) * echelle
        valeurs_test = profils_test[colonne].clip(lower=minimum) * echelle
        bornes = np.logspace(
            np.log10(min(valeurs_train.min(), valeurs_test.min())),
            np.log10(max(valeurs_train.max(), valeurs_test.max())),
            30,
        )
        for valeurs, couleur, nom in [
            (valeurs_train, COULEUR_TRAIN, "train"),
            (valeurs_test, COULEUR_TEST, "test"),
        ]:
            axe.hist(
                valeurs, bins=bornes, weights=np.full(len(valeurs), 100 / len(valeurs)),
                histtype="step", linewidth=2, color=couleur, label=nom,
            )
        axe.set_xscale("log")
        axe.set_xlabel(f"{libelle} — échelle log")
        axe.set_title(libelle)
    axes[0].set_ylabel("Part des sessions (%)")
    axes[0].legend()
    sauvegarder_figure(figure, "fig03_distributions_sessions")


def tracer_frequence_actions(actions: pd.DataFrame) -> None:
    """Fréquence des 35 libellés d'action (échelle log)."""
    figure, axe = plt.subplots(figsize=(8, 7.5))
    axe.barh(actions["action"], actions["occurrences"], height=0.7, color=COULEUR_TRAIN)
    axe.invert_yaxis()
    axe.set_xscale("log")
    axe.set_xlabel("Occurrences dans le train (échelle log)")
    axe.set_title("Fréquence des types d'action")
    axe.grid(axis="y", visible=False)
    axe.tick_params(axis="y", labelsize=7)
    sauvegarder_figure(figure, "fig04_frequence_actions")


def tracer_specificite_ecrans(utilisateurs_par_ecran: pd.Series) -> None:
    """Combien d'utilisateurs distincts visitent chaque écran ?"""
    tranches = pd.cut(utilisateurs_par_ecran, bins=TRANCHES_UTILISATEURS_PAR_ECRAN)
    effectifs = tranches.value_counts().sort_index()
    libelles = ["1", "2", "3-5", "6-10", "11-25", "26-50", "51-100", "101-247"]

    figure, axe = plt.subplots(figsize=(7, 3.5))
    barres = axe.bar(libelles, effectifs.values, width=0.7, color=COULEUR_TRAIN)
    axe.bar_label(barres, padding=2, fontsize=7, color=ENCRE_SECONDAIRE)
    axe.set_xlabel("Nombre d'utilisateurs distincts ayant visité l'écran")
    axe.set_ylabel("Nombre d'écrans")
    axe.set_title("Spécificité des écrans (train)")
    axe.grid(axis="x", visible=False)
    sauvegarder_figure(figure, "fig05_specificite_ecrans")


def tracer_profils_utilisateurs(profils_train: pd.DataFrame) -> None:
    """Boîtes à moustaches de trois indicateurs pour les utilisateurs les plus actifs."""
    utilisateurs = profils_train["util"].value_counts().index[:NB_UTILISATEURS_PROFILS]
    indicateurs = [
        ("vitesse_apm", "Vitesse (actions / minute)"),
        ("part_modif", "Part d'actions en mode modification"),
        ("nb_ecrans_distincts", "Écrans distincts par session"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for axe, (colonne, titre) in zip(axes, indicateurs):
        donnees = [
            profils_train.loc[profils_train["util"] == util, colonne] for util in utilisateurs
        ]
        axe.boxplot(
            donnees,
            tick_labels=utilisateurs,
            patch_artist=True,
            widths=0.6,
            boxprops={"facecolor": COULEUR_BOITE, "edgecolor": COULEUR_TRAIN},
            medianprops={"color": COULEUR_TRAIN, "linewidth": 2},
            whiskerprops={"color": COULEUR_TRAIN},
            capprops={"color": COULEUR_TRAIN},
            flierprops={"markeredgecolor": COULEUR_TRAIN, "markersize": 3},
        )
        axe.set_title(titre)
        axe.set_xlabel("Utilisateur")
        axe.grid(axis="x", visible=False)
    sauvegarder_figure(figure, "fig06_profils_utilisateurs")


def tracer_signatures(signatures: pd.DataFrame) -> None:
    """Carte de chaleur : part moyenne de chaque action pour quelques utilisateurs."""
    palette = LinearSegmentedColormap.from_list("bleus", RAMPE_BLEUE)
    figure, axe = plt.subplots(figsize=(10, 6))
    image = axe.imshow(signatures.values, cmap=palette, aspect="auto")
    axe.set_xticks(range(signatures.shape[1]), signatures.columns, rotation=45, ha="right")
    axe.set_yticks(range(signatures.shape[0]), signatures.index)
    axe.tick_params(labelsize=7)
    axe.grid(visible=False)
    for cote in axe.spines.values():
        cote.set_visible(False)
    barre = figure.colorbar(image, ax=axe, shrink=0.8)
    barre.set_label("Part moyenne dans la session (%)")
    axe.set_title(
        f"Répartition des actions des {NB_UTILISATEURS_SIGNATURES} utilisateurs les plus actifs"
    )
    axe.set_ylabel("Utilisateur")
    sauvegarder_figure(figure, "fig07_signatures_actions")


def afficher_chiffres_cles(
    profils_train: pd.DataFrame,
    profils_test: pd.DataFrame,
    actions: pd.DataFrame,
    utilisateurs_par_ecran: pd.Series,
) -> None:
    """Affiche les chiffres clés repris dans explications/02_exploration.md."""
    sessions_par_util = profils_train["util"].value_counts()
    navs_par_util = profils_train.groupby("util")["navigateur"].nunique()
    print("— Cible")
    print(
        f"  {len(sessions_par_util)} utilisateurs ; sessions par utilisateur : "
        f"min {sessions_par_util.min()}, médiane {sessions_par_util.median():.0f}, "
        f"max {sessions_par_util.max()} ; {(sessions_par_util < 10).sum()} "
        "utilisateurs ont moins de 10 sessions"
    )
    print(
        f"  Prédire toujours l'utilisateur majoritaire donnerait "
        f"{100 * sessions_par_util.max() / len(profils_train):.1f} % d'accuracy"
    )
    print("— Navigateur")
    print(
        f"  Utilisateurs n'ayant qu'un seul navigateur : "
        f"{(navs_par_util == 1).sum()} / {len(navs_par_util)}"
    )
    print(profils_train.groupby("navigateur")["util"].nunique().to_string())

    print("— Sessions (médianes train / test)")
    for colonne in ["nb_actions", "duree_s", "vitesse_apm", "nb_ecrans_distincts", "part_modif"]:
        print(
            f"  {colonne:<20} {profils_train[colonne].median():>8.2f} "
            f"{profils_test[colonne].median():>8.2f}"
        )
    duree_max_h = profils_train["duree_s"].max() / 3600
    print(f"  Durée maximale (train) : {duree_max_h:.1f} h")

    print("— Anomalies")
    for nom, profils in [("train", profils_train), ("test", profils_test)]:
        courtes = (profils["nb_actions"] < SEUIL_SESSION_COURTE).sum()
        print(
            f"  {nom} : {courtes} sessions de moins de {SEUIL_SESSION_COURTE} actions, "
            f"{profils['nb_marqueurs_repetes'].sum()} marqueurs répétés "
            f"(dans {(profils['nb_marqueurs_repetes'] > 0).sum()} sessions), "
            f"{profils['nb_non_decomposables'].sum()} jetons non décomposables"
        )

    print("— Actions et écrans")
    print(f"  {len(actions)} libellés d'action distincts ; les 5 plus fréquents :")
    print(actions.head(5)[["action", "part"]].to_string(index=False, float_format="%.1f"))
    ecrans_test = set().union(*profils_test["ecrans"])
    print(
        f"  {len(utilisateurs_par_ecran)} écrans distincts en train, dont "
        f"{(utilisateurs_par_ecran == 1).sum()} vus par un seul utilisateur et "
        f"{(utilisateurs_par_ecran <= 5).sum()} par 5 utilisateurs au plus"
    )
    print(
        f"  Écrans du test absents du train : "
        f"{len(ecrans_test - set(utilisateurs_par_ecran.index))} / {len(ecrans_test)}"
    )


def sauvegarder_tableaux(
    profils_train: pd.DataFrame, actions: pd.DataFrame, utilisateurs_par_ecran: pd.Series
) -> None:
    """Écrit les tableaux d'exploration en CSV (UTF-8) dans outputs/tableaux/."""
    colonnes_numeriques = [
        "nb_actions", "duree_s", "vitesse_apm", "nb_ecrans_distincts", "part_modif",
    ]
    profils_train["util"].value_counts().rename_axis("util").reset_index(
        name="nb_sessions"
    ).to_csv(DOSSIER_TABLEAUX / "sessions_par_utilisateur.csv", encoding=ENCODAGE, index=False)
    actions.to_csv(DOSSIER_TABLEAUX / "frequence_actions.csv", encoding=ENCODAGE, index=False)
    utilisateurs_par_ecran.rename_axis("ecran").reset_index().to_csv(
        DOSSIER_TABLEAUX / "utilisateurs_par_ecran.csv", encoding=ENCODAGE, index=False
    )
    profils_train[colonnes_numeriques].describe().to_csv(
        DOSSIER_TABLEAUX / "statistiques_sessions_train.csv", encoding=ENCODAGE
    )


def main() -> None:
    """Enchaîne les calculs d'exploration, les figures et les tableaux."""
    DOSSIER_FIGURES.mkdir(parents=True, exist_ok=True)
    DOSSIER_TABLEAUX.mkdir(parents=True, exist_ok=True)
    configurer_style()

    sessions_train, sessions_test = charger_sessions()
    profils_train = construire_profils(sessions_train)
    profils_test = construire_profils(sessions_test)
    actions = compter_actions(profils_train)
    utilisateurs_par_ecran = compter_utilisateurs_par_ecran(profils_train)
    signatures = calculer_signatures(
        profils_train,
        utilisateurs=list(profils_train["util"].value_counts().index[:NB_UTILISATEURS_SIGNATURES]),
        actions=list(actions["action"].head(NB_ACTIONS_SIGNATURES)),
    )

    tracer_sessions_par_utilisateur(profils_train)
    tracer_navigateurs(profils_train, profils_test)
    tracer_distributions_sessions(profils_train, profils_test)
    tracer_frequence_actions(actions)
    tracer_specificite_ecrans(utilisateurs_par_ecran)
    tracer_profils_utilisateurs(profils_train)
    tracer_signatures(signatures)

    sauvegarder_tableaux(profils_train, actions, utilisateurs_par_ecran)
    afficher_chiffres_cles(profils_train, profils_test, actions, utilisateurs_par_ecran)
    print(f"Figures dans {DOSSIER_FIGURES}, tableaux dans {DOSSIER_TABLEAUX}")


if __name__ == "__main__":
    main()
