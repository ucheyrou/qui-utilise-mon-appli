"""Étape 6 — Évaluation des modèles (métriques, courbes ROC, diagnostic).

S'appuie sur les scores « hors pli » de l'étape 5 : chaque session y est
notée par un modèle qui ne l'a jamais vue, l'évaluation est donc honnête.

1. Métriques de décision (F1 macro / micro / pondéré, accuracy, top-k) et de
   classement (AUC ROC, précision moyenne) pour tous les modèles.
2. Courbes ROC « un contre tous » micro- et macro-moyennées, ROC par
   utilisateur, courbes précision-rappel.
3. Lecture « détection d'intrusion » : taux de détection à faible taux de
   fausses alertes.
4. Diagnostic du modèle retenu : F1 et AUC par utilisateur, confusions les plus
   fréquentes, courbe d'apprentissage, variables les plus importantes.

Usage : python src/06_evaluation.py (après src/05_modele.py ; environ 3 min)
"""

from collections import Counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.preprocessing import label_binarize

from config import (
    DOSSIER_TABLEAUX,
    ENCODAGE,
    FEATURES_TRAIN,
    MODELE_FINAL,
    N_SPLITS,
    RANDOM_STATE,
    SCORES_HORS_PLI,
)
from figures import (
    COULEUR_PRINCIPALE,
    COULEUR_SECONDAIRE,
    ENCRE_SECONDAIRE,
    PALETTE_CATEGORIELLE,
    configurer_style,
    sauvegarder_figure,
)
from modeles import N_JOBS_FORETS, ContrainteNavigateur

NB_MODELES_COURBES = 5
NB_UTILISATEURS_DIFFICILES = 3
NB_CONFUSIONS = 15
NB_VARIABLES_IMPORTANTES = 20
TAUX_FAUSSES_ALERTES_CIBLES = [0.001, 0.01]
TAILLES_APPRENTISSAGE = [0.25, 0.5, 0.75, 1.0]
# Grille logarithmique de taux de faux positifs : les AUC étant proches de 1,
# toute la différence entre modèles se joue aux très faibles taux de fausses alertes.
GRILLE_FPR = np.logspace(-5, 0, 400)


def charger_resultats() -> tuple[dict, dict, pd.DataFrame]:
    """Charge les scores hors pli, le modèle final et la comparaison de l'étape 5."""
    resultats = joblib.load(SCORES_HORS_PLI)
    final = joblib.load(MODELE_FINAL)
    comparaison = pd.read_csv(DOSSIER_TABLEAUX / "comparaison_modeles.csv", encoding=ENCODAGE)
    return resultats, final, comparaison


def mesurer(scores: np.ndarray, y: np.ndarray, classes: np.ndarray, Y: np.ndarray) -> dict:
    """Calcule toutes les métriques d'un modèle à partir de ses scores hors pli.

    Paramètres
    ----------
    scores : np.ndarray
        Scores sessions x utilisateurs (probabilités ou scores de décision).
    y : np.ndarray
        Vrais utilisateurs.
    classes : np.ndarray
        Utilisateurs correspondant aux colonnes de scores.
    Y : np.ndarray
        y binarisé (une colonne 0/1 par utilisateur).

    Renvoie
    -------
    dict
        Métriques de décision (sur la prédiction argmax) et de classement (sur
        les scores). Les AUC sont calculées « un contre tous » : macro =
        moyenne des AUC par utilisateur ; micro = AUC de toutes les paires
        (session, utilisateur) mises dans un seul problème binaire.
    """
    predictions = classes[scores.argmax(axis=1)]
    return {
        "f1_macro": f1_score(y, predictions, average="macro", zero_division=0),
        "f1_micro": f1_score(y, predictions, average="micro"),
        "f1_pondere": f1_score(y, predictions, average="weighted", zero_division=0),
        "accuracy": accuracy_score(y, predictions),
        "top3_accuracy": top_k_accuracy_score(y, scores, k=3, labels=classes),
        "top5_accuracy": top_k_accuracy_score(y, scores, k=5, labels=classes),
        "auc_macro": auc_par_utilisateur(Y, scores).mean(),
        "auc_micro": roc_auc_score(Y.ravel(), scores.ravel()),
        "precision_moyenne_macro": average_precision_score(Y, scores, average="macro"),
        "precision_moyenne_micro": average_precision_score(Y, scores, average="micro"),
    }


def auc_par_utilisateur(Y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """AUC ROC de chaque utilisateur dans le problème « cet utilisateur contre les autres »."""
    return np.array([roc_auc_score(Y[:, k], scores[:, k]) for k in range(Y.shape[1])])


def roc_micro(Y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Courbe ROC micro-moyenne : toutes les paires (session, utilisateur) ensemble.

    Renvoie
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Taux de faux positifs, taux de vrais positifs et seuils.
    """
    return roc_curve(Y.ravel(), scores.ravel())


def roc_macro(Y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Courbe ROC macro-moyenne : moyenne des courbes des utilisateurs sur GRILLE_FPR.

    Chaque utilisateur a sa propre courbe, avec ses propres seuils : on lit le
    taux de vrais positifs de chacune sur une grille commune de taux de faux
    positifs (interpolation), puis on fait la moyenne.

    Renvoie
    -------
    np.ndarray
        Taux de vrais positifs moyen pour chaque valeur de GRILLE_FPR.
    """
    tpr_interpoles = []
    for k in range(Y.shape[1]):
        fpr, tpr, _ = roc_curve(Y[:, k], scores[:, k])
        tpr_interpoles.append(np.interp(GRILLE_FPR, fpr, tpr))
    return np.mean(tpr_interpoles, axis=0)


def points_de_fonctionnement(Y: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """Taux de détection atteint pour quelques taux de fausses alertes (lecture « intrusion »).

    Scénario : une session prétend venir de l'utilisateur k ; on l'accepte si le
    score de k dépasse un seuil. Une vraie session de k refusée est une fausse
    alerte ; une session d'un autre utilisateur acceptée est une usurpation
    non détectée. C'est exactement la ROC micro-moyenne.

    Renvoie
    -------
    pd.DataFrame
        Pour chaque taux de fausses alertes visé (fraction des usurpations
        acceptées) : taux de sessions légitimes acceptées et seuil correspondant.
    """
    fpr, tpr, seuils = roc_micro(Y, scores)
    lignes = []
    for cible in TAUX_FAUSSES_ALERTES_CIBLES:
        indice = np.searchsorted(fpr, cible, side="right") - 1
        lignes.append({
            "usurpations_acceptees_max": cible,
            "sessions_legitimes_acceptees": tpr[indice],
            "seuil_score": seuils[indice],
        })
    return pd.DataFrame(lignes)


def tracer_roc_modeles(Y: np.ndarray, scores_par_modele: dict) -> None:
    """ROC micro- et macro-moyennes des meilleurs modèles, axe des faux positifs en log."""
    figure, (axe_micro, axe_macro) = plt.subplots(1, 2, figsize=(12, 4.8))
    for couleur, (nom, scores) in zip(PALETTE_CATEGORIELLE, scores_par_modele.items()):
        fpr, tpr, _ = roc_micro(Y, scores)
        axe_micro.plot(fpr, tpr, color=couleur, linewidth=2,
                       label=f"{nom} (AUC {auc(fpr, tpr):.4f})")
        tpr_macro = roc_macro(Y, scores)
        axe_macro.plot(GRILLE_FPR, tpr_macro, color=couleur, linewidth=2,
                       label=f"{nom} (AUC {auc_par_utilisateur(Y, scores).mean():.4f})")
    # Pour un utilisateur, il n'y a qu'environ 3 200 sessions négatives : sa courbe
    # n'a pas de point en dessous d'un taux de faux positifs de 3e-4, d'où une
    # borne plus haute pour la ROC macro que pour la ROC micro (800 000 paires).
    for axe, titre, borne in [
        (axe_micro, "ROC micro-moyenne", 1e-5),
        (axe_macro, "ROC macro-moyenne", 1e-4),
    ]:
        axe.plot(GRILLE_FPR, GRILLE_FPR, color=ENCRE_SECONDAIRE, linestyle="--", linewidth=1,
                 label="Hasard")
        axe.set_xscale("log")
        axe.set_xlim(borne, 1)
        axe.set_ylim(0, 1.02)
        axe.set_xlabel("Taux de faux positifs (échelle log)")
        axe.set_title(titre)
        axe.legend(loc="lower right", fontsize=7)
    axe_micro.set_ylabel("Taux de vrais positifs (rappel)")
    sauvegarder_figure(figure, "fig11_roc_modeles")


def tracer_roc_utilisateurs(
    Y: np.ndarray, scores: np.ndarray, classes: np.ndarray, aucs: np.ndarray
) -> None:
    """ROC des utilisateurs les plus difficiles à reconnaître et d'un utilisateur médian."""
    ordre = np.argsort(aucs)
    choisis = list(ordre[:NB_UTILISATEURS_DIFFICILES]) + [ordre[len(ordre) // 2]]
    figure, axe = plt.subplots(figsize=(6.5, 5))
    for couleur, k in zip(PALETTE_CATEGORIELLE, choisis):
        fpr, tpr, _ = roc_curve(Y[:, k], scores[:, k])
        statut = "médian" if k == choisis[-1] else "difficile"
        libelle = f"{classes[k]} — {statut} (AUC {aucs[k]:.3f}, {int(Y[:, k].sum())} sessions)"
        axe.plot(fpr, tpr, color=couleur, linewidth=2, drawstyle="steps-post", label=libelle)
    axe.plot([0, 1], [0, 1], color=ENCRE_SECONDAIRE, linestyle="--", linewidth=1, label="Hasard")
    axe.set_xlabel("Taux de faux positifs")
    axe.set_ylabel("Taux de vrais positifs (rappel)")
    axe.set_title("ROC par utilisateur (modèle retenu)")
    axe.legend(loc="lower right", fontsize=7)
    sauvegarder_figure(figure, "fig12_roc_utilisateurs")


def tracer_par_utilisateur(par_utilisateur: pd.DataFrame) -> None:
    """Distribution des AUC par utilisateur, et F1 par utilisateur selon son nombre de sessions."""
    figure, (axe_auc, axe_f1) = plt.subplots(1, 2, figsize=(12, 4))
    axe_auc.hist(par_utilisateur["auc"], bins=30, color=COULEUR_PRINCIPALE, edgecolor="white")
    axe_auc.set_xlabel("AUC ROC de l'utilisateur (un contre tous)")
    axe_auc.set_ylabel("Nombre d'utilisateurs")
    axe_auc.set_title("AUC par utilisateur")

    axe_f1.scatter(par_utilisateur["nb_sessions"], par_utilisateur["f1"], s=14,
                   color=COULEUR_PRINCIPALE, alpha=0.6, edgecolors="none")
    tranches = pd.cut(par_utilisateur["nb_sessions"], [0, 6, 10, 15, 25, 100])
    moyennes = par_utilisateur.groupby(tranches, observed=True).agg(
        nb_sessions=("nb_sessions", "median"), f1=("f1", "mean")
    )
    axe_f1.plot(moyennes["nb_sessions"], moyennes["f1"], color=COULEUR_SECONDAIRE,
                linewidth=2, marker="o", label="F1 moyen par tranche")
    axe_f1.set_xscale("log")
    graduations = [4, 6, 10, 15, 25, 40, 75]
    axe_f1.set_xticks(graduations, [str(g) for g in graduations])
    axe_f1.minorticks_off()
    axe_f1.set_xlabel("Nombre de sessions de l'utilisateur dans le train (échelle log)")
    axe_f1.set_ylabel("F1 de l'utilisateur")
    axe_f1.set_title("F1 par utilisateur selon son nombre de sessions")
    axe_f1.legend(loc="lower right")
    sauvegarder_figure(figure, "fig13_par_utilisateur")


def tracer_precision_rappel(Y: np.ndarray, scores_par_modele: dict) -> None:
    """Courbes précision-rappel micro-moyennes des meilleurs modèles."""
    figure, axe = plt.subplots(figsize=(6.5, 5))
    for couleur, (nom, scores) in zip(PALETTE_CATEGORIELLE, scores_par_modele.items()):
        precision, rappel, _ = precision_recall_curve(Y.ravel(), scores.ravel())
        axe.plot(rappel, precision, color=couleur, linewidth=2,
                 label=f"{nom} (PM {average_precision_score(Y.ravel(), scores.ravel()):.3f})")
    axe.axhline(Y.mean(), color=ENCRE_SECONDAIRE, linestyle="--", linewidth=1,
                label=f"Hasard (précision {Y.mean():.4f})")
    axe.set_xlabel("Rappel")
    axe.set_ylabel("Précision")
    axe.set_ylim(0, 1.02)
    axe.set_title("Précision-rappel micro-moyenne")
    axe.legend(loc="lower left", fontsize=7)
    sauvegarder_figure(figure, "fig14_precision_rappel")


def tracer_courbe_apprentissage(modele, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Courbe d'apprentissage : F1 macro en entraînement et en validation selon la taille du train.

    Renvoie
    -------
    pd.DataFrame
        Taille d'entraînement, F1 macro moyen en entraînement et en validation.
    """
    validation = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    tailles, scores_train, scores_valid = learning_curve(
        clone(modele), X, y, train_sizes=TAILLES_APPRENTISSAGE, cv=validation,
        scoring="f1_macro", shuffle=True, random_state=RANDOM_STATE, n_jobs=N_JOBS_FORETS,
    )
    courbe = pd.DataFrame({
        "nb_sessions_entrainement": tailles,
        "f1_macro_entrainement": scores_train.mean(axis=1),
        "f1_macro_validation": scores_valid.mean(axis=1),
        "f1_macro_validation_ecart_type": scores_valid.std(axis=1),
    })
    figure, axe = plt.subplots(figsize=(6.5, 4.5))
    axe.plot(courbe["nb_sessions_entrainement"], courbe["f1_macro_entrainement"],
             color=COULEUR_SECONDAIRE, linewidth=2, marker="o", label="Entraînement")
    axe.errorbar(courbe["nb_sessions_entrainement"], courbe["f1_macro_validation"],
                 yerr=courbe["f1_macro_validation_ecart_type"], color=COULEUR_PRINCIPALE,
                 linewidth=2, marker="o", capsize=3, label="Validation")
    axe.set_ylim(0, 1.05)
    axe.set_xlabel("Nombre de sessions d'entraînement")
    axe.set_ylabel("F1 macro")
    axe.set_title("Courbe d'apprentissage (modèle retenu)")
    axe.legend(loc="lower right")
    sauvegarder_figure(figure, "fig15_courbe_apprentissage")
    return courbe


def extraire_importances(modele) -> pd.Series | None:
    """Importance (diminution moyenne d'impureté) des variables du modèle final.

    Renvoie
    -------
    pd.Series | None
        Importance par variable, triée ; None si le modèle n'en fournit pas.
    """
    pipeline = modele.estimateur_ if isinstance(modele, ContrainteNavigateur) else modele
    classifieur = pipeline.named_steps["modele"]
    if not hasattr(classifieur, "feature_importances_"):
        return None
    noms = pipeline.named_steps["preprocesseur"].get_feature_names_out()
    return pd.Series(classifieur.feature_importances_, index=noms).sort_values(ascending=False)


def tracer_importances(importances: pd.Series) -> None:
    """Barres horizontales des variables les plus importantes du modèle final."""
    meilleures = importances.head(NB_VARIABLES_IMPORTANTES)
    libelles = [nom.split("__", 1)[1] for nom in meilleures.index]
    figure, axe = plt.subplots(figsize=(10, 6))
    axe.barh(libelles, meilleures.values, height=0.7, color=COULEUR_PRINCIPALE)
    axe.invert_yaxis()
    axe.set_xlabel("Importance (diminution moyenne de l'impureté de Gini)")
    axe.set_title(f"Les {NB_VARIABLES_IMPORTANTES} variables les plus importantes (modèle retenu)")
    axe.grid(axis="y", visible=False)
    axe.tick_params(axis="y", labelsize=7)
    sauvegarder_figure(figure, "fig16_importance_variables")


def lister_confusions(y: np.ndarray, predictions: np.ndarray) -> pd.DataFrame:
    """Couples (vrai utilisateur, utilisateur prédit) les plus souvent confondus."""
    erreurs = Counter(
        (vrai, predit) for vrai, predit in zip(y, predictions) if vrai != predit
    )
    return pd.DataFrame(
        [{"vrai": v, "predit": p, "nb_erreurs": n} for (v, p), n in erreurs.most_common()]
    )


def main() -> None:
    """Enchaîne métriques, courbes ROC et diagnostic du modèle retenu."""
    configurer_style()
    resultats, final, comparaison = charger_resultats()
    y, classes = resultats["y"], resultats["classes"]
    Y = label_binarize(y, classes=classes)

    # 1. Métriques de tous les modèles
    evaluation = pd.DataFrame([
        {"modele": nom} | mesurer(scores, y, classes, Y)
        for nom, scores in resultats["scores"].items()
    ]).sort_values("f1_macro", ascending=False, ignore_index=True)
    evaluation.to_csv(DOSSIER_TABLEAUX / "evaluation_modeles.csv", encoding=ENCODAGE,
                      index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(evaluation.round(4).to_string(index=False))

    # 2. Courbes ROC et précision-rappel des meilleurs modèles
    meilleurs = comparaison["modele"].head(NB_MODELES_COURBES)
    scores_meilleurs = {nom: resultats["scores"][nom] for nom in meilleurs}
    tracer_roc_modeles(Y, scores_meilleurs)
    tracer_precision_rappel(Y, scores_meilleurs)

    # 3. Lecture « détection d'intrusion » sur le modèle retenu
    scores_final = resultats["scores"][final["nom"]]
    points = points_de_fonctionnement(Y, scores_final)
    points.to_csv(DOSSIER_TABLEAUX / "points_de_fonctionnement.csv", encoding=ENCODAGE,
                  index=False)
    print(f"\nModèle retenu : {final['nom']}")
    print(points.round(4).to_string(index=False))

    # 4. Diagnostic par utilisateur
    predictions = classes[scores_final.argmax(axis=1)]
    aucs = auc_par_utilisateur(Y, scores_final)
    par_utilisateur = pd.DataFrame({
        "util": classes,
        "nb_sessions": Y.sum(axis=0),
        "f1": f1_score(y, predictions, labels=classes, average=None, zero_division=0),
        "auc": aucs,
    }).sort_values("f1")
    par_utilisateur.to_csv(DOSSIER_TABLEAUX / "metriques_par_utilisateur.csv",
                           encoding=ENCODAGE, index=False)
    tracer_roc_utilisateurs(Y, scores_final, classes, aucs)
    tracer_par_utilisateur(par_utilisateur)
    print(f"\nUtilisateurs avec F1 = 1 : {(par_utilisateur['f1'] == 1).sum()} / {len(classes)} ;"
          f" F1 < 0,5 : {(par_utilisateur['f1'] < 0.5).sum()}")
    print("Utilisateurs les moins bien reconnus :")
    print(par_utilisateur.head(5).round(3).to_string(index=False))

    confusions = lister_confusions(y, predictions)
    confusions.to_csv(DOSSIER_TABLEAUX / "confusions.csv", encoding=ENCODAGE, index=False)
    print(f"\n{len(confusions)} couples confondus ; les plus fréquents :")
    print(confusions.head(NB_CONFUSIONS).to_string(index=False))

    # 5. Sur-apprentissage et variables importantes
    features = pd.read_pickle(FEATURES_TRAIN)
    courbe = tracer_courbe_apprentissage(final["modele"], features, features["util"])
    courbe.to_csv(DOSSIER_TABLEAUX / "courbe_apprentissage.csv", encoding=ENCODAGE,
                  index=False)
    print("\nCourbe d'apprentissage :")
    print(courbe.round(3).to_string(index=False))

    importances = extraire_importances(final["modele"])
    if importances is not None:
        importances.rename("importance").rename_axis("variable").reset_index().to_csv(
            DOSSIER_TABLEAUX / "importance_variables.csv", encoding=ENCODAGE, index=False
        )
        tracer_importances(importances)
        print("\nVariables les plus importantes :")
        print(importances.head(10).round(4).to_string())


if __name__ == "__main__":
    main()
