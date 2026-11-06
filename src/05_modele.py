"""Étape 5 — Modélisation : comparaison de modèles et réglage des hyperparamètres.

Pour chaque modèle du catalogue (src/modeles.py) :
1. recherche des meilleurs hyperparamètres par validation croisée (GridSearchCV,
   critère : F1 macro) ;
2. calcul des scores « hors pli » : chaque session est prédite par le modèle
   entraîné sur les 3 autres plis, donc sans jamais l'avoir vue.

On teste ensuite la contrainte « un utilisateur = un navigateur » sur les
meilleurs modèles, puis on entraîne le modèle retenu sur tout le train.

Usage : python src/05_modele.py (après src/04_features.py ; environ 15 min)
"""

import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

from config import (
    DOSSIER_TABLEAUX,
    ENCODAGE,
    FEATURES_TRAIN,
    MODELE_FINAL,
    N_SPLITS,
    RANDOM_STATE,
    SCORES_HORS_PLI,
)
from features import colonnes_numeriques
from figures import configurer_style, tracer_scores
from modeles import ContrainteNavigateur, catalogue_modeles, construire_pipeline

SUFFIXE_CONTRAINTE = " + contrainte navigateur"
# La contrainte est testée sur les 2 meilleurs modèles : le meilleur (retenu
# s'il reste en tête) et un second pour vérifier que le gain se répète.
NB_MODELES_CONTRAINTS = 2


def regler_hyperparametres(
    pipeline,
    grille: dict,
    X: pd.DataFrame,
    y: pd.Series,
    validation: StratifiedKFold,
    n_jobs: int,
) -> tuple[dict, pd.DataFrame]:
    """Cherche la meilleure combinaison d'hyperparamètres (critère : F1 macro).

    Paramètres
    ----------
    pipeline : Pipeline
        Pipeline features + modèle.
    grille : dict
        Valeurs à tester, au format de GridSearchCV ; vide = rien à régler.
    X, y : données d'entraînement.
    validation : StratifiedKFold
        Découpage commun à tous les modèles.
    n_jobs : int
        Nombre d'entraînements en parallèle.

    Renvoie
    -------
    tuple[dict, pd.DataFrame]
        Meilleurs paramètres, et tableau de toutes les combinaisons testées
        (F1 macro moyen, écart-type, temps d'entraînement).
    """
    if not grille:
        return {}, pd.DataFrame()
    recherche = GridSearchCV(
        pipeline, grille, scoring="f1_macro", cv=validation, n_jobs=n_jobs, refit=False
    )
    recherche.fit(X, y)
    resultats = pd.DataFrame(recherche.cv_results_)[
        ["params", "mean_test_score", "std_test_score", "mean_fit_time"]
    ]
    return recherche.best_params_, resultats


def calculer_scores_hors_pli(
    modele, X: pd.DataFrame, y: pd.Series, validation: StratifiedKFold, n_jobs: int
) -> np.ndarray:
    """Scores de chaque session par le modèle entraîné sans elle (validation croisée).

    On utilise les probabilités quand le modèle en fournit, sinon les scores
    de décision (SVM) : les deux permettent de classer les utilisateurs du
    plus au moins probable, ce dont ont besoin les courbes ROC de l'étape 6.

    Renvoie
    -------
    np.ndarray
        Matrice sessions x utilisateurs (colonnes dans l'ordre alphabétique).
    """
    methode = "predict_proba" if hasattr(modele, "predict_proba") else "decision_function"
    return cross_val_predict(modele, X, y, cv=validation, method=methode, n_jobs=n_jobs)


def mesurer_par_pli(
    scores: np.ndarray, classes: np.ndarray, y: pd.Series, plis: list[np.ndarray]
) -> dict:
    """F1 macro, F1 pondéré et accuracy sur chaque pli de validation (moyenne ± écart-type).

    Paramètres
    ----------
    scores : np.ndarray
        Scores hors pli (sessions x utilisateurs).
    classes : np.ndarray
        Utilisateurs correspondant aux colonnes de scores.
    y : pd.Series
        Vrais utilisateurs.
    plis : list[np.ndarray]
        Indices des sessions de chaque pli de validation.
    """
    predictions = classes[scores.argmax(axis=1)]
    mesures = {"f1_macro": [], "f1_pondere": [], "accuracy": []}
    for indices in plis:
        vrais, predits = y.iloc[indices], predictions[indices]
        mesures["f1_macro"].append(f1_score(vrais, predits, average="macro", zero_division=0))
        mesures["f1_pondere"].append(
            f1_score(vrais, predits, average="weighted", zero_division=0)
        )
        mesures["accuracy"].append(accuracy_score(vrais, predits))
    resume = {}
    for nom, valeurs in mesures.items():
        resume[nom] = np.mean(valeurs)
        resume[f"{nom}_ecart_type"] = np.std(valeurs)
    return resume


def formater_parametres(parametres: dict) -> str:
    """Rend les hyperparamètres lisibles (« C=10, class_weight=balanced »)."""
    return ", ".join(
        f"{cle.removeprefix('modele__')}={valeur}" for cle, valeur in parametres.items()
    ) or "fixés (pas de grille)"


def comparer_modeles(
    X: pd.DataFrame, y: pd.Series, validation: StratifiedKFold
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Règle puis évalue chaque modèle du catalogue.

    Renvoie
    -------
    tuple
        Tableau comparatif, tableau des grilles, scores hors pli par modèle,
        et pipelines configurés (meilleurs paramètres, non ajustés).
    """
    numeriques = colonnes_numeriques(X)
    classes = np.sort(y.unique())
    plis = [validation_indices for _, validation_indices in validation.split(X, y)]
    lignes, grilles, scores_par_modele, pipelines = [], [], {}, {}

    for nom, spec in catalogue_modeles().items():
        debut = time.perf_counter()
        pipeline = construire_pipeline(
            spec["estimateur"], numeriques, spec.get("dense", False), spec.get("sequences", False)
        )
        n_jobs = spec.get("n_jobs", -1)
        meilleurs, resultats_grille = regler_hyperparametres(
            pipeline, spec["grille"], X, y, validation, n_jobs
        )
        pipeline.set_params(**meilleurs)
        scores = calculer_scores_hors_pli(pipeline, X, y, validation, n_jobs)

        ligne = {"modele": nom, "parametres": formater_parametres(meilleurs)}
        ligne |= mesurer_par_pli(scores, classes, y, plis)
        ligne["duree_s"] = time.perf_counter() - debut
        lignes.append(ligne)
        grilles.append(resultats_grille.assign(modele=nom))
        scores_par_modele[nom] = scores
        pipelines[nom] = pipeline
        print(
            f"{nom:<32} F1 macro {ligne['f1_macro']:.3f} ± {ligne['f1_macro_ecart_type']:.3f}"
            f"  accuracy {ligne['accuracy']:.3f}  [{ligne['parametres']}]"
            f"  ({ligne['duree_s']:.0f} s)",
            flush=True,
        )

    return pd.DataFrame(lignes), pd.concat(grilles), scores_par_modele, pipelines


def evaluer_contrainte_navigateur(
    comparaison: pd.DataFrame,
    scores_par_modele: dict,
    pipelines: dict,
    X: pd.DataFrame,
    y: pd.Series,
    validation: StratifiedKFold,
) -> pd.DataFrame:
    """Applique ContrainteNavigateur aux meilleurs modèles et les évalue.

    La contrainte fonctionne avec des probabilités comme avec des scores de
    décision (SVM) : on la teste donc sur les NB_MODELES_CONTRAINTS meilleurs
    modèles, quels qu'ils soient. Complète scores_par_modele et pipelines, et
    renvoie la comparaison avec une ligne supplémentaire par modèle contraint.
    """
    plis = [indices for _, indices in validation.split(X, y)]
    meilleurs = comparaison.sort_values("f1_macro", ascending=False).head(NB_MODELES_CONTRAINTS)
    lignes = []
    for meilleur, parametres in zip(meilleurs["modele"], meilleurs["parametres"]):
        nom_contraint = meilleur + SUFFIXE_CONTRAINTE
        debut = time.perf_counter()
        pipelines[nom_contraint] = ContrainteNavigateur(pipelines[meilleur])
        n_jobs = catalogue_modeles()[meilleur].get("n_jobs", -1)
        scores_par_modele[nom_contraint] = calculer_scores_hors_pli(
            pipelines[nom_contraint], X, y, validation, n_jobs
        )
        ligne = {"modele": nom_contraint, "parametres": parametres}
        ligne |= mesurer_par_pli(scores_par_modele[nom_contraint], np.sort(y.unique()), y, plis)
        ligne["duree_s"] = time.perf_counter() - debut
        lignes.append(ligne)
        print(
            f"{nom_contraint:<32} F1 macro {ligne['f1_macro']:.3f} ± "
            f"{ligne['f1_macro_ecart_type']:.3f}  accuracy {ligne['accuracy']:.3f}",
            flush=True,
        )
    return pd.concat([comparaison, pd.DataFrame(lignes)], ignore_index=True)


def sauvegarder_resultats(
    comparaison: pd.DataFrame,
    grilles: pd.DataFrame,
    scores_par_modele: dict,
    X: pd.DataFrame,
    y: pd.Series,
    validation: StratifiedKFold,
) -> None:
    """Écrit les tableaux, la figure de comparaison et les scores hors pli (pour l'étape 6)."""
    comparaison.to_csv(
        DOSSIER_TABLEAUX / "comparaison_modeles.csv", encoding=ENCODAGE, index=False
    )
    grilles.to_csv(
        DOSSIER_TABLEAUX / "grilles_hyperparametres.csv", encoding=ENCODAGE, index=False
    )
    tracer_scores(
        comparaison,
        comparaison["modele"],
        titre="Comparaison des modèles (hyperparamètres réglés)",
        legende_x=f"Score hors pli, validation croisée stratifiée ({N_SPLITS} plis)",
        nom="fig10_comparaison_modeles",
    )
    joblib.dump(
        {
            "y": y.to_numpy(),
            "classes": np.sort(y.unique()),
            "navigateur": X["navigateur"].to_numpy(),
            "plis": [indices for _, indices in validation.split(X, y)],
            "scores": scores_par_modele,
        },
        SCORES_HORS_PLI,
    )


def main() -> None:
    """Compare les modèles, teste la contrainte navigateur et entraîne le modèle final."""
    configurer_style()
    # Avertissements de convergence (SVM, MLP) : les scores restent mesurés et
    # comparés honnêtement, et les signaler à chaque pli encombrerait la sortie.
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    features = pd.read_pickle(FEATURES_TRAIN)
    X, y = features, features["util"]
    validation = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    comparaison, grilles, scores_par_modele, pipelines = comparer_modeles(X, y, validation)
    comparaison = evaluer_contrainte_navigateur(
        comparaison, scores_par_modele, pipelines, X, y, validation
    ).sort_values("f1_macro", ascending=False, ignore_index=True)
    sauvegarder_resultats(comparaison, grilles, scores_par_modele, X, y, validation)

    retenu = comparaison["modele"].iloc[0]
    modele_final = pipelines[retenu].fit(X, y)
    # Compression : 500 arbres stockent 247 valeurs (presque toutes nulles) par
    # nœud, soit 1,5 Go sans compression.
    joblib.dump({"nom": retenu, "modele": modele_final}, MODELE_FINAL, compress=3)
    print(f"Modèle retenu : {retenu} — entraîné sur tout le train et sauvegardé.")


if __name__ == "__main__":
    main()
