"""Étape 4 — Construction des features et carnet d'essais.

1. Calcule les features simples (fonctions de src/features.py) pour le train
   et le test, et les sauvegarde avec la colonne « actions » pour le TF-IDF.
2. Classe les features simples selon leur pouvoir discriminant (test F de
   l'ANOVA, calculé sur le train uniquement).
3. Tient un « carnet d'essais » : on part du navigateur seul et on ajoute les
   familles de features une à une, en évaluant chaque essai avec le même
   modèle (forêt aléatoire) et la même validation croisée stratifiée. On mesure
   ainsi ce qu'apporte chaque idée.

Usage : python src/04_features.py (après src/03_pretraitement.py)
"""

import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline

from config import (
    DOSSIER_TABLEAUX,
    ENCODAGE,
    FEATURES_TEST,
    FEATURES_TRAIN,
    N_SPLITS,
    RANDOM_STATE,
    SESSIONS_PRETRAITEES_TEST,
    SESSIONS_PRETRAITEES_TRAIN,
)
from features import (
    ACTION_TEST_SIMPLE,
    COLONNE_COMPTAGE_SIMPLE,
    COLONNES_CATEGORIELLES,
    COLONNES_CONTENU,
    COLONNES_RYTHME,
    FAMILLES_JETONS_COMPLETES,
    PREFIXE_PART_ACTION,
    calculer_features_simples,
    colonnes_numeriques,
    colonnes_parts_actions,
    construire_preprocesseur,
)
from figures import COULEUR_PRINCIPALE, configurer_style, sauvegarder_figure, tracer_scores

NB_ARBRES = 300
NB_FEATURES_CLASSEMENT = 20


def definir_essais(colonnes_parts: list[str]) -> list[dict]:
    """Liste des essais du carnet, du plus simple au plus complet.

    Paramètres
    ----------
    colonnes_parts : list[str]
        Colonnes « part_action_<libellé> » du tableau de features.

    Renvoie
    -------
    list[dict]
        Pour chaque essai : code, description et arguments de
        construire_preprocesseur (numeriques, categorielles, familles_jetons,
        actions_completes).
    """
    comptage = [COLONNE_COMPTAGE_SIMPLE]
    rythme = comptage + COLONNES_RYTHME
    repartition = rythme + colonnes_parts
    contenu = repartition + COLONNES_CONTENU
    return [
        {"code": "E1", "description": "Navigateur seul"},
        {
            "code": "E2",
            "description": f"+ comptage d'une action ({ACTION_TEST_SIMPLE})",
            "numeriques": comptage,
        },
        {"code": "E3", "description": "+ rythme (volume, durée, vitesse)", "numeriques": rythme},
        {
            "code": "E4",
            "description": "+ répartition des 35 actions",
            "numeriques": repartition,
        },
        {
            "code": "E5",
            "description": "+ contenu et écran / chaîne principaux",
            "numeriques": contenu,
            "categorielles": COLONNES_CATEGORIELLES,
        },
        {
            "code": "E6",
            "description": "Navigateur + TF-IDF des écrans seuls",
            "familles_jetons": ("ecran",),
        },
        {
            "code": "E7",
            "description": "Navigateur + TF-IDF complet",
            "familles_jetons": FAMILLES_JETONS_COMPLETES,
        },
        {
            "code": "E8",
            "description": "Features simples (E5) + TF-IDF complet",
            "numeriques": contenu,
            "categorielles": COLONNES_CATEGORIELLES,
            "familles_jetons": FAMILLES_JETONS_COMPLETES,
        },
        {
            "code": "E9",
            "description": "E7 + TF-IDF des séquences d'actions complètes",
            "familles_jetons": FAMILLES_JETONS_COMPLETES,
            "actions_completes": True,
        },
    ]


def evaluer_essai(
    essai: dict, features: pd.DataFrame, y: pd.Series, validation: StratifiedKFold
) -> dict:
    """Évalue un essai par validation croisée avec une forêt aléatoire fixe.

    Le pré-traitement (normalisation, one-hot, TF-IDF) est dans le pipeline :
    il est réappris sur les plis d'entraînement de chaque découpage, jamais
    sur le pli de validation.

    Paramètres
    ----------
    essai : dict
        Élément de definir_essais.
    features : pd.DataFrame
        Tableau de features du train.
    y : pd.Series
        Utilisateur de chaque session.
    validation : StratifiedKFold
        Découpage utilisé pour tous les essais (comparaison équitable).

    Renvoie
    -------
    dict
        Code, description, nombre de features, F1 macro et accuracy
        (moyenne et écart-type sur les plis) et durée de l'essai.
    """
    arguments = {
        cle: essai[cle]
        for cle in ("numeriques", "categorielles", "familles_jetons", "actions_completes")
        if cle in essai
    }
    modele = make_pipeline(
        construire_preprocesseur(**arguments),
        RandomForestClassifier(n_estimators=NB_ARBRES, n_jobs=-1, random_state=RANDOM_STATE),
    )
    debut = time.perf_counter()
    scores = cross_validate(modele, features, y, cv=validation, scoring=["f1_macro", "accuracy"])
    nb_features = construire_preprocesseur(**arguments).fit_transform(features).shape[1]
    return {
        "code": essai["code"],
        "description": essai["description"],
        "nb_features": nb_features,
        "f1_macro": scores["test_f1_macro"].mean(),
        "f1_macro_ecart_type": scores["test_f1_macro"].std(),
        "accuracy": scores["test_accuracy"].mean(),
        "accuracy_ecart_type": scores["test_accuracy"].std(),
        "duree_s": time.perf_counter() - debut,
    }


def classer_features_simples(features: pd.DataFrame, colonnes: list[str]) -> pd.Series:
    """Classe les features numériques selon le test F de l'ANOVA à un facteur.

    Le score F compare la variance d'une feature entre utilisateurs à sa
    variance au sein d'un même utilisateur : plus il est grand, plus la
    feature sépare les utilisateurs. C'est un critère univarié (chaque feature
    est jugée seule), qui sert ici à l'inspection et pas à la sélection.

    Renvoie
    -------
    pd.Series
        Score F de chaque feature, trié décroissant (features constantes exclues).
    """
    scores_f, _ = f_classif(features[colonnes], features["util"])
    return pd.Series(scores_f, index=colonnes).dropna().sort_values(ascending=False)


def tracer_classement(classement: pd.Series) -> None:
    """Barres horizontales des features simples les plus discriminantes."""
    meilleures = classement.head(NB_FEATURES_CLASSEMENT)
    libelles = [nom.replace(PREFIXE_PART_ACTION, "part : ") for nom in meilleures.index]
    figure, axe = plt.subplots(figsize=(10, 6))
    axe.barh(libelles, meilleures.values, height=0.7, color=COULEUR_PRINCIPALE)
    axe.invert_yaxis()
    axe.set_xlabel("Score F de l'ANOVA (variance entre utilisateurs / variance intra-utilisateur)")
    axe.set_title(f"Les {NB_FEATURES_CLASSEMENT} features simples les plus discriminantes")
    axe.grid(axis="y", visible=False)
    axe.tick_params(axis="y", labelsize=7)
    sauvegarder_figure(figure, "fig08_classement_features_simples")


def tracer_carnet(carnet: pd.DataFrame) -> None:
    """Barres du F1 macro et de l'accuracy de chaque essai (moyenne ± écart-type)."""
    tracer_scores(
        carnet,
        carnet["code"] + " — " + carnet["description"],
        titre="Carnet d'essais : ajout progressif des features (forêt aléatoire)",
        legende_x=f"Score en validation croisée stratifiée ({N_SPLITS} plis)",
        nom="fig09_carnet_essais_features",
    )


def main() -> None:
    """Calcule et sauvegarde les features, puis évalue les essais du carnet."""
    configurer_style()
    DOSSIER_TABLEAUX.mkdir(parents=True, exist_ok=True)

    sessions_train = pd.read_pickle(SESSIONS_PRETRAITEES_TRAIN)
    sessions_test = pd.read_pickle(SESSIONS_PRETRAITEES_TEST)
    features_train = calculer_features_simples(sessions_train)
    colonnes_parts = colonnes_parts_actions(features_train)
    # Le test reprend exactement les colonnes du train (libellés absents -> 0)
    features_test = calculer_features_simples(sessions_test, colonnes_parts=colonnes_parts)
    features_train.to_pickle(FEATURES_TRAIN)
    features_test.to_pickle(FEATURES_TEST)
    print(
        f"Features simples : {features_train.shape[1] - 3} colonnes "
        f"(train {features_train.shape}, test {features_test.shape})"
    )

    classement = classer_features_simples(features_train, colonnes_numeriques(features_train))
    classement.rename("score_f").rename_axis("feature").reset_index().to_csv(
        DOSSIER_TABLEAUX / "classement_features_anova.csv", encoding=ENCODAGE, index=False
    )
    tracer_classement(classement)
    print("Features simples les plus discriminantes (score F) :")
    print(classement.head(10).round(1).to_string())
    print(f"Rang de « {COLONNE_COMPTAGE_SIMPLE} » : "
          f"{classement.index.get_loc(COLONNE_COMPTAGE_SIMPLE) + 1} / {len(classement)}")

    validation = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    resultats = []
    for essai in definir_essais(colonnes_parts):
        resultat = evaluer_essai(essai, features_train, features_train["util"], validation)
        resultats.append(resultat)
        print(
            f"{resultat['code']} {resultat['description']:<45} "
            f"{resultat['nb_features']:>5} features  "
            f"F1 macro {resultat['f1_macro']:.3f} ± {resultat['f1_macro_ecart_type']:.3f}  "
            f"accuracy {resultat['accuracy']:.3f}  ({resultat['duree_s']:.0f} s)"
        )
    carnet = pd.DataFrame(resultats)
    carnet.to_csv(DOSSIER_TABLEAUX / "carnet_essais_features.csv", encoding=ENCODAGE, index=False)
    tracer_carnet(carnet)


if __name__ == "__main__":
    main()
