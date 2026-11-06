"""Modèles de classification (module partagé par les étapes 5, 6 et 7).

Contient :
- le catalogue des modèles comparés à l'étape 5, avec leur grille d'hyperparamètres ;
- la construction du pipeline complet (features + modèle) ;
- ContrainteNavigateur, qui applique la règle métier « un utilisateur
  n'utilise qu'un navigateur » aux probabilités (ou scores) d'un modèle.

Voir explications/05_modele.md.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.metaestimators import available_if

from config import RANDOM_STATE
from features import COLONNES_CATEGORIELLES, FAMILLES_JETONS_COMPLETES, construire_preprocesseur

NB_ARBRES = 500
# Une forêt de 500 arbres occupe environ 1,1 Go en mémoire (247 valeurs par nœud) :
# au-delà de 2 entraînements en parallèle, une machine de 8 Go sature.
N_JOBS_FORETS = 2


def catalogue_modeles() -> dict[str, dict]:
    """Liste les modèles comparés et leurs hyperparamètres à explorer.

    Renvoie
    -------
    dict[str, dict]
        Pour chaque modèle : « estimateur » (non ajusté), « grille » (dict
        pour GridSearchCV, vide si aucun réglage), et si besoin « dense »
        (True si le modèle n'accepte pas les matrices creuses) et « n_jobs »
        (nombre d'entraînements en parallèle pendant la validation croisée,
        tous les cœurs par défaut) et « sequences » (True pour utiliser la
        représentation par séquences d'actions complètes au lieu de l'essai E8).
    """
    return {
        "Référence (classe majoritaire)": {
            "estimateur": DummyClassifier(strategy="most_frequent"),
            "grille": {},
        },
        "k plus proches voisins": {
            "estimateur": KNeighborsClassifier(metric="cosine", weights="distance"),
            "grille": {"modele__n_neighbors": [1, 3, 5, 10]},
        },
        "Régression logistique": {
            "estimateur": LogisticRegression(max_iter=3000),
            "grille": {"modele__C": [1, 10, 100], "modele__class_weight": [None, "balanced"]},
        },
        "SVM linéaire": {
            "estimateur": LinearSVC(max_iter=5000),
            "grille": {"modele__C": [0.03, 0.1, 0.3]},
        },
        # Même modèle que ci-dessus, mais sur la représentation « séquences
        # d'actions complètes » (voir construire_pipeline et explications/05_modele.md).
        "SVM linéaire sur séquences d'actions": {
            "estimateur": LinearSVC(class_weight="balanced", max_iter=5000),
            "grille": {"modele__C": [0.3, 1, 3]},
            "sequences": True,
        },
        "Forêt aléatoire": {
            "estimateur": RandomForestClassifier(
                n_estimators=NB_ARBRES, random_state=RANDOM_STATE
            ),
            "grille": {
                "modele__max_features": ["sqrt", 0.05],
                "modele__class_weight": [None, "balanced"],
            },
            "n_jobs": N_JOBS_FORETS,
        },
        "Extra-Trees": {
            "estimateur": ExtraTreesClassifier(
                n_estimators=NB_ARBRES, random_state=RANDOM_STATE
            ),
            "grille": {
                "modele__max_features": ["sqrt", 0.05],
                "modele__class_weight": [None, "balanced"],
            },
            "n_jobs": N_JOBS_FORETS,
        },
        "Perceptron multicouche": {
            "estimateur": MLPClassifier(max_iter=300, random_state=RANDOM_STATE),
            "grille": {
                "modele__hidden_layer_sizes": [(256,), (512,)],
                "modele__alpha": [1e-4, 1e-3],
            },
        },
        # Réglages par défaut : l'optimisation diverge (F1 macro ≈ 0,20 sur un pli).
        # Un pas plus petit, une pénalité L2 et des arbres moins profonds la stabilisent.
        "Gradient boosting": {
            "estimateur": HistGradientBoostingClassifier(
                learning_rate=0.05,
                l2_regularization=1.0,
                max_depth=4,
                min_samples_leaf=5,
                max_iter=100,
                early_stopping=False,
                random_state=RANDOM_STATE,
            ),
            "grille": {},
            "dense": True,
            # Le gradient boosting parallélise déjà en interne : un seul processus suffit
            "n_jobs": 1,
        },
    }


def construire_pipeline(
    estimateur, numeriques: list[str], dense: bool = False, sequences: bool = False
) -> Pipeline:
    """Assemble le pipeline complet : features puis modèle.

    Paramètres
    ----------
    estimateur : estimateur scikit-learn
        Modèle de classification (non ajusté).
    numeriques : list[str]
        Features simples numériques (features.colonnes_numeriques).
    dense : bool
        True pour convertir la matrice creuse en matrice dense (modèles qui
        n'acceptent pas le format creux, comme le gradient boosting).
    sequences : bool
        False : features de l'essai E8 (features simples + TF-IDF par famille).
        True : navigateur + TF-IDF par famille + TF-IDF des séquences d'actions
        complètes, sans les features simples (essai E9 de l'étape 4). Les
        features simples centrées-réduites, mêlées aux TF-IDF, dégradaient le
        SVM (F1 macro 0,93 au lieu de 0,95) : elles n'ont pas la même échelle.

    Renvoie
    -------
    Pipeline
        Étapes « preprocesseur » et « modele ».
    """
    if sequences:
        preprocesseur = construire_preprocesseur(
            familles_jetons=FAMILLES_JETONS_COMPLETES, actions_completes=True
        )
    else:
        preprocesseur = construire_preprocesseur(
            numeriques, COLONNES_CATEGORIELLES, FAMILLES_JETONS_COMPLETES
        )
    if dense:
        preprocesseur.set_params(sparse_threshold=0)
    return Pipeline([("preprocesseur", preprocesseur), ("modele", estimateur)])


def _encapsule_possede(nom: str):
    """Condition pour available_if : le modèle encapsulé possède la méthode « nom »."""
    return lambda contrainte: hasattr(contrainte.estimateur, nom)


class ContrainteNavigateur(ClassifierMixin, BaseEstimator):
    """Applique la règle « un utilisateur = un navigateur » à un classifieur.

    À l'entraînement, on retient les navigateurs utilisés par chaque
    utilisateur. À la prédiction, les utilisateurs qui n'ont jamais utilisé
    le navigateur de la session sont écartés :

    - modèle probabiliste (predict_proba) : leur probabilité est mise à zéro,
      puis on renormalise. Si le modèle n'accordait aucune probabilité aux
      utilisateurs compatibles, ceux-ci reçoivent une probabilité uniforme :
      une prédiction incompatible serait fausse à coup sûr ;
    - modèle à scores de décision (SVM, decision_function) : leur score est
      ramené sous le plus petit score de la session, ce qui les place en
      dernière position sans changer l'ordre des utilisateurs compatibles.

    Si le navigateur n'a jamais été vu, tous les utilisateurs restent possibles.

    Paramètres
    ----------
    estimateur : classifieur scikit-learn
        Modèle disposant de predict_proba ou de decision_function, appliqué à
        un DataFrame contenant une colonne « navigateur ».
    """

    def __init__(self, estimateur):
        self.estimateur = estimateur

    def fit(self, X, y):
        """Entraîne le modèle et mémorise les navigateurs de chaque utilisateur."""
        self.estimateur_ = clone(self.estimateur).fit(X, y)
        self.classes_ = self.estimateur_.classes_
        utilise = (
            X.assign(util=np.asarray(y)).groupby("navigateur")["util"].unique()
        )
        self.masques_ = {
            navigateur: np.isin(self.classes_, utilisateurs)
            for navigateur, utilisateurs in utilise.items()
        }
        return self

    def _masque(self, X) -> np.ndarray:
        """Matrice sessions x utilisateurs : True si l'utilisateur est compatible."""
        return np.array([
            self.masques_.get(navigateur, np.ones(len(self.classes_), dtype=bool))
            for navigateur in X["navigateur"]
        ])

    @available_if(_encapsule_possede("predict_proba"))
    def predict_proba(self, X):
        """Probabilités du modèle, restreintes aux utilisateurs du bon navigateur."""
        probas = self.estimateur_.predict_proba(X)
        masque = self._masque(X)
        restreintes = probas * masque
        sans_masse = restreintes.sum(axis=1) == 0
        restreintes[sans_masse] = masque[sans_masse]
        return restreintes / restreintes.sum(axis=1, keepdims=True)

    @available_if(_encapsule_possede("decision_function"))
    def decision_function(self, X):
        """Scores du modèle, ceux des utilisateurs incompatibles relégués en dernier."""
        scores = self.estimateur_.decision_function(X)
        plancher = scores.min(axis=1, keepdims=True) - 1
        return np.where(self._masque(X), scores, plancher)

    def predict(self, X):
        """Utilisateur le mieux noté parmi ceux compatibles avec le navigateur."""
        if hasattr(self.estimateur_, "predict_proba"):
            scores = self.predict_proba(X)
        else:
            scores = self.decision_function(X)
        return self.classes_[scores.argmax(axis=1)]
