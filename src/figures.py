"""Style graphique et tracés communs à toutes les figures du projet."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DOSSIER_FIGURES

# Palette : deux couleurs de séries (bleu, orange), encres de texte et grille
COULEUR_PRINCIPALE = "#2a78d6"
COULEUR_SECONDAIRE = "#eb6834"
COULEUR_BOITE = "#cde2fb"
ENCRE = "#0b0b0b"
ENCRE_SECONDAIRE = "#52514e"
COULEUR_GRILLE = "#e6e5e1"
# Couleurs des séries quand on compare plusieurs modèles, toujours dans cet ordre
# (ordre validé pour rester distinguable par les daltoniens entre voisins).
PALETTE_CATEGORIELLE = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]
RAMPE_BLEUE = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DPI = 150


def configurer_style() -> None:
    """Applique un style sobre et commun à toutes les figures."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": ENCRE_SECONDAIRE,
            "axes.labelcolor": ENCRE,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": COULEUR_GRILLE,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "xtick.color": ENCRE_SECONDAIRE,
            "ytick.color": ENCRE_SECONDAIRE,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "font.size": 9,
        }
    )


def sauvegarder_figure(figure: plt.Figure, nom: str) -> None:
    """Enregistre une figure en PNG dans outputs/figures/ puis la ferme."""
    DOSSIER_FIGURES.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(DOSSIER_FIGURES / f"{nom}.png", dpi=DPI, bbox_inches="tight")
    plt.close(figure)


def tracer_scores(
    scores: pd.DataFrame, etiquettes: pd.Series, titre: str, legende_x: str, nom: str
) -> None:
    """Barres horizontales du F1 macro et de l'accuracy (moyenne ± écart-type).

    Paramètres
    ----------
    scores : pd.DataFrame
        Colonnes f1_macro, f1_macro_ecart_type, accuracy, accuracy_ecart_type.
    etiquettes : pd.Series
        Libellé de chaque ligne, dans le même ordre que scores.
    titre, legende_x : str
        Titre de la figure et légende de l'axe horizontal.
    nom : str
        Nom du fichier PNG (sans extension).
    """
    positions = np.arange(len(scores))
    hauteur = 0.38
    figure, axe = plt.subplots(figsize=(9, 0.6 * len(scores) + 1))
    for decalage, colonne, couleur, libelle in [
        (-hauteur / 2, "f1_macro", COULEUR_PRINCIPALE, "F1 macro"),
        (hauteur / 2, "accuracy", COULEUR_SECONDAIRE, "Accuracy"),
    ]:
        barres = axe.barh(
            positions + decalage,
            scores[colonne],
            height=hauteur - 0.04,
            xerr=scores[f"{colonne}_ecart_type"],
            color=couleur,
            error_kw={"ecolor": ENCRE_SECONDAIRE, "elinewidth": 1, "capsize": 2},
            label=libelle,
        )
        axe.bar_label(barres, fmt="%.3f", padding=8, fontsize=7, color=ENCRE_SECONDAIRE)
    axe.set_yticks(positions, etiquettes)
    axe.invert_yaxis()
    axe.set_xlim(0, 1.05)
    axe.set_xlabel(legende_x)
    # Légende au-dessus du graphique : elle ne peut alors masquer aucune barre
    axe.set_title(titre, pad=24)
    axe.legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=2)
    axe.grid(axis="y", visible=False)
    sauvegarder_figure(figure, nom)
