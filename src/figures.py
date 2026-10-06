"""Style graphique commun à toutes les figures du projet."""

import matplotlib.pyplot as plt

from config import DOSSIER_FIGURES

# Palette : deux couleurs de séries (bleu, orange), encres de texte et grille
COULEUR_PRINCIPALE = "#2a78d6"
COULEUR_SECONDAIRE = "#eb6834"
COULEUR_BOITE = "#cde2fb"
ENCRE = "#0b0b0b"
ENCRE_SECONDAIRE = "#52514e"
COULEUR_GRILLE = "#e6e5e1"
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
