# Description du dataset

## Fichiers

| Fichier             | Contenu                                                                 |
| ------------------- | ----------------------------------------------------------------------- |
| `TRAIN.csv`         | Jeu d'apprentissage : traces d'utilisation du logiciel **étiquetées**   |
| `TEST.csv`          | Jeu de test : traces **non étiquetées** (il faut prédire l'utilisateur) |
| `SAMPLE_SUBMISSION` | Format du fichier à soumettre, avec les prédictions sur le jeu de test  |

---

## Données brutes (`TRAIN.csv` et `TEST.csv`)

Les fichiers de traces ont été extraits et parsés pour produire un fichier texte interprétable : **chaque ligne correspond aux actions effectuées par un utilisateur au cours d'une seule session**.

Exemple de ligne :

```text
sph,Firefox,Exécution d'un bouton(fr.infologic.core.client.modules.web.CopiloteWebSharedController)<DEFAUT>$AC$,t5,Exécution d'un bouton,t10
```

Les valeurs sont séparées par des virgules.

### Champs

| Position     | Nom          | Description                                                                                               |
| ------------ | ------------ | --------------------------------------------------------------------------------------------------------- |
| 1            | `util`       | ID unique de l'utilisateur (trigramme) — **présent uniquement dans `TRAIN.csv`** (c'est la cible)          |
| 2            | `navigateur` | Navigateur de la session : Firefox, Google Chrome, Opera ou Microsoft Edge                               |
| 3 et suivants | actions     | Actions effectuées par l'utilisateur, entrecoupées de marqueurs temporels `tXX`                          |

### Syntaxe des actions

Une action est un libellé, par exemple `Création d'un écran`, `Double-clic`, `Chainage` ou `Lancement d'une action générique`. Des informations peuvent s'y ajouter :

| Élément                     | Notation        | Exemple                                                                    |
| --------------------------- | --------------- | -------------------------------------------------------------------------- |
| Écran (navigation)          | `(...)`         | `Création d'un écran(infologic.crm.modules.CRM_ANNUAIRE.AnnuaireController)` |
| Configuration d'écran       | `<...>`         | `Création d'un écran(infologic.core.gui.controllers.BlankController)<ACCUEIL_INST>` |
| Chaîne de la fiche*         | `$...$`         | `Saisie dans un champ(MAINT)<DEF_03/24>$GP$`                               |
| Mode modification           | `1` à la fin    | `Saisie dans un champ1`                                                    |

\* La « chaîne » d'une fiche est son regroupement logique au sein du logiciel.

### Fenêtres temporelles

Les traces contiennent des champs **`tXX`**, où `XX` est un entier donnant une fenêtre temporelle en secondes :

- toutes les actions **avant `t20`** ont été effectuées dans les 20 premières secondes ;
- celles **entre `t20` et `t30`** ont été effectuées dans l'intervalle [20 ; 30] s ;
- etc.

> Le nombre de fenêtres temporelles et d'actions **varie d'une ligne à l'autre** : les utilisateurs ne vont pas tous à la même vitesse et les sessions n'ont pas toutes la même durée.

---

## ⚠️ Remarque importante

Après avoir lu cette description, vous devez pouvoir répondre aux questions suivantes :

- De quels fichiers ai-je besoin ?
- Quel format de données dois-je m'attendre à trouver ?
- Que dois-je prédire ?
- Quels acronymes vais-je rencontrer ?

Si ce n'est pas le cas, demandez à votre professeur !
