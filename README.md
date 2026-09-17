# ROTATION — rotation sectorielle, suivi hebdomadaire

Repère le secteur qui **monte** dans le classement (la rotation qui commence), et le leader qui s'essouffle. Pas le premier : le premier a déjà tourné.

Tourne tout seul sur GitHub Actions chaque samedi à 9 h (Paris). Une **issue** est ouverte (→ e-mail) uniquement au début d'un nouveau signal.

| Marché | Univers | Rapport |
|---|---|---|
| US | Russell 1000, secteurs GICS | [`US/rotation.md`](US/rotation.md) |
| FR | à venir | |

## Structure

```
rotation.py                   le calcul, commun à tous les marchés
<MARCHÉ>/universe.csv         ticker (Yahoo), name, sector
<MARCHÉ>/rotation.md          dernier rapport (généré)
<MARCHÉ>/sector_history.csv   historique hebdo par secteur (généré)
.github/workflows/rotation.yml
```

Ajouter un marché : un dossier avec son `universe.csv` + une entrée dans `MARKETS` (`rotation.py`). Le workflow le prend automatiquement.

En local : `pip install yfinance pandas numpy` puis `python rotation.py US`.

## Mise à jour de l'univers US

`US/universe.csv` est un instantané de `TRADING/US/UNIVERSE/russell1000_full_enriched.csv`, réduit à 3 colonnes. Les secteurs bougent peu : le rafraîchir après la reconstitution annuelle du Russell (fin juin).

## Limites

- Seuils non calibrés : aucun backtest.
- Univers d'aujourd'hui appliqué au passé : l'historique est flatteur.
- Un secteur en rotation indique où chercher, pas quoi acheter.

*Travail d'analyse quantitative, sans valeur de conseil en investissement.*
