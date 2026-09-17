# ROTATION — rotation sectorielle FR et US

Repère le secteur qui **monte** dans le classement (la rotation qui commence) et le leader qui s'essouffle. Pas le premier : le premier a déjà tourné.

Tourne tout seul sur GitHub Actions du lundi au vendredi à 22 h (Paris). Un **message Telegram** part uniquement le soir où un signal apparaît (secrets `TELEGRAM_TOKEN` et `TELEGRAM_CHAT_ID`, les mêmes que tes scanners FR).

| | Script | Univers | Rapport |
|---|---|---|---|
| FR | `rotation_fr.py` | `universe_fr.csv` — Euronext Paris + Growth, industries ICB | [`rotation_fr.md`](rotation_fr.md) |
| US | `rotation_us.py` | `universe_us.csv` — Russell 1000, secteurs GICS | [`rotation_us.md`](rotation_us.md) |

`history_fr.csv` / `history_us.csv` : historique hebdomadaire par secteur (généré).

En local : `pip install yfinance pandas numpy` puis `python rotation_fr.py` ou `python rotation_us.py`.

## Univers

Instantanés réduits à 3 colonnes (`ticker, name, sector`) de :
- FR : `TRADING/FR/euronext_full_enriched.csv`
- US : `TRADING/US/UNIVERSE/russell1000_full_enriched.csv`

Les secteurs bougent peu : un rafraîchissement par an suffit.

## Limites

- Seuils non calibrés : aucun backtest.
- Univers d'aujourd'hui appliqué au passé : l'historique est flatteur.
- FR : marché étroit, les secteurs Énergie et Télécoms comptent peu de titres liquides — signal plus bruité.
- Un secteur en rotation indique où chercher, pas quoi acheter.

*Travail d'analyse quantitative, sans valeur de conseil en investissement.*
