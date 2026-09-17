"""
rotation.py - Suivi hebdomadaire de la rotation sectorielle, multi-marchés
=========================================================================
Usage : python rotation.py US        (plus tard : python rotation.py FR)

Question : quel secteur est en train de MONTER dans le classement, et quel leader
s'essouffle ? Pas « qui est premier » : le premier a déjà tourné.

Par marché, dans le dossier <MARCHÉ>/ :
  Input  : universe.csv         colonnes ticker (Yahoo), name, sector
  Output : rotation.md          rapport lisible (s'affiche tel quel sur GitHub)
           sector_history.csv   historique hebdo par secteur (reconstruit à chaque run)
           alert.txt            créé UNIQUEMENT si un nouveau signal apparaît cette semaine
                                → le workflow GitHub ouvre alors une issue (= e-mail)

Méthode (tout à parts égales : une rotation naissante se voit dans les titres
moyens avant les poids lourds de l'indice) :
  RS        indice sectoriel équipondéré / univers équipondéré
  Rang      classement des secteurs sur la variation du RS en 13 semaines
  Largeur   % des titres du secteur au-dessus de leur MM50

Statuts :
  ROTATION ENTRANTE   3 semaines sur 4 : rang gagné >= 3 places en 8 sem., venait de
                      la moitié basse, RS au-dessus de sa moyenne 26 sem. et en hausse
                      sur 13 sem., largeur +15 pts en 8 sem. et au-dessus du marché
  À SURVEILLER        mêmes conditions cette semaine, pas encore confirmées
  LEADER ESSOUFFLÉ    3 semaines sur 4 : top 3 il y a 8 sem., a perdu >= 3 places,
                      largeur -15 pts en 8 sem.
  LEADER              top 3 actuel

/!\ Seuils = choix de conception, NON calibrés (pas de backtest). Univers non
point-in-time (survivants seulement). Travail d'analyse, pas un conseil.
"""

import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ======================
# CONFIG
# ======================
ROOT = os.path.dirname(os.path.abspath(__file__))

# Un marché = un dossier + un libellé + la traduction des noms de secteurs.
# Ajouter la France = créer FR/universe.csv et une entrée ici.
MARKETS = {
    "US": {
        "title": "Russell 1000 (GICS)",
        "sector_fr": {
            "Information Technology": "Technologie",
            "Financials": "Finance",
            "Industrials": "Industrie",
            "Consumer Discretionary": "Conso. discrétionnaire",
            "Health Care": "Santé",
            "Real Estate": "Immobilier",
            "Consumer Staples": "Conso. de base",
            "Materials": "Matériaux",
            "Communication Services": "Communication",
            "Utilities": "Services publics",
            "Energy": "Énergie",
        },
    },
}
MARKET = None   # renseigné par main()
SECTOR_FR = {}

PERIOD          = "2y"     # 2 ans de cours → ~1 an d'historique de signaux exploitable
MOM_WEEKS       = 13       # momentum du RS (1 trimestre)
RS_MA_WEEKS     = 26       # moyenne longue du RS
LOOKBACK_WEEKS  = 8        # fenêtre de comparaison du rang et de la largeur
MIN_RANK_GAIN   = 3        # places gagnées (ou perdues pour un leader)
BREADTH_DELTA   = 15.0     # points de % de titres > MM50
PERSIST         = (3, 4)   # 3 semaines sur 4
MAX_WEEKLY_MOVE = 0.5      # rendement hebdo écrêté à ±50 % (données aberrantes)
MAX_STALE_DAYS  = 5        # garde : refus d'imprimer si les cours sont périmés
MIN_COVERAGE    = 0.80     # garde : au moins 80 % des tickers téléchargés

STATUS_IN    = "ROTATION ENTRANTE"
STATUS_WATCH = "À SURVEILLER"
STATUS_OUT   = "LEADER ESSOUFFLÉ"
STATUS_LEAD  = "LEADER"



# ======================
# DONNÉES
# ======================
def load_universe(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df = df.dropna(subset=["sector"])
    df = df[df["ticker"].ne("") & df["ticker"].ne("nan")]
    return df.drop_duplicates("ticker")[["ticker", "sector"]]


def download_closes(tickers, chunk=150, passes=3):
    """Téléchargement par paquets avec reprises : Yahoo limite fortement le débit
    depuis les serveurs GitHub, un seul appel de 1 000 tickers échoue souvent."""
    import time
    import yfinance as yf

    def _dl(tks):
        try:
            data = yf.download(tks, period=PERIOD, interval="1d", auto_adjust=True,
                               progress=False, threads=False, group_by="column")
        except Exception as e:  # rate limit, réseau...
            print(f"    paquet en erreur : {type(e).__name__}: {e}")
            return pd.DataFrame()
        if data is None or data.empty:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            return data["Close"]
        return data[["Close"]].set_axis(tks, axis=1)

    frames, todo = [], list(tickers)
    for p in range(1, passes + 1):
        if not todo:
            break
        print(f"  Passe {p} : {len(todo)} tickers")
        for i in range(0, len(todo), chunk):
            part = _dl(todo[i:i + chunk])
            if not part.empty:
                frames.append(part.dropna(axis=1, how="all"))
            time.sleep(2 * p)
        got = set().union(*(f.columns for f in frames)) if frames else set()
        todo = [t for t in tickers if t not in got]

    if not frames:
        return pd.DataFrame()
    close = pd.concat(frames, axis=1)
    close = close.loc[:, ~close.columns.duplicated()]
    close.index = pd.to_datetime(close.index)
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    return close.sort_index()


# ======================
# CALCUL
# ======================
def compute_history(close, sectors):
    """close : cours quotidiens (dates x tickers). sectors : Series ticker → secteur.
    Retourne un DataFrame long, une ligne par (semaine, secteur)."""
    sectors = sectors[sectors.index.isin(close.columns)]
    close = close[sectors.index]

    # --- Hebdo : clôture du dernier jour de cotation de la semaine ---
    last = close.index.max()

    def _weekly(df, how):
        out = getattr(df.resample("W-FRI"), how)()
        out.index = out.index.where(out.index <= last, last)  # semaine en cours : date réelle
        return out

    wk = _weekly(close, "last")
    wret = wk.pct_change(fill_method=None).clip(-MAX_WEEKLY_MOVE, MAX_WEEKLY_MOVE)

    # --- Largeur : % > MM50 (quotidien, relevé en fin de semaine) ---
    above50 = (close > close.rolling(50, min_periods=50).mean()).where(close.notna())
    above50 = above50.where(close.rolling(50, min_periods=50).count() >= 50)
    wk_above = _weekly(above50, "last")

    # --- Plus hauts 52 sem. atteints dans la semaine (affiché, non filtrant) ---
    hi252 = close.rolling(252, min_periods=200).max()
    at_high = (close >= hi252).where(hi252.notna())
    wk_high = _weekly(at_high, "max")

    univ_ret = wret.mean(axis=1)
    univ_idx = (1 + univ_ret.fillna(0)).cumprod()
    univ_breadth = wk_above.mean(axis=1) * 100

    frames = []
    for sec, tks in sectors.groupby(sectors).groups.items():
        tks = list(tks)
        idx = (1 + wret[tks].mean(axis=1).fillna(0)).cumprod()
        rs = idx / univ_idx
        frames.append(pd.DataFrame({
            "week": wk.index,
            "sector": sec,
            "n": wk[tks].notna().sum(axis=1).values,
            "rs": rs.values,
            "rs_mom13": (rs / rs.shift(MOM_WEEKS) - 1).values * 100,
            "rs_above_ma26": (rs > rs.rolling(RS_MA_WEEKS).mean()).values,
            "breadth": (wk_above[tks].mean(axis=1) * 100).values,
            "pct_52w_high": (wk_high[tks].mean(axis=1) * 100).values,
            "univ_breadth": univ_breadth.values,
        }))
    h = pd.concat(frames, ignore_index=True)

    # --- Rang hebdo (1 = meilleur momentum de RS) ---
    h["rank"] = h.groupby("week")["rs_mom13"].rank(ascending=False, method="min")
    h = h.sort_values(["sector", "week"]).reset_index(drop=True)
    g = h.groupby("sector")
    h["rank_8w_ago"] = g["rank"].shift(LOOKBACK_WEEKS)
    h["rank_gain_8w"] = h["rank_8w_ago"] - h["rank"]
    h["breadth_d8w"] = h["breadth"] - g["breadth"].shift(LOOKBACK_WEEKS)

    valid = h["rank_8w_ago"].notna() & h["breadth_d8w"].notna() & h["rs_mom13"].notna()
    h["cond_in"] = valid & (
        (h["rank_gain_8w"] >= MIN_RANK_GAIN)
        & (h["rank_8w_ago"] >= 6)
        & (h["rs_mom13"] > 0)
        & h["rs_above_ma26"]
        & (h["breadth_d8w"] >= BREADTH_DELTA)
        & (h["breadth"] > h["univ_breadth"])
    )
    h["cond_out"] = valid & (
        (h["rank_8w_ago"] <= 3)
        & (h["rank_gain_8w"] <= -MIN_RANK_GAIN)
        & (h["breadth_d8w"] <= -BREADTH_DELTA)
    )
    k, n = PERSIST
    g = h.groupby("sector")
    h["in_confirmed"] = g["cond_in"].transform(lambda s: s.astype(int).rolling(n).sum() >= k)
    h["out_confirmed"] = g["cond_out"].transform(lambda s: s.astype(int).rolling(n).sum() >= k)

    def _status(r):
        if r.in_confirmed:
            return STATUS_IN
        if r.out_confirmed:
            return STATUS_OUT
        if r.cond_in:
            return STATUS_WATCH
        if r["rank"] <= 3:
            return STATUS_LEAD
        return ""
    h["status"] = h.apply(_status, axis=1)

    # Début d'épisode : statut fort nouveau, et absent des 8 semaines précédentes
    # (évite de ré-alerter quand un signal clignote d'une semaine à l'autre)
    strong = h["status"].where(h["status"].isin([STATUS_IN, STATUS_OUT]), "")
    h["new_signal"] = False
    for sec, idx in h.groupby("sector").groups.items():
        st = strong.loc[idx].tolist()
        for i, v in enumerate(st):
            if v and v not in st[max(0, i - LOOKBACK_WEEKS):i]:
                h.loc[idx[i], "new_signal"] = True
    h = h[h["rs_mom13"].notna()].copy()
    return h.sort_values(["week", "rank"]).reset_index(drop=True)


# ======================
# RAPPORT
# ======================
def _fmt(x, nd=0, sign=False):
    if pd.isna(x):
        return "—"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace(".", ",").replace("-", "−")


def build_report(h, last_date):
    weeks = sorted(h["week"].unique())
    cur_w = weeks[-1]
    cur = h[h["week"] == cur_w].sort_values("rank")

    # Nouveaux signaux = statut fort qui n'existait pas la semaine précédente
    new = [(r.status, r.sector) for _, r in cur[cur["new_signal"]].iterrows()]

    L = [f"# Rotation sectorielle {MARKET} — {pd.Timestamp(cur_w):%d/%m/%Y}", ""]
    L += [f"Classement des {cur.shape[0]} secteurs — {MARKETS[MARKET]['title']} — **à parts égales**, sur la "
          "variation de leur force relative en 13 semaines. **Ce qui compte : qui monte**, "
          "pas qui est premier.", ""]

    L += ["## Cette semaine", ""]
    if new:
        for st, sec in new:
            L.append(f"- **{st}** : {SECTOR_FR.get(sec, sec)}")
    else:
        L.append("- Aucun nouveau signal. C'est le cas normal la plupart des semaines.")
    watch = cur[cur["status"] == STATUS_WATCH]
    if len(watch):
        L.append("- À surveiller (conditions réunies, pas encore 3 semaines sur 4) : "
                 + ", ".join(SECTOR_FR.get(s, s) for s in watch["sector"]))
    L += ["", f"Largeur du marché : **{_fmt(cur['univ_breadth'].iloc[0])} %** des titres au-dessus de leur MM50.", ""]

    L += ["## Classement", "",
          "| Rang | Il y a 8 sem. | Secteur | RS 13 sem. | % > MM50 | Δ 8 sem. | % plus haut 52 s. | Statut |",
          "|---:|---:|---|---:|---:|---:|---:|---|"]
    for _, r in cur.iterrows():
        gain = r.rank_gain_8w
        arrow = "" if pd.isna(gain) or gain == 0 else (" ▲" if gain > 0 else " ▼")
        L.append(
            f"| {int(r['rank'])} | {_fmt(r.rank_8w_ago)}{arrow} | {SECTOR_FR.get(r.sector, r.sector)} "
            f"| {_fmt(r.rs_mom13, 1, True)} % | {_fmt(r.breadth)} % | {_fmt(r.breadth_d8w, 0, True)} pts "
            f"| {_fmt(r.pct_52w_high)} % | {('**' + r.status + '**') if r.status in (STATUS_IN, STATUS_OUT) else r.status} |"
        )

    # Journal des signaux confirmés sur l'historique disponible (début de chaque épisode)
    L += ["", "## Signaux des 12 derniers mois", "",
          "_Première semaine de chaque épisode confirmé. Sert à juger si le signal a eu du sens a posteriori._", ""]
    ev = h[h["new_signal"]]
    ev = ev[ev["week"] >= pd.Timestamp(cur_w) - pd.Timedelta(weeks=52)].sort_values("week", ascending=False)
    if len(ev):
        L += ["| Semaine | Secteur | Signal | Rang alors | Rang aujourd'hui |", "|---|---|---|---:|---:|"]
        now_rank = cur.set_index("sector")["rank"]
        for _, r in ev.iterrows():
            L.append(f"| {pd.Timestamp(r.week):%d/%m/%Y} | {SECTOR_FR.get(r.sector, r.sector)} | {r.status} "
                     f"| {int(r['rank'])} | {_fmt(now_rank.get(r.sector))} |")
    else:
        L.append("Aucun épisode confirmé.")

    L += ["", "## À lire avant d'agir", "",
          "- **Seuils non calibrés.** Aucun backtest : ce sont des choix de conception.",
          "- **Univers d'aujourd'hui** appliqué au passé (survivants seulement) : l'historique est flatteur.",
          "- **Un secteur en rotation n'est pas un achat.** C'est un endroit où chercher : croiser avec "
          "les screeners et setups du marché.",
          "", "---",
          f"*Dernière séance : {last_date:%d/%m/%Y}. Généré le {datetime.now(timezone.utc):%d/%m/%Y %H:%M} UTC. "
          "Travail d'analyse quantitative, sans valeur de conseil en investissement.*", ""]
    return "\n".join(L), new


# ======================
# MAIN
# ======================
def main():
    global MARKET, SECTOR_FR
    if len(sys.argv) != 2 or sys.argv[1].upper() not in MARKETS:
        sys.exit(f"Usage : python rotation.py <{'|'.join(MARKETS)}>")
    MARKET = sys.argv[1].upper()
    SECTOR_FR = MARKETS[MARKET]["sector_fr"]
    mdir = os.path.join(ROOT, MARKET)
    report_md, history_csv, alert_txt = (os.path.join(mdir, f) for f in
                                         ("rotation.md", "sector_history.csv", "alert.txt"))

    print(f"ROTATION {MARKET} — suivi hebdomadaire de la rotation sectorielle")
    uni = load_universe(os.path.join(mdir, "universe.csv"))
    tickers = uni["ticker"].tolist()
    print(f"Univers : {len(tickers)} tickers")

    close = download_closes(tickers)
    if close.empty:
        sys.exit("ÉCHEC : aucun cours téléchargé (Yahoo inaccessible ou limite de débit).")
    coverage = close.shape[1] / len(tickers)
    last_date = close.index.max()
    stale = (pd.Timestamp.now().normalize() - last_date.normalize()).days
    print(f"Téléchargés : {close.shape[1]} ({coverage:.0%}) — dernière séance {last_date:%d/%m/%Y}")

    if coverage < MIN_COVERAGE:
        sys.exit(f"ÉCHEC : couverture {coverage:.0%} < {MIN_COVERAGE:.0%}, rapport non généré.")
    if stale > MAX_STALE_DAYS:
        sys.exit(f"ÉCHEC : cours vieux de {stale} jours, rapport non généré.")

    h = compute_history(close, uni.set_index("ticker")["sector"])
    report, new = build_report(h, last_date)

    cols = ["week", "sector", "n", "rank", "rank_8w_ago", "rank_gain_8w", "rs", "rs_mom13",
            "breadth", "breadth_d8w", "pct_52w_high", "univ_breadth", "status"]
    out = h[cols].copy()
    for c in cols[2:-1]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(2)
    out.to_csv(history_csv, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write(report)

    if os.path.exists(alert_txt):
        os.remove(alert_txt)
    if new:
        title = f"Rotation {MARKET} : " + " · ".join(f"{SECTOR_FR.get(s, s)} {st.lower()}" for st, s in new)
        with open(alert_txt, "w", encoding="utf-8") as f:
            f.write(title + "\n")
        print(f"ALERTE : {title}")
    else:
        print("Aucun nouveau signal.")
    print(f"[EXPORT] {report_md}\n[EXPORT] {history_csv}")


if __name__ == "__main__":
    main()
