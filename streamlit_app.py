import os
import sys
from datetime import datetime, date
import pandas as pd
import streamlit as st
from pathlib import Path

#---Helpers---
def _ensure_vendor_csv_local(drive, vendor_name: str) -> tuple[bool, str]:
    """
    Si nt_<vendor>.csv n'existe pas en local, tente de le télécharger depuis
    Drive: BASE_FOLDER_ID / 'Fichiers CSV' / 'nt_<vendor>.csv'.
    Retourne (ok, chemin_local).
    """
    vn = vendor_name.lower()
    local_path = os.path.join(CSV_DIR, f"nt_{vn}.csv")
    if os.path.exists(local_path):
        return True, local_path

    exports_id = drive.find_folder_id("Fichiers CSV")
    if not exports_id:
        return False, local_path

    q = f"'{exports_id}' in parents and name='nt_{vn}.csv' and trashed=false"
    res = drive.service.files().list(
        q=q,
        fields="files(id, name)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    files = res.get("files", [])
    if not files:
        return False, local_path

    fid = files[0]["id"]
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    drive.download_file(fid, local_path)
    return os.path.exists(local_path), local_path


def _upload_vendor_csv_to_drive(drive, vendor_name: str) -> bool:
    """Upload nt_<vendor>.csv local vers Drive/Fichiers CSV (overwrite)."""
    vn = vendor_name.lower()
    local_path = os.path.join(CSV_DIR, f"nt_{vn}.csv")
    if not os.path.exists(local_path):
        return False
    exports_id = drive.find_folder_id("Fichiers CSV")
    if not exports_id:
        return False
    drive.upload_file(local_path, f"nt_{vn}.csv", exports_id)
    return True

def drive_find_file(drive, parent_name: str, filename: str):
    """Retourne (file_id, modifiedTime) pour filename dans le dossier parent_name (à la racine BASE_FOLDER_ID)."""
    parent_id = drive.find_folder_id(parent_name)
    if not parent_id:
        return None, None
    res = drive.service.files().list(
        q=f"'{parent_id}' in parents and name='{filename}' and trashed=false",
        fields="files(id,name,modifiedTime)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    files = res.get("files", [])
    if not files:
        return None, None
    f = files[0]
    return f["id"], f.get("modifiedTime")

def sync_df_traitee_local_from_drive(drive) -> bool:
    """Télécharge df_traitee.csv depuis Drive si (absent) ou (Drive plus récent que local)."""
    fid, modified = drive_find_file(drive, "Fichiers CSV", "df_traitee.csv")
    if not fid:
        return False
    # si local existe et est à jour, on ne retélécharge pas
    if os.path.exists(CSV_TRAITE) and modified:
        import datetime, dateutil.parser  # si dateutil indispo, parsers via pandas.to_datetime
        remote_dt = pd.to_datetime(modified, utc=True)
        local_dt = pd.to_datetime(os.path.getmtime(CSV_TRAITE), unit="s", utc=True)
        if local_dt >= remote_dt:
            return True
    os.makedirs(os.path.dirname(CSV_TRAITE), exist_ok=True)
    drive.download_file(fid, CSV_TRAITE)
    return os.path.exists(CSV_TRAITE)

def charger_dfs_bruts_depuis_drive(drive, fournisseurs: list[str]) -> dict:
    """Télécharge (si besoin) et charge en DataFrame tous les nt_<f>.csv depuis Drive."""
    dfs = {}
    for f in fournisseurs:
        ok, local = _ensure_vendor_csv_local(drive, f)  # déjà dans ton code
        if ok:
            try:
                dfs[f] = pd.read_csv(local)
            except Exception:
                pass
    return dfs

def rebuild_df_traitee_from_drive_nt(drive) -> bool:
    """Reconstruit df_traitee.csv à partir des nt_*.csv (Drive-first), puis uploade sur Drive."""
    dfs = charger_dfs_bruts_depuis_drive(drive, FOURNISSEURS)
    if not dfs:
        return False
    traiteur = TraiterDFs(dfs, base_path=DATA_DIR)  # exporter_csv() écrira dans DATA_DIR/Fichiers CSV
    df_global = traiteur.traiter_df()
    # upload Drive
    parent_id = drive.find_folder_id("Fichiers CSV")
    if parent_id and os.path.exists(CSV_TRAITE):
        drive.upload_file(CSV_TRAITE, "df_traitee.csv", parent_id)
    return True


# --- Chemins & imports sûrs -------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Modules projet
from traiter_factures import TraiterDFs
from visualiser import Visualisation
from gestion_factures import GestionFactures

# OAuth Drive
from oauth_drive_manager import OAuthDriveManager, FileTokenStore

# --- Helpers ----------------------------------------------------------------
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, 'data'))
CSV_DIR = os.path.join(DATA_DIR, 'Fichiers CSV')
os.makedirs(CSV_DIR, exist_ok=True)

CSV_TRAITE = os.path.join(CSV_DIR, 'df_traitee.csv')

FOURNISSEURS = ["APDV", "CHERITEL", "LBB", "SUPERGROUP", "AUTENTIK"]

#@st.cache_resource(show_spinner=False)
import streamlit as st
from google.auth.exceptions import RefreshError

SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.file",
]

def _client_config_from_secrets():
    return {
        "web": {
            "client_id": st.secrets["OAUTH_CLIENT_ID"],
            "client_secret": st.secrets["OAUTH_CLIENT_SECRET"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }

def get_oauth_drive(base_folder_id: str | None, user_email: str):
    if not user_email:
        st.error("Renseigne ton email Google pour utiliser l'OAuth.")
        st.stop()

    redirect_uri = st.secrets["OAUTH_REDIRECT_URI"]

    store = FileTokenStore(root_dir=os.path.join(BASE_DIR, ".oauth_tokens"))
    mgr = OAuthDriveManager(
        client_config=_client_config_from_secrets(),
        token_store=store,
        user_key=user_email,
        base_folder_id=base_folder_id or "",
        scopes=SCOPES,
    )

    # 1) Tente refresh si possible
    try:
        mgr.ensure_valid()
    except RefreshError:
        # refresh_token mort => on force une nouvelle auth
        pass

    # 2) Si pas de service, on déclenche le flow web
    if mgr.service is None:
        st.warning("Connexion Google requise pour accéder à Drive.")

        # --- callback: si Google a renvoyé ?code=...
        qp = st.query_params
        if "code" in qp:
            code = qp["code"]
            mgr.fetch_token_from_code(code=code, redirect_uri=redirect_uri)

            # nettoyage URL (retire code/state)
            st.query_params.clear()
            st.rerun()

        # --- sinon : afficher le lien de connexion
        auth_url, state = mgr.get_auth_url(redirect_uri=redirect_uri)

        st.link_button("🔐 Se connecter à Google", auth_url)

        st.stop()

    return mgr


#@st.cache_resource(show_spinner=False)
def get_gestionnaire_oauth(base_folder_id: str | None, user_email: str):
    drive_mgr = get_oauth_drive(base_folder_id or "", user_email)
    # Injection du drive manager OAuth dans la classe projet
    return GestionFactures(base_folder_id=base_folder_id, drive_manager=drive_mgr)

@st.cache_data(show_spinner=False)
def lire_df_traitee():
    if os.path.exists(CSV_TRAITE):
        return pd.read_csv(CSV_TRAITE)
    return pd.DataFrame()

# --- UI ---------------------------------------------------------------------
st.set_page_config(page_title="Analyse Factures – OAuth", layout="wide")
st.title("📦 Analyse & gestion des factures")

with st.sidebar:
    st.header("⚙️ Paramètres")
    base_folder_id = st.text_input(
        "Google Drive BASE_FOLDER_ID",
        value=os.environ.get("BASE_FOLDER_ID", "11tAEJqUKrgC40l-pmJJGUsymjQ-T1TWY"),
        help=(
            "ID du dossier racine Drive (celui qui contient les sous-dossiers"
            "'Factures APDV', 'Factures LBB', etc.)."
        ),
    )

    user_email = st.text_input(
        "Email Google",
        value=os.environ.get("OAUTH_USER_EMAIL", ""),
        help="Adresse utilisée pour créer/charger les tokens dans .oauth_tokens/",
    )

    # Statut OAuth
    if user_email:
        st.caption(f"🔐 OAuth prêt pour : {user_email}")
    else:
        st.caption("🔐 Saisis ton email Google pour activer l'OAuth")

# Onglets
onglets = st.tabs(["1) Extraction", "2) Traitement", "3) Analyse", "4) Prédiction"])

# --- 1) Extraction -----------------------------------------------------------
with onglets[0]:
    st.subheader("Synchronisation & extraction des factures depuis Google Drive")
    colA, colB = st.columns([2,1])
    with colA:
        fournisseurs = st.multiselect("Fournisseurs à extraire", FOURNISSEURS, default=FOURNISSEURS)
    with colB:
        dl_from_drive = st.toggle(
            "Télécharger les PDF depuis Drive",
            value=True,
            help="Télécharge les nouveaux PDF depuis le sous-dossier Drive du fournisseur.",
        )
    if st.button("🔎 Diagnostiquer Drive (compter les factures PDF par fournisseur)"):
        gestion = get_gestionnaire_oauth(base_folder_id or None, user_email)
        for f in fournisseurs:
            F = f.upper()
            ids = []
            # mêmes candidats qu'en prod
            for path in [f"Factures {F}", ("temp_pdfs", F), ("temps_pdfs", F)]:
                if isinstance(path, str):
                    fid = gestion.drive.find_folder_id(path)
                    label = path
                else:
                    root = gestion.drive.find_folder_id(path[0])
                    fid = gestion.drive.find_folder_id(path[1], parent_id=root) if root else None
                    label = "/".join(path)
                if fid:
                    files = gestion.drive.list_pdfs_in_folder(fid) or []
                    st.write(f"• {label}: {len(files)} PDF")

    st.markdown("---")
    if st.button("🚀 Exporter les nouvelles données"):
        if not base_folder_id:
            st.error("Renseigne BASE_FOLDER_ID.")
        gestion = get_gestionnaire_oauth(base_folder_id or None, user_email)
        drive = gestion.drive

        total_new = 0
        dfs_result = {}

        for f in fournisseurs:
            with st.status(f"Préparation {f}…", expanded=False) as status:
                try:
                    # 1) Récupérer/assurer le CSV brut nt_<f>.csv depuis Drive
                    ok_csv, _ = _ensure_vendor_csv_local(drive, f)
                    if ok_csv:
                        status.update(label=f"{f} : CSV brut existant (Drive)", state="running")
                    else:
                        status.update(label=f"{f} : pas de CSV brut, première extraction", state="running")

                    # 2) Détecter & rapatrier les NOUVEAUX PDF depuis Drive
                    new_count = gestion.synchroniser_factures(f) or 0
                    total_new += int(new_count)

                    # 3) (Ré)extraire vers nt_<f>.csv — idéalement incrémental (tes extracteurs)
                    extracteur = gestion.extracteurs.get(f)
                    if extracteur is None:
                        raise RuntimeError(f"Aucun extracteur pour {f}")
                    df_f = extracteur()  # tes extracteurs écrivent nt_<f>.csv et renvoient un DF
                    dfs_result[f] = df_f

                    # 3bis) Re-pousser nt_<f>.csv sur Drive
                    _upload_vendor_csv_to_drive(drive, f)

                    status.update(label=f"{f} : OK – +{new_count} PDF, {len(df_f)} lignes", state="complete")
                except Exception as e:
                    status.update(label=f"{f} : Échec – {e}", state="error")

        # 4) (Re)construire df_traitee.csv SI (nouvelles factures) OU (df_traitee absent sur Drive)
        df_id, _ = drive_find_file(drive, "Fichiers CSV", "df_traitee.csv")
        need_global = (total_new > 0) or (df_id is None)

        if need_global:
            ok = rebuild_df_traitee_from_drive_nt(drive)
            if ok:
                st.success("Fichier CSV reconstruit et uploadé sur Drive ✅")
            else:
                st.warning("Impossible de reconstruire le fichier CSV (Aucun fichier CSV accessibles sur le Drive).")
        else:
            st.info("Aucune nouvelle facture détectée → Fichier CSV conservé tel quel.")

        # Et on garde un aperçu local (optionnel)
        if dfs_result:
            st.session_state["dfs_extraction"] = {k: v.head(50) for k, v in dfs_result.items()}
            for k, v in st.session_state["dfs_extraction"].items():
                with st.expander(f"Aperçu {k}"):
                    st.dataframe(v)

# --- 2) Traitement -----------------------------------------------------------
with onglets[1]:
    st.subheader("Nettoyage, harmonisation & export CSV global")
    st.caption("Permet de reconstruire le fichier CSV en agrégeant les 5 fournisseurs + étalement des ventes.\nA n'utiliser qu'en cas d'échec lors de l'extraction des factures.")

    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("🧱 Recalculer le fichier CSV"):
            try:
                gestion = get_gestionnaire_oauth(base_folder_id or None, user_email)
                drive = gestion.drive
                ok = rebuild_df_traitee_from_drive_nt(drive)
                if ok:
                    st.success(f"Fichier CSV reconstruit avec succès → données mises à jour.")
                    sync_df_traitee_local_from_drive(drive)
                    st.dataframe(pd.read_csv(CSV_TRAITE).head(50))
                else:
                    st.warning("Aucun fichier CSV accessibles depuis le Drive. disponible. Merci de lancer l’extraction de factures.")
            except Exception as e:
                st.exception(e)

    with c2:
        if st.button("👀 Visualiser le fichier CSV (200 lignes)"):
            df = lire_df_traitee()
            if df.empty:
                st.warning("Aucun CSV trouvé. Lancez d'abord l'extraction.")
            else:
                st.dataframe(df.head(200))

# --- 3) Analyse --------------------------------------------------------------
with onglets[2]:
    st.subheader("Analyse interactive : filtres, KPI & graphiques")

    # 🔹 Synchronise df_traitee.csv local avec Drive avant analyse
    gestion = get_gestionnaire_oauth(base_folder_id or None, user_email)
    drive = gestion.drive
    if not sync_df_traitee_local_from_drive(drive):
        st.info("Aucun df_traitee.csv sur Drive. Générez-le dans l’onglet Extraction ou Traitement.")
        st.stop()

    visu = Visualisation(csv_path=CSV_TRAITE)  # ✅ lit le CSV (copie Drive) garanti à jour
    df_init = visu.get_df_initial().copy()

    # Filtres
    with st.container():
        f1, f2, f3, f4 = st.columns([1,1,1,2])
        with f1:
            fournisseurs_filtre = st.multiselect(
                "Fournisseur", sorted(df_init['fournisseur'].dropna().unique().tolist())
            )
        with f2:
            familles_filtre = st.multiselect(
                "Famille", sorted([x for x in df_init['famille'].dropna().unique()])
            )
        with f3:
            # Filtre par libellé (designation)
            designations_all = (
                df_init['designation']
                .dropna()
                .astype(str)
                .sort_values()
                .unique()
                .tolist()
            )
            designations_filtre = st.multiselect("Article (designation)", options=designations_all[:1000])
        with f4:
            # Période : par défaut année en cours
            today = pd.Timestamp.today()
            debut_def = pd.Timestamp(year=today.year, month=1, day=1)
            debut, fin = st.date_input(
                "Période",
                value=(debut_def.date(), today.date()),
                format="DD/MM/YYYY",
            )

            start_date = end_date = None

            if isinstance(debut, date) and isinstance(fin, date):
                start_date = pd.to_datetime(debut)
                end_date = pd.to_datetime(fin)
                periode = (start_date, end_date)
            else:
                # Message discret uniquement pendant la sélection
                st.caption("⏳ Sélectionne une date de fin pour appliquer le filtre.")

    df_filtre = visu.appliquer_filtres(
        periode=periode,
        fournisseur=fournisseurs_filtre or None,
        famille=familles_filtre or None,
        article=None,
        designation=designations_filtre or None,
        mettre_a_jour=True,
    )

    # KPI
    st.markdown("---")
    kpis = visu.get_kpis()
    km1 = visu.get_kpis_m_1()
    kn1 = visu.get_kpis_n_1()

    # 🔹 Titre pour KPI de base
    st.markdown("### 📊 KPIs")

    cA, cB, cC, cD = st.columns(4)
    cA.metric("CA (filtré)", f"{kpis['chiffre_affaires']:.0f} €")
    cB.metric("Qtés vendues (filtré)", f"{kpis['quantites_vendues']:.0f}")
    cC.metric("Qtés commandées (filtré)", f"{kpis['quantites_commandees']:.0f}")
    cD.metric("Nb commandes", f"{kpis['nb_commandes']}")

    cE, cF, cG, cH = st.columns(4)
    cE.metric("Prix moyen", f"{kpis['prix_moyen_catalogue']:.2f} €")
    cF.metric("Prix médian", f"{kpis['prix_median_catalogue']['median']:.2f} €")
    # Helpers listes cohérentes avec Visualisation.get_nb_refs_actives / get_nouvelles_refs
    today = pd.to_datetime("today").normalize()
    lim_actives = today - pd.DateOffset(months=3)
    lim_nouvelles = today - pd.DateOffset(months=1)


    def _to_dt_list(lst):
        if isinstance(lst, list) and lst:
            return pd.to_datetime(lst, errors="coerce")
        return pd.to_datetime([])


    df_active = df_init[df_init["dates_commandes"].apply(lambda lst: (_to_dt_list(lst) >= lim_actives).any())].copy()
    df_new = df_init[df_init["dates_commandes"].apply(
        lambda lst: (len(_to_dt_list(lst)) > 0) and (_to_dt_list(lst).min() >= lim_nouvelles))].copy()


    def _enrich(df):
        # Dates utiles pour lecture
        dts = df["dates_commandes"].apply(_to_dt_list)
        df["premiere_commande"] = dts.apply(lambda x: x.min() if len(x) else pd.NaT)
        df["derniere_commande"] = dts.apply(lambda x: x.max() if len(x) else pd.NaT)
        cols = [c for c in ["code", "designation", "fournisseur", "famille", "nb_commandes", "premiere_commande",
                            "derniere_commande"] if c in df.columns]
        return df[cols].sort_values(["derniere_commande", "designation"], ascending=[False, True])


    cG.metric("Réfs actives (3 mois)", f"{kpis['nb_refs_actives']}")
    if cG.button("Voir", key="btn_show_actives"):
        st.session_state["show_actives"] = True

    cH.metric("Nouvelles réfs (1 mois)", f"{kpis['nouvelles_refs']}")
    if cH.button("Voir", key="btn_show_new"):
        st.session_state["show_new"] = True

    # Affichage (sous les KPI)
    if st.session_state.get("show_actives"):
        with st.expander("✅ Références actives (commande dans les 3 derniers mois)", expanded=True):
            st.dataframe(_enrich(df_active), use_container_width=True)
            st.download_button(
                "Télécharger (CSV)",
                data=_enrich(df_active).to_csv(index=False).encode("utf-8"),
                file_name="refs_actives.csv",
                mime="text/csv",
            )

    if st.session_state.get("show_new"):
        with st.expander("🆕 Nouvelles références (1ère commande dans le dernier mois)", expanded=True):
            st.dataframe(_enrich(df_new), use_container_width=True)
            st.download_button(
                "Télécharger (CSV)",
                data=_enrich(df_new).to_csv(index=False).encode("utf-8"),
                file_name="nouvelles_refs.csv",
                mime="text/csv",
            )
    # Séparateur + Titre pour les évolutions
    st.markdown("---")
    st.markdown("### 📈 Évolutions (M-1 & N-1)")

    # 🔹 Évolutions M-1
    cI, cJ = st.columns(2)
    with cI:
        st.metric("CA mois vs M-1", f"{km1['ca_mois']:.0f} €",
                  delta=("N/C" if km1['evolution_ca_m_1'] == "N/C" else f"{km1['evolution_ca_m_1']}%"))
    with cJ:
        st.metric("Qtés mois vs M-1", f"{km1['quantites_mois']:.0f}",
                  delta=("N/C" if km1['evolution_quantites_m_1'] == "N/C" else f"{km1['evolution_quantites_m_1']}%"))

    # 🔹 Évolutions N-1
    def _fmt_pct(v):
        try:
            return f"{float(v):.1f}%"
        except Exception:
            return "N/C"

    cK, cL = st.columns(2)
    cK.metric("Évol. volumétrie vs N-1", _fmt_pct(kpis.get('evolution_volumetrie', 'N/C')))
    cL.metric("Évol. CA vs N-1", _fmt_pct(kpis.get('evolution_ca', 'N/C')))

    st.markdown("---")

    # Graphiques
    g1 = visu.plot_ca_hebdo_comparatif()
    g2 = visu.plot_ca_cumule_comparatif()
    g3 = visu.plot_ca_hebdo_multi_annees()
    g4 = visu.plot_ca_cumule_prevision()

    st.plotly_chart(g1, use_container_width=True)
    st.plotly_chart(g2, use_container_width=True)
    st.plotly_chart(g3, use_container_width=True)
    st.plotly_chart(g4, use_container_width=True)

    st.markdown("---")
    with st.expander("🔎 Fichier CSV – aperçu"):
        st.dataframe(df_filtre.head(200))

# --- 4) Prédiction -----------------------------------------------------------
with onglets[3]:
    st.subheader("Prédiction : Bons de commande & Articles")

    # ... au début de l’onglet Prédiction
    csv_mtime = Path(CSV_TRAITE).stat().st_mtime if os.path.exists(CSV_TRAITE) else None
    if 'pred_csv_mtime' in st.session_state and st.session_state['pred_csv_mtime'] != csv_mtime:
        # Le CSV a changé -> on invalide les résultats en cache
        for k in ['pred_sg', 'pred_lbb', 'pred_apdv']:
            st.session_state.pop(k, None)
    st.session_state['pred_csv_mtime'] = csv_mtime

    st.markdown("### 📦 Bons de commande hebdo par fournisseur")
    action_col, info_col = st.columns([1,3])
    with action_col:
        if st.button("🧮 Prédire la commande"):
            try:
                visu_pred = Visualisation(csv_path=CSV_TRAITE)
                df_sg, df_lbb, df_apdv = visu_pred.predire_commandes()
                st.session_state['pred_sg'] = df_sg
                st.session_state['pred_lbb'] = df_lbb
                st.session_state['pred_apdv'] = df_apdv
                st.success("Prédiction terminée.")
            except Exception as e:
                st.exception(e)
    with info_col:
        st.caption("Exclut CHERITEL & AUTENTIK et fruits & légumes. Seuil: 10% du conditionnement.\n\nLes prédictions n'exclut pas de vérifier les stocks en magasin.")

    def _affiche_bon(nom, df):
        st.markdown(f"#### {nom}")
        if df is None or df.empty:
            st.info("Aucun article éligible.")
            return
        a_commander = df[df.get("quantite_a_commander", 0) > 0].copy()
        if a_commander.empty:
            st.success("✅ Aucun réapprovisionnement nécessaire (quantité en stock > seuil).")
        else:
            cols_aff = [c for c in ["code","designation","stock","quantite_a_commander","alerte"] if c in a_commander.columns]
            st.dataframe(a_commander[cols_aff].sort_values("designation"))
            csv = a_commander[cols_aff].to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"💾 Télécharger CSV {nom}",
                data=csv,
                file_name=f"bon_commande_{nom.lower()}.csv",
                mime="text/csv",
            )

    if 'pred_sg' in st.session_state or 'pred_lbb' in st.session_state or 'pred_apdv' in st.session_state:
        _affiche_bon("SUPERGROUP", st.session_state.get('pred_sg'))
        _affiche_bon("LBB", st.session_state.get('pred_lbb'))
        _affiche_bon("APDV", st.session_state.get('pred_apdv'))

    st.markdown("---")
    st.markdown("### 🔎 Simulation de stock par article (libellé)")
    try:
        visu_sim = Visualisation(csv_path=CSV_TRAITE)
        df_base = visu_sim.get_df_initial().copy()
        options = sorted(df_base["designation"].dropna().astype(str).unique().tolist())
        if options:
            sel = st.selectbox("Choisir un article", options=options)
            fig = visu_sim.afficher_commandes_et_stock(sel)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Impossible de générer la simulation pour cet article (pas assez de données présentes pour cet article).")
        else:
            st.info("Aucune désignation disponible.")
    except Exception as e:
        st.exception(e)


st.caption("En cas de problèmes recontrés sur l'application, merci de contacter : owen.devlogiciel@gmail.com")
