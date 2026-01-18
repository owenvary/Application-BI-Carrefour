import plotly.graph_objects as go
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import re

import math

# --- visualiser.py ---
#pb dans la visuamlisation du stock par article/prédiction commandes car factures ne colle pas avec le graph qui sort de la fonction
class Visualisation:
    def __init__(self, csv_path: str | None = None):
        self._csv_path = csv_path  # <-- nouveau
        self.df_initial = self.charger_df_csv()
        print("✅ DataFrame chargée :", self.df_initial.shape)
        if not self.df_initial.empty:
            self.convertir_listes_manuel()
        self.df_filtered = self.df_initial.copy()
        self.filtres_actuels = {}


    def detect_base_path(self):
        # Pointe vers la racine/data (et plus src/data)
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

    def charger_df_csv(self):
        # Si un chemin explicite est fourni par l’app, on l’utilise
        if self._csv_path and os.path.exists(self._csv_path):
            chemin_csv = self._csv_path
        else:
            base_path = self.detect_base_path()
            chemin_csv = os.path.join(base_path, 'Fichiers CSV', 'df_traitee.csv')

        try:
            df = pd.read_csv(chemin_csv)
        except Exception:
            return pd.DataFrame()
        return df


    def convertir_listes_manuel(self):
        colonnes_a_convertir = {
            'dates_commandes': 'date',
            'quantites_commandees': float,
            'quantites_vendues': float,
            'ca_article': float
        }

        for col, typ in colonnes_a_convertir.items():
            if col in self.df_initial.columns:
                def contient_valeurs_invalides(x):
                    if isinstance(x, str):
                        x = x.strip()
                        return any(inval in x.lower() for inval in ['none', 'nan', 'null']) or x == ''
                    return pd.isna(x)

                valeurs_invalides = self.df_initial[col][self.df_initial[col].apply(contient_valeurs_invalides)]

                if not valeurs_invalides.empty:
                    print(f"⚠️ Valeurs invalides détectées dans '{col}' :")
                    print(valeurs_invalides.head(5))

                self.df_initial[col] = self.df_initial[col].apply(
                    lambda x: self.parse_liste_manuel(x, type_element=typ)
                )

        # Étape 2 – remplacement des None dans ca_article par 0
        if 'ca_article' in self.df_initial.columns:
            self.df_initial['ca_article'] = self.df_initial['ca_article'].apply(
                lambda lst: [0 if x is None else x for x in lst] if isinstance(lst, list) else lst
            )

        # Étape 3 – ajout des colonnes isActive et isNew
        today = pd.Timestamp.today()
        trois_mois_avant = today - pd.DateOffset(months=3)
        un_mois_avant = today - pd.DateOffset(months=1)

        def get_is_active(dates):
            if isinstance(dates, list):
                return any(d >= trois_mois_avant for d in dates if pd.notna(d))
            return False

        def get_is_new(dates):
            if isinstance(dates, list) and len(dates) > 0:
                premiere = min(d for d in dates if pd.notna(d))
                return premiere >= un_mois_avant
            return False

        self.df_initial['isActive'] = self.df_initial['dates_commandes'].apply(get_is_active)
        self.df_initial['isNew'] = self.df_initial['dates_commandes'].apply(get_is_new)




    @staticmethod
    def parse_liste_manuel(chaine, type_element=str):
        if not isinstance(chaine, str) or chaine.strip() == '':
            return []

        chaine = chaine.strip().strip('[]')
        if chaine == '':
            return []

        elements = chaine.split(',')
        resultat = []

        for elt in elements:
            elt = elt.strip().strip("'").strip('"')

            if elt == '':
                continue

            # Extraire le contenu de np.float64(...) si présent
            match = re.match(r"np\.float64\((.*?)\)", elt)
            if match:
                elt = match.group(1)

            try:
                if type_element == str:
                    resultat.append(elt)
                elif type_element == float:
                    elt = elt.replace(',', '.')  # décimale française
                    resultat.append(float(elt))
                elif type_element == int:
                    resultat.append(int(float(elt)))  # au cas où elt soit un float déguisé
                elif type_element == 'date':
                    parsed_date = pd.to_datetime(elt, errors='coerce')
                    if pd.isna(parsed_date):
                        print(f"❌ Date invalide détectée : '{elt}'")
                    resultat.append(parsed_date)
                else:
                    resultat.append(elt)
            except Exception as e:
                print(f"❌ Échec de conversion de '{elt}' en {type_element} → {e}")
                resultat.append(None)

        return resultat





    def ca_total(self):
        total_global = sum(
            sum(x for x in ca_list if x is not None)
            for ca_list in self.df_initial['ca_article'] if isinstance(ca_list, list)
        )
        print(f"Chiffre d'affaires total cumulé : {total_global}")
        return total_global



    def quantites_commandees_totales(self):
        if 'quantites_commandees' not in self.df_initial.columns:
            print("⚠️ Colonne 'quantites_commandees' non trouvée")
            return None
        # On somme les listes dans chaque ligne
        total = 0
        for qlist in self.df_initial['quantites_commandees']:
            if isinstance(qlist, list):
                total += sum([q for q in qlist if isinstance(q, (int,float))])
            else:
                # au cas où ce n’est pas encore converti
                try:
                    total += float(qlist)
                except Exception:
                    pass
        print(f"Quantités commandées totales = {total}")
        return total

    def check_none_values(self):
        none_counts = self.df_initial['ca_article'].apply(lambda lst: sum(x is None for x in lst) if isinstance(lst, list) else 0)
        print("Nombre de None par ligne dans ca_article :", none_counts.describe())


#Get dates importantes

    def get_index_semaines(self):
        """
        Retourne un index de semaines aligné du lundi de la première commande globale
        au lundi de la semaine courante.
        """
        # Obtenir la date de la première commande alignée sur lundi
        df_copy = self.df_initial.copy()
        toutes_dates = df_copy['dates_commandes'].explode()
        date_min = toutes_dates.min()
        date_debut = date_min - timedelta(days=date_min.weekday())

        # Obtenir le lundi de la semaine actuelle
        today = pd.Timestamp.today()
        date_fin = today - timedelta(days=today.weekday())

        # Créer l'index hebdomadaire
        index_semaines = pd.date_range(start=date_debut, end=date_fin, freq='W-MON')

        return index_semaines


    def get_annees_disponibles(self):
        # Récupère le lundi de la première commande
        date_premiere_commande = self.get_date_premiere_commande_globale()

        if pd.isna(date_premiere_commande):
            return []

        annee_min = date_premiere_commande.year
        annee_max = datetime.today().year  # ou annee la plus récente dans les données

        return list(range(annee_min, annee_max + 1))



    def get_date_premiere_commande_globale(self):
        df_copy = self.df_initial.copy()
        toutes_dates = df_copy['dates_commandes'].explode()
        date_min = toutes_dates.min()
        # Retourne le lundi de la semaine de la date min
        return date_min - timedelta(days=date_min.weekday())

    def get_date_semaine_courante(self):
        today = pd.Timestamp.today()
        return today - timedelta(days=today.weekday())

    def filtrer_par_periode(self, ligne, idx_debut, idx_fin, date_debut, date_fin):
        dates = ligne['dates_commandes']

        # Filtrer les dates de commande dans l’intervalle réel (utile pour KPI nb commandes)
        if isinstance(dates, list):
            dates_filtrees = [d for d in dates if date_debut <= d <= date_fin]
        else:
            dates_filtrees = []

        # Slicer les données quantitatives
        def safe_slice(lst):
            if isinstance(lst, list) and len(lst) >= idx_fin:
                return lst[idx_debut:idx_fin]
            elif isinstance(lst, list):
                # Complète avec des zéros si la liste est trop courte
                return (lst[idx_debut:] if idx_debut < len(lst) else []) + [0.0] * max(0, idx_fin - len(lst))
            else:
                return [0.0] * (idx_fin - idx_debut)

        return (
            dates_filtrees,
            safe_slice(ligne.get('quantites_commandees')),
            safe_slice(ligne.get('quantites_vendues')),
            safe_slice(ligne.get('ca_article')),
        )



#--------------KPI---------------

    def appliquer_filtres(self, periode=None, fournisseur=None, famille=None, article=None, designation=None,
                          mettre_a_jour=True):
        """
        Ajout du filtre `designation` (liste de libellés).
        `article` conserve le filtre par code pour compat.
        """
        self.filtres_actuels = {
            "periode": periode,
            "fournisseur": fournisseur,
            "famille": famille,
            "article": article,
            "designation": designation,
        }

        df = self.df_initial.copy()

        # fournisseur
        if fournisseur:
            if isinstance(fournisseur, str):
                fournisseur = [fournisseur]
            df = df[df['fournisseur'].isin(fournisseur)]

        # famille
        if famille:
            if isinstance(famille, str):
                famille = [famille]
            df = df[df['famille'].isin(famille)]

        # article (code)
        if article:
            if isinstance(article, str):
                article = [article]
            df = df[df['code'].astype(str).isin([str(a) for a in article])]

        # designation (libellé)
        if designation:
            if isinstance(designation, str):
                designation = [designation]
            labels = [str(x) for x in designation]
            df = df[df['designation'].astype(str).isin(labels)]

        # période
        if periode:
            date_debut, date_fin = periode

            index_semaines = self.get_index_semaines()
            idx_debut = max(0, index_semaines.get_indexer([pd.to_datetime(date_debut)], method='pad')[0])
            idx_fin = min(len(index_semaines),
                          index_semaines.get_indexer([pd.to_datetime(date_fin)], method='pad')[0] + 1)

            colonnes_filtrees = {
                'dates_commandes': [],
                'quantites_commandees': [],
                'quantites_vendues': [],
                'ca_article': [],
                'nb_commandes': []
            }

            for _, ligne in df.iterrows():
                dates_f, qcmd_f, qvendues_f, ca_f = self.filtrer_par_periode(
                    ligne, idx_debut, idx_fin, date_debut, date_fin
                )
                colonnes_filtrees['dates_commandes'].append(dates_f)
                colonnes_filtrees['quantites_commandees'].append(qcmd_f)
                colonnes_filtrees['quantites_vendues'].append(qvendues_f)
                colonnes_filtrees['ca_article'].append(ca_f)
                colonnes_filtrees['nb_commandes'].append(len(dates_f))

            df['dates_commandes'] = colonnes_filtrees['dates_commandes']
            df['quantites_commandees'] = colonnes_filtrees['quantites_commandees']
            df['quantites_vendues'] = colonnes_filtrees['quantites_vendues']
            df['ca_article'] = colonnes_filtrees['ca_article']
            df['nb_commandes'] = colonnes_filtrees['nb_commandes']

        # 🔹 Mise à jour du cache interne
        if mettre_a_jour:
            self.df_filtered = df

        return df

    def reset_filtres(self):
        """Réinitialise les filtres sur df_exploded"""
        self.df_filtered = self.get_df_initial()
        self.filtres_actuels = {}
        print("🔄 Filtres réinitialisés.")

    def get_filtres_actuels(self):
        return self.filtres_actuels

    def get_filtres_sans_periode(self):
        """Retourne les filtres actuels sauf la période"""
        filtres = self.get_filtres_actuels().copy()
        filtres.pop('periode', None)
        return filtres

    def get_filtered_df(self):
        return self.df_filtered

    def get_df_initial(self):
        return self.df_initial

    def get_ca(self):
        df = self.get_filtered_df()
        return df['ca_article'].apply(lambda x: sum(x) if isinstance(x, list) else 0).sum()

    def get_quantites_vendues(self):
        df = self.get_filtered_df()
        return df['quantites_vendues'].apply(lambda x: sum(x) if isinstance(x, list) else 0).sum()

    def get_quantites_commandees(self):
        df = self.get_filtered_df()

        def sum_pos(lst):
            if not isinstance(lst, list):
                return 0
            return sum(x for x in lst if pd.notna(x) and x > 0)

        return df['quantites_commandees'].apply(sum_pos).sum()

    def get_prix_moyen(self):
        df = self.get_filtered_df()
        return df['prix_unitaire_moyen'].mean() if 'prix_unitaire_moyen' in df.columns else 0

    def get_prix_central(self):
        df = self.get_filtered_df()
        if 'prix_unitaire_moyen' not in df.columns or df.empty:
            return {'moyen': 0, 'median': 0, 'combine': 0}

        prix_moyen = df['prix_unitaire_moyen'].mean()
        prix_median = df['prix_unitaire_moyen'].median()

        return {
            'moyen': round(prix_moyen, 2),
            'median': round(prix_median, 2),

        }

    def get_nb_commandes(self):
        df = self.get_filtered_df()

        if 'fournisseur' not in df.columns or 'dates_commandes' not in df.columns:
            return 0

        total = 0
        for fournisseur, groupe in df.groupby('fournisseur'):
            toutes_les_dates = []
            for dates in groupe['dates_commandes']:
                if isinstance(dates, list):
                    toutes_les_dates.extend(dates)

            nb_uniques = len(set(toutes_les_dates))
            total += nb_uniques

        return total

    def get_frequence_commandes(self):
        nb_commandes = self.get_nb_commandes()
        periode = self.filtres_actuels.get("periode")
        if periode:
            debut, fin = periode
            nb_semaines = (fin - debut).days // 7 + 1
            return round(nb_commandes / nb_semaines, 2) if nb_semaines > 0 else 0
        return 0

        # Indépendants des filtres

    def get_nb_refs_actives(self):
        df = self.df_initial
        date_limite = pd.to_datetime('today') - pd.DateOffset(months=3)

        return df['dates_commandes'].apply(
            lambda lst: any(pd.to_datetime(d) >= date_limite for d in lst) if isinstance(lst, list) else False
        ).sum()

    def get_nouvelles_refs(self):
        df = self.get_df_initial()
        date_debut_mois = pd.to_datetime('today') - pd.DateOffset(months=1)

        return df['dates_commandes'].apply(
            lambda lst: min(pd.to_datetime(lst)) >= date_debut_mois if isinstance(lst, list) and lst else False
        ).sum()

    def get_evolution_volumetrie_n1(self):
        periode = self.filtres_actuels.get("periode")
        if not periode:
            return "N/C"

        debut, fin = periode
        delta = fin - debut
        debut_n1 = debut - pd.DateOffset(years=1)
        fin_n1 = debut_n1 + delta

        df_actuel = self.get_filtered_df()
        df_n1 = self.appliquer_filtres(periode=(debut_n1, fin_n1), mettre_a_jour=False)

        vol_actuel = df_actuel['quantites_commandees'].apply(lambda x: sum(x) if isinstance(x, list) else 0).sum()
        vol_n1 = df_n1['quantites_commandees'].apply(lambda x: sum(x) if isinstance(x, list) else 0).sum()

        if vol_n1 == 0:
            return "N/C" if vol_actuel == 0 else "+∞"

        return round((vol_actuel - vol_n1) / vol_n1 * 100, 2)

    def get_evolution_ca_n1(self):
        periode = self.filtres_actuels.get("periode")
        if not periode:
            return "N/C"

        debut, fin = periode
        delta = fin - debut
        debut_n1 = debut - pd.DateOffset(years=1)
        fin_n1 = debut_n1 + delta

        df_actuel = self.get_filtered_df()
        df_n1 = self.appliquer_filtres(periode=(debut_n1, fin_n1), mettre_a_jour=False)

        ca_actuel = df_actuel['ca_article'].apply(lambda x: sum(x) if isinstance(x, list) else 0).sum()
        ca_n1 = df_n1['ca_article'].apply(lambda x: sum(x) if isinstance(x, list) else 0).sum()

        if ca_n1 == 0:
            return "N/C" if ca_actuel == 0 else "+∞"

        return round((ca_actuel - ca_n1) / ca_n1 * 100, 2)

    def get_kpis(self):
        return {
            "chiffre_affaires": self.get_ca(),
            "quantites_vendues": self.get_quantites_vendues(),
            "quantites_commandees": self.get_quantites_commandees(),
            "prix_moyen_catalogue": self.get_prix_moyen(),
            "prix_median_catalogue": self.get_prix_central(),
            "nb_commandes": self.get_nb_commandes(),
            "frequence_commandes": self.get_frequence_commandes(),
            "nb_refs_actives": self.get_nb_refs_actives(),
            "nouvelles_refs": self.get_nouvelles_refs(),
            "evolution_volumetrie": self.get_evolution_volumetrie_n1(),
            "evolution_ca": self.get_evolution_ca_n1(),
        }

    def get_df_comparatif(self, type_periode="mois", decalage=0):
        """
        Retourne un DataFrame filtré pour la période spécifiée :
        - type_periode : "mois" ou "annee"
        - decalage : 0 = période actuelle, 1 = période précédente, etc.
        """
        today = pd.to_datetime("today").normalize()

        if type_periode == "mois":
            fin = today
            debut = today.replace(day=1) - pd.DateOffset(months=decalage)
            fin = debut + pd.DateOffset(days=(today.day - 1))  # même jour du mois que today
        elif type_periode == "annee":
            fin = today
            debut = today.replace(month=1, day=1) - pd.DateOffset(years=decalage)
            fin = debut + pd.DateOffset(days=(today.dayofyear - 1))
        else:
            raise ValueError("type_periode doit être 'mois' ou 'annee'")

        filtres_sans_periode = self.get_filtres_sans_periode()
        return self.appliquer_filtres(periode=(debut, fin), **filtres_sans_periode, mettre_a_jour=False)

    def get_kpis_m_1(self):
        df_mois = self.get_df_comparatif("mois", decalage=0)
        df_prec = self.get_df_comparatif("mois", decalage=1)

        def somme(df, col):
            return df[col].apply(lambda x: sum(x) if isinstance(x, list) else 0).sum()

        ca_1 = round(somme(df_mois, 'ca_article'), 2)
        ca_2 = round(somme(df_prec, 'ca_article'), 2)
        qte_1 = somme(df_mois, 'quantites_commandees')
        qte_2 = somme(df_prec, 'quantites_commandees')

        return {
            "ca_mois": ca_1,
            "ca_mois_prec": ca_2,
            "evolution_ca_m_1": "N/C" if ca_2 == 0 else round((ca_1 - ca_2) / ca_2 * 100, 2),
            "quantites_mois": qte_1,
            "quantites_mois_prec": qte_2,
            "evolution_quantites_m_1": "N/C" if qte_2 == 0 else round((qte_1 - qte_2) / qte_2 * 100, 2),
        }

    def get_kpis_n_1(self):
        df_annee = self.get_df_comparatif("annee", decalage=0)
        df_n1 = self.get_df_comparatif("annee", decalage=1)

        def somme(df, col):
            return df[col].apply(lambda x: sum(x) if isinstance(x, list) else 0).sum()

        ca_now = round(somme(df_annee, 'ca_article'), 2)
        ca_n1 = round(somme(df_n1, 'ca_article'), 2)
        qte_now = somme(df_annee, 'quantites_commandees')
        qte_n1 = somme(df_n1, 'quantites_commandees')

        return {
            "ca_annuel": ca_now,
            "ca_annuel_n_1": ca_n1,
            "evolution_ca_n_1": "N/C" if ca_n1 == 0 else round((ca_now - ca_n1) / ca_n1 * 100, 2),
            "quantites_annuelles": qte_now,
            "quantites_annuelles_n_1": qte_n1,
            "evolution_quantites_n_1": "N/C" if qte_n1 == 0 else round((qte_now - qte_n1) / qte_n1 * 100, 2),
        }

    #-------------GRAPHS-------------

# test

    def get_graphs(self):
        df = self.get_filtered_df()
        df_initial = self.get_df_initial()

        fig_hebdo_comparatif = self.plot_ca_hebdo_comparatif()
        fig_cumule_comparatif = self.plot_ca_cumule_comparatif()

        fig_ca_hebdo_multi_annees = self.plot_ca_hebdo_multi_annees()
        fig_pred = self.plot_ca_cumule_prevision()

        graphs = {
            "CA hebdo (global vs filtré)": fig_hebdo_comparatif,
            "CA cumulé (global vs filtré)": fig_cumule_comparatif,
            "CA hebdo cumulé par année": fig_ca_hebdo_multi_annees,
            "Projection CA cumulé (réel vs prévision)": fig_pred,
        }

        for titre, graph in graphs.items():
            print(titre)
            graph.show()

    def plot_ca_hebdo_comparatif(self):
        import plotly.graph_objects as go

        df_global = self.get_df_initial().copy()
        df_filtre = self.get_filtered_df().copy()

        # Dates limites
        today = pd.Timestamp.today()
        annee = today.year
        debut_annee = pd.Timestamp(f"{annee}-01-01")
        debut_lundi = debut_annee - timedelta(days=debut_annee.weekday())
        fin_lundi = today - timedelta(days=today.weekday())
        semaines = pd.date_range(start=debut_lundi, end=fin_lundi, freq='W-MON')
        nb_sem = len(semaines)

        # Fonction CA hebdo
        def get_ca_hebdo(df):
            df['ca_article'] = df['ca_article'].apply(lambda lst: lst[-nb_sem:] if isinstance(lst, list) else [])
            return [df['ca_article'].apply(lambda x: x[i] if i < len(x) else 0).sum() for i in range(nb_sem)]

        ca_global = get_ca_hebdo(df_global)
        ca_filtre = get_ca_hebdo(df_filtre)

        # Tracé
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=semaines, y=ca_global, mode='lines+markers', name='CA hebdo (global)'))
        fig.add_trace(go.Scatter(x=semaines, y=ca_filtre, mode='lines+markers', name='CA hebdo (filtré)'))

        fig.update_layout(
            title="Comparaison du CA hebdomadaire - Global vs Filtré",
            xaxis_title="Semaine",
            yaxis_title="CA (€)",
            hovermode="x unified",
            template="plotly_white"
        )
        return fig

    def plot_ca_cumule_comparatif(self):
        import plotly.graph_objects as go

        df_global = self.get_df_initial().copy()
        df_filtre = self.get_filtered_df().copy()

        today = pd.Timestamp.today()
        annee = today.year
        debut_annee = pd.Timestamp(f"{annee}-01-01")
        debut_lundi = debut_annee - timedelta(days=debut_annee.weekday())
        fin_lundi = today - timedelta(days=today.weekday())
        semaines = pd.date_range(start=debut_lundi, end=fin_lundi, freq='W-MON')
        nb_sem = len(semaines)

        # Fonction CA cumulé
        def get_ca_cumule(df):
            df['ca_article'] = df['ca_article'].apply(lambda lst: lst[-nb_sem:] if isinstance(lst, list) else [])
            cumul = 0
            cumul_list = []
            for i in range(nb_sem):
                val = df['ca_article'].apply(lambda x: x[i] if i < len(x) else 0).sum()
                cumul += val
                cumul_list.append(cumul)
            return cumul_list

        ca_cumule_global = get_ca_cumule(df_global)
        ca_cumule_filtre = get_ca_cumule(df_filtre)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=semaines, y=ca_cumule_global, mode='lines+markers', name='CA cumulé (global)'))
        fig.add_trace(go.Scatter(x=semaines, y=ca_cumule_filtre, mode='lines+markers', name='CA cumulé (filtré)'))

        fig.update_layout(
            title="Comparaison du CA cumulé - Global vs Filtré",
            xaxis_title="Semaine",
            yaxis_title="CA cumulé (€)",
            hovermode="x unified",
            template="plotly_white"
        )
        return fig

    def plot_ca_hebdo_multi_annees(self):
        #import plotly.graph_objects as go

        df_initial = self.get_df_initial().copy()

        # 1. Date de référence = plus ancienne date de commande
        date_ref = pd.Timestamp(min([min(dates) for dates in df_initial['dates_commandes'] if dates]))

        # 2. Créer un DataFrame long : une ligne par semaine
        all_rows = []
        for _, row in df_initial.iterrows():
            ca_list = row['ca_article']
            if not isinstance(ca_list, list):
                continue
            for i, ca in enumerate(ca_list):
                semaine_date = date_ref + pd.Timedelta(weeks=i)
                all_rows.append({'date': semaine_date, 'ca': ca})

        df_long = pd.DataFrame(all_rows)

        # 3. Ajouter colonne année et CA cumulé par année
        df_long['annee'] = df_long['date'].dt.year
        df_long['jour_mois'] = df_long['date'].apply(lambda d: d.replace(year=2000))  # Pour tracer sans tenir compte de l'année

        df_cumule = df_long.groupby(['annee', 'jour_mois']).agg({'ca': 'sum'}).groupby(level=0).cumsum().reset_index()

        # 4. Tracer les courbes par année
        fig = go.Figure()
        annees = sorted(df_cumule['annee'].unique())

        for annee in annees:
            df_annee = df_cumule[df_cumule['annee'] == annee]
            fig.add_trace(go.Scatter(
                x=df_annee['jour_mois'],
                y=df_annee['ca'],
                mode='lines+markers',
                name=str(annee)
            ))

        fig.update_layout(
            title="CA cumulé hebdomadaire - Comparaison par année",
            xaxis_title="Date (mois/jour)",
            yaxis_title="CA cumulé (€)",
            xaxis=dict(tickformat="%b"),  # Affiche uniquement les mois
            hovermode="x unified",
            template="plotly_white"
        )

        return fig

    def plot_ca_cumule_prevision(self):
        import plotly.graph_objects as go
        from sklearn.linear_model import LinearRegression
        import numpy as np

        df_initial = self.get_df_initial()
        date_actuelle = pd.Timestamp.today()
        annee_actuelle = date_actuelle.year
        date_debut_annee = pd.Timestamp(f"{annee_actuelle}-01-01")
        date_debut_lundi = date_debut_annee - timedelta(days=date_debut_annee.weekday())
        date_fin_lundi = date_actuelle - timedelta(days=date_actuelle.weekday())

        semaines = pd.date_range(start=date_debut_lundi, end=date_fin_lundi, freq='W-MON')
        nb_sem = len(semaines)

        df_temp = df_initial.copy()
        df_temp['ca_article'] = df_temp['ca_article'].apply(lambda lst: lst[-nb_sem:] if isinstance(lst, list) else [])

        ca_hebdo = []
        ca_hebdo_cumule = []
        cumul = 0

        for i in range(nb_sem):
            somme_semaine = df_temp['ca_article'].apply(lambda x: x[i] if i < len(x) else 0).sum()
            ca_hebdo.append(somme_semaine)
            cumul += somme_semaine
            ca_hebdo_cumule.append(cumul)

        # 1. Préparer X (semaines) et y (CA cumulé)
        X = np.arange(1, nb_sem + 1).reshape(-1, 1)
        y = np.array(ca_hebdo_cumule)

        # 2. Entraîner la régression linéaire
        model = LinearRegression()
        model.fit(X, y)

        # 3. Étendre jusqu’à la semaine 52
        semaines_totales = pd.date_range(start=date_debut_lundi, periods=52, freq='W-MON')
        X_future = np.arange(1, 53).reshape(-1, 1)
        y_pred = model.predict(X_future)

        # 4. Tracer les deux courbes
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=semaines,
            y=ca_hebdo_cumule,
            mode='lines+markers',
            name='CA cumulé réel',
            line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=semaines_totales,
            y=y_pred,
            mode='lines',
            name='Prédiction linéaire',
            line=dict(color='red', dash='dash')
        ))

        fig.update_layout(
            title=f"CA cumulé {annee_actuelle} + prédiction linéaire",
            xaxis_title='Semaine',
            yaxis_title='CA cumulé (€)',
            hovermode='x unified',
            template='plotly_white'
        )

        return fig




    def detecter_conditionnement(self, code_article):
        """
        Estime le conditionnement comme étant la plus petite quantité STRICTEMENT POSITIVE
        jamais commandée pour un article donné.
        """
        df = self.get_df_initial()
        qtes = df.loc[df["code"] == code_article, "quantites_commandees"]

        qtes = [q for q in qtes.explode() if pd.notna(q) and q > 0]

        if len(qtes) == 0:
            return 1  # raisonnable si aucune donnée

        return min(qtes)

    def detecter_conditionnement_par_designation(self, designation: str) -> int:
        row = self._select_article_by_designation(designation)
        if row is None:
            return 1
        return self.detecter_conditionnement(row["code"])


    def simuler_stock_et_commandes(self, qte_initiale, prix_unitaire, marge, ca_prevu, conditionnement):
        """
        Simule le stock et les commandes hebdomadaires selon le CA prévu.
        """
        stock = qte_initiale
        commandes = []
        stock_semaine = []

        for ca in ca_prevu:
            qte_prevue = ca / (prix_unitaire * marge)
            qte_prevue = np.ceil(qte_prevue)

            # Commande si stock < 20% du conditionnement
            seuil = 0.2 * conditionnement
            if stock < seuil:
                manque = qte_prevue + conditionnement - stock
                commande = int(np.ceil(manque / conditionnement) * conditionnement)
                stock += commande
            else:
                commande = 0

            stock -= qte_prevue
            stock = max(0, stock)

            commandes.append(commande)
            stock_semaine.append(int(round(stock)))

        return commandes, stock_semaine

    def afficher_commandes_et_stock(self, designation: str):
        """
        Retourne la figure de simulation stock/commandes pour un libellé (designation).
        Exclut CHERITEL et AUTENTIK. Retourne None si données insuffisantes.
        """
        df = self.get_df_initial()
        row = self._select_article_by_designation(designation)
        if row is None:
            return None

        fournisseur = row['fournisseur']
        if fournisseur in ['CHERITEL', 'AUTENTIK']:
            return None

        code = row['code']
        prix = row['prix_unitaire_moyen']
        designation = row['designation']
        marge = 1.3
        ca_article = row['ca_article']
        quantites_commandes = row['quantites_commandees']
        dates_commandes = row['dates_commandes']

        if not isinstance(ca_article, list) or len(ca_article) < 4:
            return None

        dates_commandes = pd.to_datetime(dates_commandes)
        premiere_commande = min(dates_commandes)
        start_date = premiere_commande - pd.Timedelta(days=premiere_commande.weekday())
        aujourdhui = pd.to_datetime("today").normalize()

        nb_sem_pass = len(ca_article)
        nb_sem_fut = 4
        total_sem = nb_sem_pass + nb_sem_fut
        toutes_les_dates = [start_date + pd.Timedelta(weeks=i) for i in range(total_sem)]

        # Régression sur le CA cumulé
        x = np.arange(nb_sem_pass).reshape(-1, 1)
        y = np.cumsum(ca_article)
        model = LinearRegression().fit(x, y)
        y_pred = model.predict(np.arange(total_sem).reshape(-1, 1))
        ca_prevu_total = np.diff(np.insert(y_pred, 0, 0))
        ca_prevu_total = np.clip(ca_prevu_total, 0, None)

        # Conditionnement basé sur le code
        conditionnement = self.detecter_conditionnement(code)
        commandes, stock = self.simuler_stock_et_commandes(
            qte_initiale=0,
            prix_unitaire=prix,
            marge=marge,
            ca_prevu=ca_prevu_total,
            conditionnement=conditionnement
        )

        # Fenêtre : 3 mois avant aujourd’hui, +4 semaines
        date_debut_aff = aujourdhui - pd.DateOffset(months=3)
        date_fin_aff = aujourdhui + pd.DateOffset(weeks=4)
        masque = [(d >= date_debut_aff) and (d <= date_fin_aff) for d in toutes_les_dates]

        dates_f = [d for i, d in enumerate(toutes_les_dates) if masque[i]]
        cmd_f = [c for i, c in enumerate(commandes) if masque[i]]
        stock_f = [s for i, s in enumerate(stock) if masque[i]]

        cmd_passees = [c if d < aujourdhui else 0 for c, d in zip(cmd_f, dates_f)]
        cmd_futures = [c if d >= aujourdhui else 0 for c, d in zip(cmd_f, dates_f)]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=dates_f, y=cmd_passees, name="Commandes passées"))
        fig.add_trace(go.Bar(x=dates_f, y=cmd_futures, name="Commandes futures"))
        fig.add_trace(go.Scatter(x=dates_f, y=stock_f, name="Stock simulé", mode='lines+markers'))

        titre = f"Prédiction des commandes – {designation} ({fournisseur})"
        nb_commandes = len([q for q in (quantites_commandes or []) if q and q > 0])
        if nb_sem_pass < 5 or nb_commandes < 2:
            titre += "\n⚠️ Données insuffisantes, prédiction potentiellement bruitée"

        fig.update_layout(
            title=titre,
            xaxis_title="Semaine",
            yaxis_title="Commandes / Stock",
            legend=dict(x=0.01, y=0.99),
            bargap=0.2,
            template="plotly_white"
        )
        return fig


    def predire_commandes(self):

        print("🔎 Prédiction des commandes hebdomadaires...")

        # --- Étape 1 : Récupération et filtrage du DataFrame de base ---
        df = self.get_df_initial().copy()

        df = df[
            (~df["fournisseur"].isin(["CHERITEL", "AUTENTIK"])) &
            (df["famille"] != "Fruits et legumes") &
            (df["isActive"] == True)
        ]

        # --- Étape 2 : Création de DataFrames par fournisseur ---
        fournisseurs = ["SUPERGROUP", "LBB", "APDV"]
        resultats = {}

        for fournisseur in fournisseurs:
            df_fourn = df[df["fournisseur"] == fournisseur].copy()

            df_fourn["commandes_totales"] = np.nan
            df_fourn["ventes_totales"] = np.nan
            df_fourn["stock"] = np.nan
            df_fourn["quantite_a_commander"] = 0
            df_fourn["alerte"] = ""

            for idx, row in df_fourn.iterrows():
                quantites_commandes = row["quantites_commandees"]
                quantites_vendues = row["quantites_vendues"]
                commandes_non_vides = [q for q in row["quantites_commandees"] if q > 0]
                conditionnement = min(commandes_non_vides) if commandes_non_vides else 0


                if not quantites_commandes or not quantites_vendues:
                    df_fourn.at[idx, "alerte"] = "Manque données historiques"
                    continue

                if len(quantites_commandes) != len(quantites_vendues):
                    df_fourn.at[idx, "alerte"] = "Longueurs incohérentes"
                    continue

                commandes_totales = sum(quantites_commandes)
                ventes_totales = sum(quantites_vendues)
                stock = round(commandes_totales - ventes_totales)

                df_fourn.at[idx, "commandes_totales"] = commandes_totales
                df_fourn.at[idx, "ventes_totales"] = ventes_totales
                df_fourn.at[idx, "stock"] = stock

                # Seuil de commande = 10% du conditionnement
                seuil = 0.1 * conditionnement
                if stock < seuil:
                    quantite_a_commander = round(conditionnement)
                    df_fourn.at[idx, "quantite_a_commander"] = quantite_a_commander

            # --- Étape 3 : Impression console du bon de commande ---
            a_commander = df_fourn[df_fourn["quantite_a_commander"] > 0]
            if not a_commander.empty:
                print(f"\n🧾 Bon de commande {fournisseur} :\n")
                for _, ligne in a_commander.iterrows():
                    code = ligne["code"]
                    designation = ligne["designation"]
                    qte = int(ligne["quantite_a_commander"])
                    print(f"• {code} - {designation} : {qte} unités à commander")
            else:
                print(f"\n✅ Aucun réapprovisionnement nécessaire pour {fournisseur}.")

            resultats[fournisseur.lower()] = df_fourn

        print("\n✅ Prédiction terminée.\n")
        return resultats["supergroup"], resultats["lbb"], resultats["apdv"]

    def _select_article_by_designation(self, designation: str):
        df = self.get_df_initial().copy()

        def _norm(s: str) -> str:
            s = re.sub(r"\s+", " ", str(s)).strip().casefold()
            s = s.replace("–", "-")  # tirets différents
            return s

        key = _norm(designation)
        desig_norm = df["designation"].astype(str).apply(_norm)

        candidates = df[desig_norm == key]
        if candidates.empty:
            candidates = df[desig_norm.str.contains(re.escape(key), na=False)]

        # Exclusions globales
        candidates = candidates[~candidates["fournisseur"].isin(["CHERITEL", "AUTENTIK"])]
        if candidates.empty:
            return None

        def _sum_list(x):
            try:
                return float(sum(v for v in (x or []) if isinstance(v, (int, float))))
            except:
                return 0.0

        def _last_date(lst):
            try:
                ds = [pd.to_datetime(d) for d in (lst or []) if pd.notna(d)]
                return max(ds) if ds else pd.NaT
            except:
                return pd.NaT

        # Scores pour le tri
        candidates = candidates.assign(
            _isActive=(
                candidates["isActive"].fillna(False).astype(bool) if "isActive" in candidates.columns else False),
            _vol=candidates["quantites_vendues"].apply(_sum_list),
            _last=candidates["dates_commandes"].apply(_last_date),
        )

        # Priorités : plus récent, actif, volume
        row = candidates.sort_values(["_last", "_isActive", "_vol"], ascending=[False, False, False]).iloc[0]
        return row

#TEST2