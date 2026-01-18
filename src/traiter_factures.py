from datetime import timedelta, date
import pandas as pd
import os
import numpy as np
import sys
import math

class TraiterDFs:
    def __init__(self, dfs: dict, base_path):
        self.dfs = dfs
        self.df_concatenee = None
        self.df_articles_groupes = None
        self.base_path = base_path


    def traiter_df(self):
        self._concatener_dfs()
        self._regrouper_commandes_par_article()
        self.etaler_ventes_cheritel()
        self.etaler_ventes_autentik()
        self.etaler_ventes_apdv_lbb_sg()
        self.exporter_csv()
        return self.df_articles_groupes

    def get_date_premiere_commande_globale(self):
        toutes_dates = self.df_articles_groupes['dates_commandes'].explode()
        date_min = toutes_dates.min()
        # Retourne le lundi de la semaine de la date min
        return date_min - timedelta(days=date_min.weekday())

    def get_date_semaine_courante(self):
        today = pd.Timestamp.today()
        return today - timedelta(days=today.weekday())

    def get_toutes_semaines(self):
        """
        Calcule la liste des dates des lundis (début de semaine) entre
        la date de la première commande et la semaine courante.
        """
        date_debut = self.get_date_premiere_commande_globale()
        date_fin = self.get_date_semaine_courante()
        return pd.date_range(start=date_debut, end=date_fin, freq='W-MON')


    def _concatener_dfs(self):
        dfs_harmonisees = []
        for fournisseur, df in self.dfs.items():
            df_harmo = self._harmoniser_df(df, fournisseur)
            dfs_harmonisees.append(df_harmo)

        self.df_concatenee = pd.concat(dfs_harmonisees, ignore_index=True)

    def _regrouper_commandes_par_article(self):
        df = self.df_concatenee.copy()
        df['date_commande'] = pd.to_datetime(df['date_commande'], errors='coerce')

        # on récupère la liste des semaines dynamiquement
        toutes_semaines = pd.date_range(start=df['date_commande'].min() - pd.to_timedelta(df['date_commande'].min().weekday(), unit='d'),
                                        end=pd.Timestamp.today() - pd.to_timedelta(pd.Timestamp.today().weekday(), unit='d'),
                                        freq='W-MON')

        def regrouper(article_df):
            quantites_par_semaine = article_df.groupby(article_df['date_commande'].dt.to_period('W').apply(lambda r: r.start_time))['quantite'].sum().to_dict()
            quantites_commandees = [quantites_par_semaine.get(s, 0) for s in toutes_semaines]

            return pd.Series({
                'prix_unitaire_moyen': article_df.sort_values('date_commande')['prix_unitaire_ht'].iloc[-1],
                'quantites_commandees': quantites_commandees,
                'dates_commandes': sorted([d.strftime('%Y-%m-%d') for d in article_df['date_commande'].dropna()]),
                'nb_commandes': (article_df['quantite'] > 0).sum(),
                'famille': article_df['famille'].iloc[0] if not article_df['famille'].empty else None,
                'fournisseur': article_df['fournisseur'].iloc[0],
                'type_code': article_df['type_code'].iloc[0],
                'designation': article_df['designation'].iloc[0]
            })

        df_grouped = df.groupby('code', group_keys=False).apply(regrouper).reset_index()

        # Plus besoin de la colonne 'semaines'
        # Si besoin on peut stocker la liste globale toutes_semaines dans un attribut
        self.toutes_semaines = toutes_semaines

        colonnes_finales = ['code', 'type_code', 'designation', 'prix_unitaire_moyen', 'nb_commandes',
                            'quantites_commandees', 'dates_commandes',
                            'famille', 'fournisseur']
        self.df_articles_groupes = df_grouped[colonnes_finales]


    def _harmoniser_df(self, df, fournisseur):
        df = df.copy()
        df['fournisseur'] = fournisseur.upper()

        rename_dict = {
            'prix_ht_unitaire': 'prix_unitaire_ht',
            'prix_unit_ht': 'prix_unitaire_ht',
            'date_cmd': 'date_commande',
            'dates': 'date_commande',
            'gencode': 'code',
            'date_facture': 'date_commande',
            'reference': 'code'
        }
        df.rename(columns=rename_dict, inplace=True)

        for col in ['code', 'designation', 'quantite', 'prix_unitaire_ht', 'date_commande', 'famille']:
            if col not in df.columns:
                df[col] = None

        if 'date_commande' in df.columns:
            df['date_commande'] = df['date_commande'].apply(lambda x: x.strftime('%Y-%m-%d') if isinstance(x, date) else x)
            df['date_commande'] = pd.to_datetime(df['date_commande'], errors='coerce')

        # 🔄 Dernier prix unitaire par code
        if 'prix_unitaire_ht' in df.columns and 'date_commande' in df.columns:
            df = df.sort_values('date_commande')  # pour garantir un groupby.last() cohérent
            derniers_prix = (
                df.dropna(subset=['code', 'prix_unitaire_ht', 'date_commande'])
                  .groupby('code')
                  .last()['prix_unitaire_ht']
            )
            df['prix_moyen_unitaire_ht'] = df['code'].map(derniers_prix)
        else:
            df['prix_moyen_unitaire_ht'] = None

        if fournisseur == 'APDV':
            df['type_code'] = 'EAN13'
        elif fournisseur in ['SUPERGROUP', 'LBB']:
            df['type_code'] = 'IFLS'
        elif fournisseur == 'CHERITEL':
            df['type_code'] = 'Code indépendant'
            df['code'] = df['designation'].astype('category').cat.codes + 1
        elif fournisseur == 'AUTENTIK':
            df['type_code'] = 'Code Fournisseur'

        colonnes_finales = ['code', 'type_code', 'designation', 'quantite', 'prix_unitaire_ht',
                            'prix_moyen_unitaire_ht', 'date_commande', 'famille', 'fournisseur']
        return df[colonnes_finales]




    def etaler_ventes_cheritel(self):
        """
        Étale chaque commande CHERITEL sur 2 semaines : moitié la semaine de la commande, moitié la semaine suivante.
        """
        print("[CHERITEL] Étaler les ventes sur 2 semaines...")
        ca_articles = []
        quantites_vendues_liste = []
        marge = 1.3

        for _, row in self.df_articles_groupes.iterrows():
            if row['fournisseur'] != 'CHERITEL':
                ca_articles.append([0] * len(self.toutes_semaines))
                quantites_vendues_liste.append([0] * len(self.toutes_semaines))
                continue

            qtes = row['quantites_commandees']
            prix = row['prix_unitaire_moyen']
            ca_article = [0] * len(qtes)
            quantites_vendues = [0] * len(qtes)

            for i, qte in enumerate(qtes):
                if qte > 0:
                    ca = (qte * prix * marge) / 2
                    qte_vendue = qte / 2
                    ca_article[i] += ca
                    quantites_vendues[i] += qte_vendue
                    if i + 1 < len(qtes):
                        ca_article[i + 1] += ca
                        quantites_vendues[i + 1] += qte_vendue

            ca_articles.append(ca_article)
            quantites_vendues_liste.append(quantites_vendues)

        self.df_articles_groupes['ca_article'] = ca_articles
        self.df_articles_groupes['quantites_vendues'] = quantites_vendues_liste



    def etaler_ventes_autentik(self):
        print("[AUTENTIK] Calcul des ventes étalées...")

        marge = 1.3
        df = self.df_articles_groupes.copy()
        df = df[df['fournisseur'] == 'AUTENTIK'].copy()
        df.sort_values(by=['code'], inplace=True)

        # Liste de toutes les semaines utilisées comme base
        toutes_semaines = self.toutes_semaines
        nb_semaines = len(toutes_semaines)

        # Ajout des colonnes si elles n'existent pas encore
        if 'ca_article' not in self.df_articles_groupes.columns:
            self.df_articles_groupes['ca_article'] = None
        if 'quantites_vendues' not in self.df_articles_groupes.columns:
            self.df_articles_groupes['quantites_vendues'] = None

        compteur = 0

        for _, ligne in df.iterrows():
            code = ligne['code']
            if code.startswith('R'):
                continue  # Ne pas traiter les lignes de reprise directement

            qte_commandees = ligne['quantites_commandees']
            prix_moyen = ligne['prix_unitaire_moyen']

            # Cherche la ligne de reprise correspondante
            code_reprise = 'R' + code
            ligne_reprise = df[df['code'] == code_reprise]
            qte_reprises = ligne_reprise.iloc[0]['quantites_commandees'] if not ligne_reprise.empty else [0] * nb_semaines

            # On complète les listes si nécessaire
            if len(qte_commandees) < nb_semaines:
                qte_commandees += [0] * (nb_semaines - len(qte_commandees))
            if len(qte_reprises) < nb_semaines:
                qte_reprises += [0] * (nb_semaines - len(qte_reprises))

            # Calcul des quantités vendues par semaine
            quantites_vendues = [cmd - rep for cmd, rep in zip(qte_commandees, qte_reprises)]

            # Calcul du CA simulé par semaine
            ca = [q * prix_moyen * marge for q in quantites_vendues]

            # Mise à jour dans la DataFrame principale
            idx = self.df_articles_groupes[
                (self.df_articles_groupes['fournisseur'] == 'AUTENTIK') &
                (self.df_articles_groupes['code'] == code)
            ].index

            if len(idx) == 1:
                self.df_articles_groupes.at[idx[0], 'quantites_vendues'] = quantites_vendues
                self.df_articles_groupes.at[idx[0], 'ca_article'] = ca
                compteur += 1
            else:
                print(f"⚠️ Code '{code}' non trouvé ou multiple dans df_articles_groupes")

        print(f"✅ Étalement terminé pour {compteur} articles AUTENTIK.")

    # traiter_factures.py
    import math
    import numpy as np

    def etaler_ventes_apdv_lbb_sg(self,
                                  consommer_derniere_commande: bool = True,
                                  stock_safety_ratio: float = 0.0,
                                  methode_taux: str = "median"):
        """
        APDV/LBB/SUPERGROUP :
        - Étale chaque commande uniformément de sa semaine jusqu'à la veille de la prochaine.
        - Pour la DERNIÈRE commande (pas de suivante), consomme à un "taux de base"
          jusqu'au LUNDI DE LA SEMAINE COURANTE (inclus), de sorte que la semaine actuelle
          ne soit plus à 0. Un stock de sécurité (ratio) peut être conservé.

        consommer_derniere_commande=True  -> active la conso progressive de la dernière.
        stock_safety_ratio=0.0            -> % de la dernière commande gardé en stock (0 = tout dispo).
        methode_taux="median"             -> taux de base = médiane des taux historiques (quantité/semaines).
        """
        import math
        import numpy as np
        import pandas as pd

        print(
            "[APDV/LBB/SUPERGROUP] Ventes étalées: entre commandes (uniforme) + conso dernière incluant la semaine courante.")

        # Sous-ensemble des fournisseurs concernés
        df_src = self.df_articles_groupes[
            self.df_articles_groupes['fournisseur'].isin(['APDV', 'LBB', 'SUPERGROUP'])
        ].copy()

        toutes_semaines = list(self.toutes_semaines)  # grille de lundis
        nb_semaines = len(toutes_semaines)
        if nb_semaines == 0 or df_src.empty:
            print("⚠️ Aucune semaine ou aucun article à traiter.")
            return

        # Indice de la semaine courante (lundi)
        aujourdhui = pd.Timestamp.today().normalize()
        lundi_courant = aujourdhui - pd.to_timedelta(aujourdhui.weekday(), unit='D')
        # plus grand index tel que semaine <= lundi_courant, sinon borne fin
        idx_today = max((i for i, d in enumerate(toutes_semaines) if pd.Timestamp(d) <= lundi_courant),
                        default=nb_semaines - 1)

        marge = 1.3
        maj = 0

        for _, ligne in df_src.iterrows():
            q_cmd = ligne.get('quantites_commandees', [])
            prix = ligne.get('prix_unitaire_moyen', None)
            code = ligne.get('code')

            if prix is None or not isinstance(q_cmd, list):
                continue

            # indices des semaines où commande > 0
            indices_cmd = [i for i, q in enumerate(q_cmd) if q and q > 0]
            if not indices_cmd:
                continue

            q_vendues = [0.0] * nb_semaines
            ca_article = [0.0] * nb_semaines

            # 1) Intervalles fermés : étalement strictement UNIFORME
            for k in range(len(indices_cmd) - 1):
                start = indices_cmd[k]
                end = indices_cmd[k + 1]  # non inclus
                span = end - start
                if span <= 0:
                    continue

                q = float(q_cmd[start] or 0.0)
                weekly = q / span  # uniforme sans arrondi cumulatif

                for off in range(span):
                    idx_w = start + off
                    if 0 <= idx_w < nb_semaines:
                        q_vendues[idx_w] += weekly
                        ca_article[idx_w] += weekly * prix * marge

            # 2) Intervalle ouvert (dernière commande) : consommer jusqu'à la semaine courante (incluse)
            if consommer_derniere_commande and len(indices_cmd) >= 1:
                last = indices_cmd[-1]
                # borne de fin = lundi courant (inclus) mais bornée par la grille
                end_open = min(nb_semaines, idx_today + 1)
                if end_open > last:
                    span_open = end_open - last

                    # Taux de base estimé à partir des intervalles fermés
                    rates = []
                    for k in range(len(indices_cmd) - 1):
                        qk = float(q_cmd[indices_cmd[k]] or 0.0)
                        w = indices_cmd[k + 1] - indices_cmd[k]
                        if qk > 0 and w > 0:
                            rates.append(qk / w)
                    if rates:
                        r_hat = float(np.median(rates)) if methode_taux == "median" else float(np.mean(rates))
                    else:
                        # Fallback : moyenne des commandes positives / 4
                        pos_orders = [float(x) for x in q_cmd if x and x > 0]
                        r_hat = (sum(pos_orders) / max(1, len(pos_orders))) / 4.0 if pos_orders else 0.0

                    q_last = float(q_cmd[last] or 0.0)
                    safety = max(0.0, stock_safety_ratio * q_last)
                    remaining = max(0.0, q_last - safety)

                    if remaining > 0:
                        # Ne pas tout vider : weekly borné par le quota restant sur l'intervalle
                        weekly = min(r_hat if r_hat > 0 else (remaining / span_open), remaining / span_open)
                        for off in range(span_open):
                            idx_w = last + off
                            if idx_w >= nb_semaines or remaining <= 0:
                                break
                            used = min(weekly, remaining)
                            q_vendues[idx_w] += used
                            ca_article[idx_w] += used * prix * marge
                            remaining -= used

            # Injection dans la DF principale (ligne unique attendue)
            idx_df = self.df_articles_groupes[
                (self.df_articles_groupes['fournisseur'] == ligne['fournisseur']) &
                (self.df_articles_groupes['code'] == code)
                ].index

            if len(idx_df) == 1:
                self.df_articles_groupes.at[idx_df[0], 'quantites_vendues'] = q_vendues
                self.df_articles_groupes.at[idx_df[0], 'ca_article'] = ca_article
                maj += 1
            else:
                print(f"⚠️ Code {code} non trouvé de façon unique dans df_articles_groupes.")

        print(f"✅ Étalement terminé pour {maj} articles APDV/LBB/SUPERGROUP (incluant la semaine courante).")

    def detect_base_path(self):
        # Retourne le chemin absolu vers le dossier "data"
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

    def exporter_csv(self, nom_fichier='df_traitee.csv'):
        if not hasattr(self, 'df_articles_groupes'):
            raise AttributeError("La DataFrame 'df_articles_groupes' n'existe pas.")

        dossier_export = os.path.join(self.base_path, 'Fichiers CSV')
        os.makedirs(dossier_export, exist_ok=True)
        chemin_complet = os.path.join(dossier_export, nom_fichier)
        self.df_articles_groupes.to_csv(chemin_complet, index=False)
        print(f"✅ DataFrame exportée avec succès : {chemin_complet}")

