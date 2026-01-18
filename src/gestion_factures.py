import os
import re
import pandas as pd
import pdfplumber
from datetime import datetime
from src.google_drive_manager import GoogleDriveManager
#from googleapiclient.errors import HttpError

class GestionFactures:
    def __init__(self, credentials_path='credentials.json', base_folder_id=None, drive_manager=None):
        self.base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
        if drive_manager is not None:
            # ✅ Utilise l’instance OAuth passée par l’app
            self.drive = drive_manager
        else:
            # fallback SA (si jamais tu veux garder l’ancien mode)
            from src.google_drive_manager import GoogleDriveManager
            self.drive = GoogleDriveManager(credentials_path, base_folder_id)
        self.extracteurs = {
            'APDV': self.extraire_apdv,
            'CHERITEL': self.extraire_cheritel,
            'LBB': self.extraire_lbb,
            'SUPERGROUP': self.extraire_supergroup,
            'AUTENTIK': self.extraire_autentik
        }

        self.dossier_temp_drive_ids = {
            'SUPERGROUP': '1L_B7BHiId8kcX9wYKU663-RpX9zc_YQS',
            'LBB': '1POGMNEmOyi9Txm5oVUa9TCPs7ncAXBBh',
            'APDV': '1ArFPlEnEUsz6hafCOe4xpFocpS_0Y0Vi',
            'AUTENTIK': '1ZvXDGHxj7CFBVy0igxnErs9bYFEyhjwz',
            'CHERITEL': '1Olb08TXS_2xrkEscENpJjMKS7a3o0Lzo'
        }

    def synchroniser_factures(self, fournisseur):
        """
        Synchronise les factures d'un fournisseur depuis Google Drive vers le stockage local.

        Ordre de recherche des dossiers SOURCE sur Drive (sous BASE_FOLDER_ID) :
          1) "Factures {FOURNISSEUR}"
          2) "temp_pdfs/{FOURNISSEUR}"
          3) "temps_pdfs/{FOURNISSEUR}" (tolérance faute de frappe)
        """
        import os
        try:
            from googleapiclient.errors import HttpError
        except Exception:
            HttpError = Exception

        F = str(fournisseur).upper()

        # Dossier local cible
        local_dir = os.path.join(self.base_path, "temp_pdfs", F)
        os.makedirs(local_dir, exist_ok=True)

        # --- Résolution du dossier source Drive ---------------------------------
        tried = []
        folder_id = None

        # 1) Factures {F}
        fid = self.drive.find_folder_id(f"Factures {F}")
        tried.append(f"Factures {F}")
        if fid:
            folder_id = fid

        # 2) temp_pdfs/{F}
        if not folder_id:
            temp_root = self.drive.find_folder_id("temp_pdfs")
            tried.append("temp_pdfs")
            if temp_root:
                fid = self.drive.find_folder_id(F, parent_id=temp_root)
                tried.append("temp_pdfs/" + F)
                if fid:
                    folder_id = fid

        # 3) temps_pdfs/{F} (variante)
        if not folder_id:
            temp_root_alt = self.drive.find_folder_id("temps_pdfs")
            tried.append("temps_pdfs")
            if temp_root_alt:
                fid = self.drive.find_folder_id(F, parent_id=temp_root_alt)
                tried.append("temps_pdfs/" + F)
                if fid:
                    folder_id = fid

        if not folder_id:
            print(f"❌ Dossier source introuvable pour {F}. Cherché : {', '.join(tried)}")
            return

        # --- Listing & téléchargement -------------------------------------------
        try:
            fichiers_drive = self.drive.list_pdfs_in_folder(folder_id) or []
        except HttpError as e:
            print(f"❌ Erreur listage Drive : {e}")
            return

        noms_locaux = set(os.listdir(local_dir))
        # ... même code qu’aujourd’hui pour la détection ...

        telecharges, deja_present = 0, 0
        for fichier in fichiers_drive:
            nom = fichier.get("name")
            fid = fichier.get("id")
            if not nom or not fid:
                continue

            if nom in noms_locaux:
                deja_present += 1
                continue

            print(f"⬇️ Téléchargement : {nom}")
            dest_path = os.path.join(local_dir, nom)
            try:
                self.drive.download_file(fid, dest_path)
                telecharges += 1
            except HttpError as e:
                print(f"❌ Erreur téléchargement {nom} : {e}")

        print(f"✅ Sync {F} — téléchargés: {telecharges}, déjà présents: {deja_present}.")
        return telecharges  # ➜ ✅ on renvoie le nombre de nouveaux PDF

    def extraire_toutes_les_df(self):
        return {fournisseur: extracteur() for fournisseur, extracteur in self.extracteurs.items()}

    def convertir_vers_float(self, valeur_str):
        if valeur_str is None:
            return 0.0
        return float(valeur_str.replace('.', '').replace(',', '.'))

    def extraire_supergroup(self):
        self.synchroniser_factures('SUPERGROUP')
        dossier_pdf = os.path.join(self.base_path, 'temp_pdfs', 'SUPERGROUP')
        dossier_csv = os.path.join(self.base_path, 'Fichiers CSV')
        os.makedirs(dossier_csv, exist_ok=True)
        chemin_csv_sortie = os.path.join(dossier_csv, 'nt_supergroup.csv')

        if os.path.exists(chemin_csv_sortie):
            df_existantes = pd.read_csv(chemin_csv_sortie)
            factures_deja_traitees = set(df_existantes['fichier'].unique()) if 'fichier' in df_existantes.columns else set()
        else:
            df_existantes = pd.DataFrame()
            factures_deja_traitees = set()

        factures_disponibles = [f for f in os.listdir(dossier_pdf) if f.lower().endswith('.pdf')]
        nouvelles_factures = [f for f in factures_disponibles if f not in factures_deja_traitees]

        familles_possibles = ["CONFISERIE", "BOISSONS", "EPICERIE", "PATISSERIE",
                              "GUM", "PIPIER", "BISCUITERIE", "BISCUITERIE SALEE", "CHOCOLAT"]

        toutes_lignes = []

        print(f"[SUPERGROUP] 📂 Factures disponibles : {len(factures_disponibles)}")
        print(f"[SUPERGROUP] 🆕 Nouvelles factures à traiter : {len(nouvelles_factures)}")

        for fichier in nouvelles_factures:
            chemin_pdf = os.path.join(dossier_pdf, fichier)
            articles = []
            articles_sans_famille = []

            try:
                date_match = re.search(r'_(\d{8})_', fichier)
                date_facture = datetime.strptime(date_match.group(1), '%Y%m%d').strftime('%Y-%m-%d') if date_match else None
            except Exception:
                date_facture = None

            try:
                with pdfplumber.open(chemin_pdf) as pdf:
                    for page in pdf.pages:
                        texte = page.extract_text()
                        if not texte:
                            continue
                        lignes = texte.split('\n')

                        for ligne in lignes:
                            l = ligne.strip()
                            if re.match(r'^>?(\d{6,13})\s+', l):
                                blocs = l.split()
                                try:
                                    code = re.findall(r'\d{6,13}', blocs[0])[0]
                                    quantite = int(blocs[-5])
                                    cond = int(blocs[-4])
                                    prix_u_ht = self.convertir_vers_float(blocs[-3])
                                    montant_ht = self.convertir_vers_float(blocs[-2])
                                    tva = self.convertir_vers_float(blocs[-1])
                                    designation = " ".join(blocs[1:-5])
                                except Exception:
                                    continue

                                articles_sans_famille.append([
                                    code, designation, quantite, cond,
                                    prix_u_ht, montant_ht, tva, None,
                                    "SUPERGROUP", date_facture, fichier
                                ])
                            else:
                                for f in familles_possibles:
                                    if f in l.upper():
                                        famille_detectee = f.capitalize()
                                        for a in articles_sans_famille:
                                            a[7] = famille_detectee
                                            articles.append(a)
                                        articles_sans_famille = []
                                        break

                for a in articles_sans_famille:
                    a[7] = "Inconnue"
                    articles.append(a)

                for article in articles:
                    article[2] = article[2] * article[3]

                if articles:
                    df_facture = pd.DataFrame(articles, columns=[
                        "code", "designation", "quantite", "conditionnement",
                        "prix_unitaire_ht", "montant_ht", "tva", "famille",
                        "fournisseur", "date_commande", "fichier"
                    ])
                    toutes_lignes.append(df_facture)
                    print(f"✅ {fichier} → OK ({len(df_facture)} lignes)")
                else:
                    print(f"⚠️ {fichier} → Aucune donnée extraite.")
            except Exception as e:
                print(f"❌ {fichier} → Erreur : {e}")

        df_supergroup = pd.concat(toutes_lignes, ignore_index=True) if toutes_lignes else pd.DataFrame()

        if not df_existantes.empty:
            df_concat = pd.concat([df_existantes, df_supergroup], ignore_index=True)
        else:
            df_concat = df_supergroup

        factures_restantes = set(factures_disponibles)
        df_concat = df_concat[df_concat['fichier'].isin(factures_restantes)]

        df_concat.to_csv(chemin_csv_sortie, index=False)
        print(f"✅ Extraction terminée pour SUPERGROUP. Données sauvegardées dans : {chemin_csv_sortie}")
        print(f"Nombre total de lignes : {len(df_concat)}")

        return df_concat




#----------------------------------CHERITEL-------------------------------------
    def extraire_infos_cheritel(self, texte):
        lignes = texte.split('\n')
        en_lecture = False
        date_facture = None
        articles = []

        for ligne in lignes:
            if "BL N°" in ligne and "DU" in ligne:
                try:
                    date_str = ligne.strip().split()[-1]
                    date_facture = datetime.strptime(date_str, "%d/%m/%Y").strftime('%Y-%m-%d')
                except Exception as e:
                    print(f"Erreur de parsing de date dans la ligne : {ligne}\n{e}")
                break

        for ligne in lignes:
            if "BL N°" in ligne:
                en_lecture = True
                continue
            if "TOTAL BL" in ligne:
                break
            if not en_lecture:
                continue

            match = re.search(
                r"^(?P<designation>.+?)\s+(?P<colis>[\d,]+)\s+(?P<poids>[\d,]+)\s+(?P<unite>[A-Z]{1,3})\s+(?P<pu_ht>[\d,]+)\s+(?P<montant>[\d,]+)",
                ligne.strip()
            )
            if match:
                d = match.groupdict()
                try:
                    poids = float(d["poids"].replace(",", "."))
                    pu_ht = float(d["pu_ht"].replace(",", "."))
                    articles.append({
                        "designation": d["designation"].strip(),
                        "quantite": poids,
                        "prix_unitaire_ht": pu_ht,
                        "date_commande": date_facture,
                        "famille": "Fruits et légumes"
                    })
                except Exception as e:
                    print(f"Erreur de conversion sur la ligne : {ligne}\n{e}")
                    continue

        return pd.DataFrame(articles)

    def extraire_cheritel(self):
        self.synchroniser_factures('CHERITEL')
        dossier_pdf = os.path.join(self.base_path, 'temp_pdfs', 'CHERITEL')
        dossier_csv = os.path.join(self.base_path, 'Fichiers CSV')
        os.makedirs(dossier_csv, exist_ok=True)

        chemin_csv_sortie = os.path.join(dossier_csv, 'nt_cheritel.csv')
        chemin_vides = os.path.join(dossier_csv, 'factures_vides_cheritel.csv')

        # Charger les factures déjà traitées
        if os.path.exists(chemin_csv_sortie):
            df_existantes = pd.read_csv(chemin_csv_sortie)
            factures_deja_traitees = set(df_existantes['fichier'].unique())
        else:
            df_existantes = pd.DataFrame()
            factures_deja_traitees = set()

        # Charger les factures déjà identifiées comme vides
        factures_vides = set()
        if os.path.exists(chemin_vides):
            factures_vides = set(pd.read_csv(chemin_vides)['fichier'].tolist())

        factures_disponibles = [f for f in os.listdir(dossier_pdf) if f.lower().endswith('.pdf')]
        nouvelles_factures = [f for f in factures_disponibles if
                              f not in factures_deja_traitees and f not in factures_vides]

        print(f"[CHERITEL] 📂 Factures disponibles : {len(factures_disponibles)}")
        print(f"[CHERITEL] 🆕 Nouvelles factures à traiter : {len(nouvelles_factures)}")

        toutes_lignes = []
        nouvelles_vides = []

        for fichier in nouvelles_factures:
            chemin = os.path.join(dossier_pdf, fichier)
            try:
                with pdfplumber.open(chemin) as pdf:
                    texte = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

                df_facture = self.extraire_infos_cheritel(texte)
                if not df_facture.empty:
                    df_facture['famille'] = "Fruits et légumes"
                    df_facture['fichier'] = fichier
                    df_facture['fournisseur'] = 'CHERITEL'
                    toutes_lignes.append(df_facture)
                    print(f"✅ {fichier} → OK ({len(df_facture)} lignes)")
                else:
                    print(f"⚠️ {fichier} → Aucune donnée extraite.")
                    nouvelles_vides.append(fichier)
            except Exception as e:
                print(f"❌ {fichier} → Erreur : {e}")
                nouvelles_vides.append(fichier)

        # Sauvegarder les nouvelles factures vides détectées
        if nouvelles_vides:
            df_vides = pd.DataFrame({'fichier': nouvelles_vides})
            if os.path.exists(chemin_vides):
                df_vides.to_csv(chemin_vides, mode='a', index=False, header=False)
            else:
                df_vides.to_csv(chemin_vides, index=False)
            print(f"🧾 {len(nouvelles_vides)} fichier(s) ajouté(s) à factures_vides_cheritel.csv")

        df_cheritel = pd.concat(toutes_lignes, ignore_index=True) if toutes_lignes else pd.DataFrame()

        if not df_existantes.empty:
            df_concat = pd.concat([df_existantes, df_cheritel], ignore_index=True)
        else:
            df_concat = df_cheritel

        factures_restantes = set(factures_disponibles)
        df_concat = df_concat[df_concat['fichier'].isin(factures_restantes)]

        df_concat.to_csv(chemin_csv_sortie, index=False)
        print(f"✅ Extraction terminée pour CHERITEL. Données sauvegardées dans : {chemin_csv_sortie}")
        print(f"Nombre total de lignes : {len(df_concat)}")

        return df_concat

    def get_pattern_ligne_lbb(self):
        return re.compile(
            r'^(?P<code>\d{5,7})\s+'
            r'(?P<designation>.+?)\s+'
            r'(?P<gencod>\d{12,13})\s+'
            r'(?P<cond>\d+\s\w{2,4})\s+'
            r'(?P<quantite>\d+)\s+'
            r'(?P<unite>\w{2,4})\s+'
            r'(?P<prix_unitaire_ht>\d+[,.]\d+)\s+'
            r'(?P<montant_ht>\d+[,.]\d+)\s+'
            r'(?P<droits>[\d,.]+)'
        )

    def nettoyer_ligne_lbb(self, ligne):
        ligne = ligne.strip()
        ligne = re.sub(r'\s+', ' ', ligne)
        return ligne

    def extraire_lignes_lbb(self, chemin_pdf):
        lignes = []
        with pdfplumber.open(chemin_pdf) as pdf:
            for page in pdf.pages:
                texte = page.extract_text()
                if texte:
                    lignes.extend(texte.split('\n'))
        return lignes

    def extraire_date_lbb(self, lignes):
        date_pattern = re.compile(r'(\d{2}/\d{2}/\d{4})')
        for ligne in lignes:
            matches = date_pattern.findall(ligne)
            if matches:
                try:
                    return datetime.strptime(matches[0], "%d/%m/%Y").strftime('%Y-%m-%d')
                except:
                    continue
        return None

    def parser_lignes_lbb(self, lignes, date_facture, fichier):
        pattern = self.get_pattern_ligne_lbb()
        data = []
        for ligne in lignes:
            ligne_nettoye = self.nettoyer_ligne_lbb(ligne)
            match = pattern.match(ligne_nettoye)
            if match:
                ligne_dict = match.groupdict()
                ligne_dict['code'] = str(ligne_dict['code']).zfill(6)
                ligne_dict['fichier'] = fichier
                ligne_dict['date_commande'] = date_facture
                ligne_dict['famille'] = "Boissons"
                ligne_dict['fournisseur'] = 'LBB'
                data.append(ligne_dict)
        return data

    def transformer_donnees_lbb(self, data):
        df = pd.DataFrame(data)
        if not df.empty:
            try:
                df['prix_unitaire_ht'] = df['prix_unitaire_ht'].str.replace(',', '.').astype(float)
                df['montant_ht'] = df['montant_ht'].str.replace(',', '.').astype(float)
                df['droits'] = df['droits'].str.replace(',', '.').astype(float)
                df['quantite'] = df['quantite'].astype(int)
            except Exception as e:
                print("Erreur de conversion numérique :", e)
            df['designation'] = df['designation'].str.strip().str.replace(r'\(cid:176\)', '°', regex=True)
            df['cond'] = df['cond'].str.strip()
            df = df.drop(columns=['cond', 'droits', 'unite', 'montant_ht'])
        return df

    def extraire_lbb(self):
        print("[LBB] Extraction incrémentale en cours...")

        self.synchroniser_factures('LBB')
        dossier_factures = os.path.join(self.base_path, 'temp_pdfs', 'LBB')
        dossier_csv = os.path.join(self.base_path, 'Fichiers CSV')
        os.makedirs(dossier_csv, exist_ok=True)

        chemin_csv_sortie = os.path.join(dossier_csv, 'nt_lbb.csv')
        chemin_vides = os.path.join(dossier_csv, 'factures_vides_lbb.csv')

        if os.path.exists(chemin_csv_sortie):
            df_existantes = pd.read_csv(chemin_csv_sortie)
            factures_deja_traitees = set(df_existantes['fichier'].unique())
        else:
            df_existantes = pd.DataFrame()
            factures_deja_traitees = set()

        factures_vides = set()
        if os.path.exists(chemin_vides):
            factures_vides = set(pd.read_csv(chemin_vides)['fichier'].tolist())

        factures_disponibles = [f for f in os.listdir(dossier_factures) if f.lower().endswith('.pdf')]
        nouvelles_factures = [f for f in factures_disponibles if
                              f not in factures_deja_traitees and f not in factures_vides]

        print(f"[LBB] 📂 Factures disponibles : {len(factures_disponibles)}")
        print(f"[LBB] 🆕 Nouvelles factures à traiter : {len(nouvelles_factures)}")

        toutes_lignes = []
        nouvelles_vides = []

        for fichier in nouvelles_factures:
            chemin = os.path.join(dossier_factures, fichier)
            try:
                lignes = self.extraire_lignes_lbb(chemin)
                date_facture = self.extraire_date_lbb(lignes)
                data = self.parser_lignes_lbb(lignes, date_facture, fichier)
                df_lignes = self.transformer_donnees_lbb(data)

                if df_lignes.empty:
                    print(f"⚠️ {fichier} → Aucune donnée extraite.")
                    nouvelles_vides.append(fichier)
                else:
                    toutes_lignes.append(df_lignes)
                    print(f"✅ {fichier} → OK ({len(df_lignes)} lignes)")
            except Exception as e:
                print(f"❌ Erreur sur le fichier {fichier} : {e}")
                nouvelles_vides.append(fichier)

        if nouvelles_vides:
            df_vides = pd.DataFrame({'fichier': nouvelles_vides})
            if os.path.exists(chemin_vides):
                df_vides.to_csv(chemin_vides, mode='a', index=False, header=False)
            else:
                df_vides.to_csv(chemin_vides, index=False)
            print(f"🧾 {len(nouvelles_vides)} fichier(s) ajouté(s) à factures_vides_lbb.csv")

        df_lbb = pd.concat(toutes_lignes, ignore_index=True) if toutes_lignes else pd.DataFrame()

        if not df_existantes.empty:
            df_concat = pd.concat([df_existantes, df_lbb], ignore_index=True)
        else:
            df_concat = df_lbb

        factures_restantes = set(factures_disponibles)
        df_concat = df_concat[df_concat['fichier'].isin(factures_restantes)]

        df_concat.to_csv(chemin_csv_sortie, index=False)
        print(f"✅ Extraction terminée pour LBB. Données sauvegardées dans : {chemin_csv_sortie}")
        print(f"Nombre total de lignes : {len(df_concat)}")

        return df_concat

    #-----------------------------------APDV----------------------------------------

    def extraire_apdv(self):
        print("[APDV] Extraction incrémentale en cours...")

        self.synchroniser_factures('APDV')
        dossier_factures = os.path.join(self.base_path, 'temp_pdfs', 'APDV')
        dossier_csv = os.path.join(self.base_path, 'Fichiers CSV')
        os.makedirs(dossier_csv, exist_ok=True)

        fichier_csv = os.path.join(dossier_csv, 'nt_apdv.csv')
        fichier_vides = os.path.join(dossier_csv, 'factures_vides_apdv.csv')

        if os.path.exists(fichier_csv):
            df_existantes = pd.read_csv(fichier_csv)
            factures_traitees = set(df_existantes['fichier'].unique())
        else:
            df_existantes = pd.DataFrame()
            factures_traitees = set()

        if os.path.exists(fichier_vides):
            factures_vides = set(pd.read_csv(fichier_vides)['fichier'].tolist())
        else:
            factures_vides = set()

        factures_disponibles = [f for f in os.listdir(dossier_factures) if f.lower().endswith('.pdf')]
        nouvelles_factures = [f for f in factures_disponibles if f not in factures_traitees and f not in factures_vides]

        if not nouvelles_factures:
            print("✅ Aucune nouvelle facture à traiter.")
            factures_restantes = set(factures_disponibles)
            df_existantes = df_existantes[df_existantes['fichier'].isin(factures_restantes)]
            df_existantes.to_csv(fichier_csv, index=False)
            return df_existantes

        print(f"🔍 Nouvelles factures détectées : {len(nouvelles_factures)}")

        toutes_nouvelles_lignes = []
        nouvelles_vides = []

        for i, nom_fichier in enumerate(nouvelles_factures, 1):
            chemin_pdf = os.path.join(dossier_factures, nom_fichier)
            print(f"📄 Traitement de : {nom_fichier} ({i}/{len(nouvelles_factures)})")

            try:
                texte_complet = ""
                with pdfplumber.open(chemin_pdf) as pdf:
                    for page in pdf.pages:
                        texte = page.extract_text()
                        if texte:
                            texte_complet += texte + "\n"
                        else:
                            print(f"[Page sans texte détectée dans {nom_fichier}]")

                texte_clean = re.sub(r'-{5,} PAGE \d+ -{5,}', '', texte_complet)
                lignes = texte_clean.splitlines()

                # Repérer les dates des commandes
                dates_positions = []
                for idx, ligne in enumerate(lignes):
                    match_date = re.search(r'Commande n° \d+ du (\d{2}/\d{2}/\d{4})', ligne)
                    if match_date:
                        try:
                            date_formatted = datetime.strptime(match_date.group(1), "%d/%m/%Y").strftime('%Y-%m-%d')
                            dates_positions.append((idx, date_formatted))
                        except:
                            continue

                # Extraire les lignes produits
                lignes_parsees = 0
                for i_ligne, ligne in enumerate(lignes):
                    match = re.match(r'^(.*)Bouteille\s+0\.75\s+L\s+(\d+,\d{2})\s+€\s+(\d+)\s+([\d,]+)', ligne)
                    if match:
                        designation = match.group(1).strip()
                        prix_unitaire = float(match.group(2).replace(',', '.'))
                        quantite = int(match.group(3))
                        total_ht = float(match.group(4).replace(',', '.'))

                        gencode = None
                        for j in range(i_ligne + 1, i_ligne + 5):
                            if j < len(lignes):
                                gencode_match = re.search(r'Gencode.*?:\s*(\d+)', lignes[j])
                                if gencode_match:
                                    gencode = gencode_match.group(1)
                                    break

                        # Date commande la plus proche
                        date_commande = None
                        for pos, date in reversed(dates_positions):
                            if pos < i_ligne:
                                date_commande = date
                                break
                        if date_commande is None and dates_positions:
                            date_commande = dates_positions[0][1]

                        sous_famille = designation.split()[0] if designation else None

                        toutes_nouvelles_lignes.append({
                            'fichier': nom_fichier,
                            'date_commande': date_commande,
                            'designation': designation,
                            'volume': '0.75 L',
                            'prix_unitaire_ht': prix_unitaire,
                            'quantite': quantite,
                            'total_ht': total_ht,
                            'code': gencode,
                            'famille': 'Boissons',
                            'sous_famille': sous_famille,
                            'fournisseur': 'APDV'
                        })
                        lignes_parsees += 1

                if lignes_parsees == 0:
                    print(f"⚠️ {nom_fichier} → Aucune ligne produit détectée.")
                    nouvelles_vides.append(nom_fichier)
            except Exception as e:
                print(f"❌ Erreur sur le fichier {nom_fichier} : {e}")
                nouvelles_vides.append(nom_fichier)

        if nouvelles_vides:
            df_vides = pd.DataFrame({'fichier': nouvelles_vides})
            if os.path.exists(fichier_vides):
                df_vides.to_csv(fichier_vides, mode='a', index=False, header=False)
            else:
                df_vides.to_csv(fichier_vides, index=False)
            print(f"🧾 {len(nouvelles_vides)} fichier(s) ajouté(s) à factures_vides_apdv.csv")

        df_nouvelles = pd.DataFrame(toutes_nouvelles_lignes)
        df_mis_a_jour = pd.concat([df_existantes, df_nouvelles], ignore_index=True)

        factures_restantes = set(factures_disponibles)
        df_mis_a_jour = df_mis_a_jour[df_mis_a_jour['fichier'].isin(factures_restantes)]

        df_mis_a_jour.to_csv(fichier_csv, index=False)
        print(f"✅ {len(nouvelles_factures)} facture(s) ajoutée(s).")
        return df_mis_a_jour

    def nettoyer_texte_autentik(self, text):
        text = re.sub(r'(BL N° : \d+ du)(\d{2}/\d{2}/\d{4})', r'\1 \2', text)
        corrections = {
            r'\bcr pes\b': 'crêpes',
            r'\bcr pes Authentique\b': 'crêpes Authentique',
            r'\bg teaux\b': 'gâteaux',
            r'\bpaquet \(s\)': 'paquet(s)',
        }
        for motif, remplacement in corrections.items():
            text = re.sub(motif, remplacement, text)
        lignes = text.split('\n')
        lignes_nettoyees = [ligne for ligne in lignes if ligne.strip() and ligne.strip().lower() != 'g']
        return '\n'.join(lignes_nettoyees)

    def extraire_articles_autentik(self, texte):
        lignes = texte.split('\n')
        articles = []
        capture = False
        for ligne in lignes:
            ligne = ligne.strip()
            if "Référence Désignation Quantité P.U. HT Montant HT" in ligne:
                capture = True
                continue
            if capture and (
                    ligne.startswith(("Total", "Taux", "Echéance", "BNP", "SARL", "A reporter", "REPORT")) or not ligne
            ):
                break
            if capture:
                articles.append(ligne)
        return articles

    def parser_lignes_articles_autentik(self, articles):
        lignes_parsees = []
        date_bl = None

        for ligne in articles:
            # Détecter les lignes de type "BL N° : ... du dd/mm/yyyy"
            match_bl = re.match(r'BL N° ?: \d+ du (\d{2}/\d{2}/\d{4})', ligne)
            if match_bl:
                date_bl = match_bl.group(1)
                continue

            parts = ligne.split()
            if len(parts) < 5:
                continue

            try:
                reference = parts[0]
                quantite = float(parts[-4].replace(',', '.'))
                prix_unitaire_ht = float(parts[-3].replace(',', '.'))
                designation = ' '.join(parts[1:-4])
            except ValueError:
                continue

            lignes_parsees.append({
                'code': reference,
                'designation': designation,
                'quantite': quantite,
                'prix_unitaire_ht': prix_unitaire_ht,
                'date_commande': date_bl,
                'famille': 'Epicerie',
                'fournisseur': 'AUTENTIK'
            })

        return lignes_parsees

    def extraire_autentik(self):
        print("[AUTENTIK] Extraction incrémentale en cours...")

        self.synchroniser_factures('AUTENTIK')
        dossier_factures = os.path.join(self.base_path, 'temp_pdfs', 'AUTENTIK')
        dossier_csv = os.path.join(self.base_path, 'Fichiers CSV')
        os.makedirs(dossier_csv, exist_ok=True)

        chemin_csv_sortie = os.path.join(dossier_csv, 'nt_autentik.csv')
        chemin_vides = os.path.join(dossier_csv, 'factures_vides_autentik.csv')

        # Lecture des factures déjà traitées
        if os.path.exists(chemin_csv_sortie):
            try:
                df_existantes = pd.read_csv(chemin_csv_sortie)
                factures_traitees = set(
                    df_existantes['fichier'].unique()) if 'fichier' in df_existantes.columns else set()
            except Exception as e:
                print(f"❌ Erreur lecture CSV existant : {e}")
                df_existantes = pd.DataFrame()
                factures_traitees = set()
        else:
            df_existantes = pd.DataFrame()
            factures_traitees = set()

        if os.path.exists(chemin_vides):
            factures_vides = set(pd.read_csv(chemin_vides)['fichier'].tolist())
        else:
            factures_vides = set()

        factures_disponibles = [f for f in os.listdir(dossier_factures) if f.lower().endswith('.pdf')]
        nouvelles_factures = [f for f in factures_disponibles if f not in factures_traitees and f not in factures_vides]
        factures_restantes = set(factures_disponibles)

        if not nouvelles_factures:
            print("✅ Aucune nouvelle facture à traiter.")
            if not df_existantes.empty and 'fichier' in df_existantes.columns:
                df_mis_a_jour = df_existantes[df_existantes['fichier'].isin(factures_restantes)]
                if len(df_mis_a_jour) != len(df_existantes):
                    print("🧹 Des factures ont été supprimées, mise à jour du CSV.")
                    df_mis_a_jour.to_csv(chemin_csv_sortie, index=False)
                return df_mis_a_jour
            return df_existantes

        print(f"🔍 Nouvelles factures détectées : {len(nouvelles_factures)}")

        toutes_lignes = []
        nouvelles_vides = []

        for i, nom_fichier in enumerate(nouvelles_factures, 1):
            chemin_pdf = os.path.join(dossier_factures, nom_fichier)
            print(f"📄 Traitement de : {nom_fichier} ({i}/{len(nouvelles_factures)})")

            try:
                with pdfplumber.open(chemin_pdf) as pdf:
                    texte = '\n'.join(page.extract_text() for page in pdf.pages if page.extract_text())

                texte_nettoye = self.nettoyer_texte_autentik(texte)
                articles = self.extraire_articles_autentik(texte_nettoye)
                lignes_parsees = self.parser_lignes_articles_autentik(articles)

                if not lignes_parsees:
                    print(f"⚠️ {nom_fichier} → Aucun article détecté.")
                    nouvelles_vides.append(nom_fichier)
                    continue

                for ligne in lignes_parsees:
                    ligne["fichier"] = nom_fichier
                    try:
                        ligne['date_commande'] = datetime.strptime(ligne['date_commande'], "%d/%m/%Y").strftime(
                            '%Y-%m-%d') if ligne['date_commande'] else None
                    except:
                        ligne['date_commande'] = None

                toutes_lignes.extend(lignes_parsees)
                print(f"✅ {nom_fichier} → OK ({len(lignes_parsees)} lignes)")

            except Exception as e:
                print(f"❌ Erreur sur {nom_fichier} : {e}")
                nouvelles_vides.append(nom_fichier)

        # Enregistrement des fichiers vides détectés
        if nouvelles_vides:
            df_vides = pd.DataFrame({'fichier': nouvelles_vides})
            if os.path.exists(chemin_vides):
                df_vides.to_csv(chemin_vides, mode='a', index=False, header=False)
            else:
                df_vides.to_csv(chemin_vides, index=False)
            print(f"🧾 {len(nouvelles_vides)} fichier(s) ajouté(s) à factures_vides_autentik.csv")

        df_nouvelles = pd.DataFrame(toutes_lignes)
        df_mis_a_jour = pd.concat([df_existantes, df_nouvelles], ignore_index=True)
        df_mis_a_jour = df_mis_a_jour[df_mis_a_jour['fichier'].isin(factures_restantes)]

        df_mis_a_jour.to_csv(chemin_csv_sortie, index=False)
        print(f"✅ Extraction terminée pour AUTENTIK → {len(df_nouvelles)} facture(s) ajoutée(s)")
        return df_mis_a_jour


if __name__ == "__main__":
    gf = GestionFactures()
    dfs = gf.extraire_toutes_les_df()
    print({k: len(v) for k, v in dfs.items()})
