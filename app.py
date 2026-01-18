import pandas as pd
import os
from gestion_factures import GestionFactures
from traiter_factures import TraiterDFs
from visualiser import Visualisation


def main():
    base_folder_id = "1nQ8Sqz2hRD6X5_OdCN2TdiYK3mYSGSCf"

    gestion = GestionFactures(base_folder_id=base_folder_id)
    dfs = gestion.extraire_toutes_les_df()

    total_lignes = sum(len(df) for df in dfs.values())
    print(f"\n📦 Extraction terminée : {total_lignes} lignes récupérées depuis les 5 fournisseurs.")

    traiteur = TraiterDFs(dfs, base_path=gestion.base_path)
    df_global = traiteur.traiter_df()
    traiteur.exporter_csv(nom_fichier='df_traitee.csv')

    print("\n📊 Aperçu de la DataFrame finale :")
    print(df_global.head(10))

    print(df_global[df_global['code'] == '8062268'])
    chemin_csv = os.path.join(gestion.base_path, 'Fichiers CSV', 'df_traitee.csv')
    if os.path.exists(chemin_csv):
        print(f"\n✅ CSV exporté avec succès : {chemin_csv}")
    else:
        print(f"\n❌ Problème : le fichier CSV n’a pas été généré.")

    # Création de la classe Visualisation
    visu = Visualisation()

    """"# Optionnel : filtrage par période (utilisé pour get_kpis uniquement)
    date_debut = pd.to_datetime("2024-01-01")
    date_fin = pd.to_datetime("2024-01-31")
    periode = (date_debut, date_fin)
    visu.appliquer_filtres(periode=periode)"""

    date_debut = pd.to_datetime("2024-01-01")
    date_fin = pd.to_datetime("2024-12-31")
    periode = (date_debut, date_fin)

    visu.appliquer_filtres(fournisseur='SUPERGROUP', periode=periode)
    # 🔎 Récupération des KPI
    print("\n🔢 KPI généraux :")
    print(visu.get_kpis())

    print("\n📈 KPI N-1 :")
    print(visu.get_kpis_n_1())

    print("\n📊 KPI M-1 :")
    print(visu.get_kpis_m_1())

    # 📉 Graphiques
    visu.get_graphs()


if __name__ == "__main__":
    main()
