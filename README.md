## Structure du projet

```text
Application-BI-Carrefour/
│
├── streamlit_app.py               # ✅ Entrypoint Streamlit (UI principale : Extraction / Traitement / Analyse / Prédiction)
├── requirements.txt               # Dépendances Python (Streamlit Cloud)
├── README.md                      # Documentation projet
├── .gitignore                     # Ignore tokens/secrets/cache/local data
│
├── src/
│   ├── gestion_factures.py        # Sync Drive → local runtime, extraction incrémentale par fournisseur
│   ├── traiter_factures.py        # Traitements, étalement des ventes, construction du df global df_traitee.csv
│   ├── visualiser.py              # KPI, filtres, graphes, prédictions, simulation stock & commandes
│   ├── oauth_drive_manager.py     # ✅ OAuth web : login, refresh automatique, client Drive, persistance tokens
│   └── google_drive_manager.py    # (legacy) Service account / ancien manager Drive (non utilisé en prod)
│
├── .streamlit/
│   ├── config.toml                # (optionnel) Config UI Streamlit (theme, layout…)
│   └── secrets.toml               # ❌ Local only (non commité) : BASE_FOLDER_ID + OAuth client (Cloud = Settings/Secrets)
│
├── .oauth_tokens/                 # ❌ Local/runtime only (non commité) : tokens OAuth par user (peut être éphémère en cloud)
│
└── data/                          # ❌ Cache local/runtime (non commité)
    ├── temp_pdfs/                 # PDFs téléchargés depuis Drive pour extraction (runtime)
    │   ├── APDV/
    │   ├── LBB/
    │   ├── SUPERGROUP/
    │   ├── CHERITEL/
    │   └── AUTENTIK/
    │
    └── Fichiers CSV/              # Cache local des CSV (source de vérité = Drive)
        ├── nt_apdv.csv
        ├── nt_lbb.csv
        ├── nt_supergroup.csv
        ├── nt_cheritel.csv
        ├── nt_autentik.csv
        └── df_traitee.csv         # CSV global utilisé par l’analyse (reconstruit sans reparser tous les PDF)
'''
