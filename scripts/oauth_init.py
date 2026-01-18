# scripts/oauth_init.py
import os, sys
from pathlib import Path

# --- Localisation projet ---
BASE_DIR = Path(__file__).resolve().parents[1]  # -> racine du repo
SRC_DIR = BASE_DIR / "src"
sys.path.append(str(SRC_DIR))

from oauth_drive_manager import OAuthDriveManager, FileTokenStore, DEFAULT_SCOPES
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# ⚠️ À RENSEIGNER
CLIENT_SECRETS = BASE_DIR / "client_secret_desktop.json"   # le JSON téléchargé
USER_EMAIL = "owen.devlogiciel@gmail.com"                # ton vrai email Google
BASE_FOLDER_ID = "11tAEJqUKrgC40l-pmJJGUsymjQ-T1TWY"              # optionnel pour le test

# --- Normalisation clé utilisateur + emplacement des tokens ---
USER_KEY = USER_EMAIL.strip().lower()
TOKENS_DIR = BASE_DIR / ".oauth_tokens"
TOKENS_DIR.mkdir(exist_ok=True)
expected_token_path = TOKENS_DIR / f"token_{USER_KEY}.json"

print("== Infos ==")
print("BASE_DIR:", BASE_DIR)
print("TOKENS_DIR:", TOKENS_DIR)
print("Token attendu:", expected_token_path)

# --- Option A : flow Desktop avec serveur local (recommandé) ---
try:
    mgr = OAuthDriveManager.from_desktop_flow(
        client_secrets_file=str(CLIENT_SECRETS),
        user_key=USER_KEY,
        token_store=FileTokenStore(str(TOKENS_DIR)),
        base_folder_id=BASE_FOLDER_ID,
    )
    print("✅ Auth OK via serveur local.")
except Exception as e:
    print("⚠️ Échec du flow local, on tente le flow console :", e)
    # --- Option B : flow console (copier-coller du code dans le terminal) ---
    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS), scopes=DEFAULT_SCOPES)
    creds = flow.run_console(prompt="consent")
    # Persistance manuelle
    mgr = OAuthDriveManager(
        client_config=str(CLIENT_SECRETS),
        token_store=FileTokenStore(str(TOKENS_DIR)),
        user_key=USER_KEY,
        base_folder_id=BASE_FOLDER_ID,
    )
    mgr.creds = creds
    # forcer le refresh pour obtenir refresh_token si besoin
    if mgr.creds and mgr.creds.expired and mgr.creds.refresh_token:
        mgr.creds.refresh(Request())
    # écrit le fichier token_<clé>.json
    # (la méthode _persist_tokens est interne ; on déclenche via l’accès au service)
    from googleapiclient.discovery import build
    mgr.service = build("drive", "v3", credentials=mgr.creds)
    # persist explicitement :
    data = {
        "token": mgr.creds.token,
        "refresh_token": mgr.creds.refresh_token,
        "token_uri": mgr.creds.token_uri,
        "client_id": mgr.creds.client_id,
        "client_secret": mgr.creds.client_secret,
        "scopes": mgr.creds.scopes,
    }
    FileTokenStore(str(TOKENS_DIR)).save(USER_KEY, data)
    print("✅ Auth OK via flow console.")

# --- Vérifications & petit test ---
if expected_token_path.exists():
    print("🎯 Token présent :", expected_token_path)
else:
    print("❌ Token introuvable : ", expected_token_path)

try:
    # Test léger : récupérer driveId du dossier racine (si fourni)
    if BASE_FOLDER_ID:
        did = mgr.get_drive_id(BASE_FOLDER_ID)
        print("driveId (None = Mon Drive | valeur = Drive partagé) :", did)
except Exception as e:
    print("ℹ️ Test driveId ignoré :", e)
