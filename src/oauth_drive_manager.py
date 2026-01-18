"""
OAuth Drive Manager – Gestion Google Drive via OAuth (utilisateur final)
-----------------------------------------------------------------------

Objectif : fournir une API quasi-équivalente à GoogleDriveManager (Service Account),
mais basée sur OAuth pour agir *au nom de l'utilisateur connecté*.

Points clés :
- Compatible *Mon Drive* et *Drives partagés* (supportsAllDrives).
- Multi‑utilisateur : tokens stockés par "clé utilisateur" (ex. e‑mail Google).
- Deux modes d'init :
  * Desktop/dev : InstalledAppFlow (run_local_server)
  * Web/app : flow Web (générer URL d'autorisation + callback fetch_token)
- API principale :
  * get_authorization_url(...), fetch_token(...), revoke()
  * set_base_folder(...), get_drive_id(...)
  * find_folder_id(...), ensure_folder(...), ensure_path(...)
  * list_pdfs_in_folder(...), download_file(...), upload_file(...), get_file(...), delete_file(...)

Dépendances :
    google-api-python-client
    google-auth
    google-auth-oauthlib

NB : pensez à sécuriser la persistance des tokens en production (chiffrement au repos).
"""
from __future__ import annotations

import os
import io
import json
from typing import Dict, List, Optional, Tuple

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from googleapiclient.errors import HttpError

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow, Flow
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request



# ----------------------------- Constantes API -------------------------------
FOLDER_MIME = "application/vnd.google-apps.folder"
_COMMON_LIST_KW = {"supportsAllDrives": True, "includeItemsFromAllDrives": True}
_COMMON_GET_KW = {"supportsAllDrives": True}
_COMMON_CREATE_KW = {"supportsAllDrives": True}

# Scopes par défaut : lecture globale + écriture limitée aux fichiers créés par l'appli
DEFAULT_SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.file",
]


# ------------------------------ Token Store --------------------------------
class FileTokenStore:
    """Stockage simple des tokens OAuth par utilisateur.

    Les tokens sont stockés en JSON sur disque (non chiffrés). En production,
    préférez un stockage chiffré (KMS/DB chiffrée) et ne committez jamais ces fichiers.
    """

    def __init__(self, root_dir: str = ".oauth_tokens"):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)

    def _path(self, user_key: str) -> str:
        safe = user_key.replace("/", "_")
        return os.path.join(self.root_dir, f"token_{safe}.json")

    def load(self, user_key: str) -> Optional[Dict]:
        path = self._path(user_key)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                return None

    def save(self, user_key: str, token_dict: Dict) -> None:
        path = self._path(user_key)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(token_dict, f)

    def delete(self, user_key: str) -> None:
        path = self._path(user_key)
        if os.path.exists(path):
            os.remove(path)


# --------------------------- OAuth Drive Manager ----------------------------
class OAuthDriveManager:
    def __init__(
        self,
        client_config: Dict | str,
        token_store: FileTokenStore | None,
        user_key: str,
        scopes: List[str] | None = None,
        base_folder_id: Optional[str] = None,
        user_email_hint: Optional[str] = None,
    ):
        """
        client_config: dict OU chemin vers client_secret.json (Google Cloud Console)
        token_store: implémentation de stockage des tokens (FileTokenStore par défaut)
        user_key: identifiant de l'utilisateur final (ex. e‑mail) pour indexer les tokens
        scopes: liste des scopes OAuth (défaut = DEFAULT_SCOPES)
        base_folder_id: dossier racine Drive à utiliser par défaut (optionnel)
        user_email_hint: optionnel, pour pré-remplir le compte sur la page d'auth
        """
        self.scopes = scopes or DEFAULT_SCOPES
        self.user_key = user_key
        self.user_email_hint = user_email_hint
        self.token_store = token_store or FileTokenStore()
        self.base_folder_id = base_folder_id

        # Charger la conf client
        if isinstance(client_config, str):
            with open(client_config, "r", encoding="utf-8") as f:
                self.client_config = json.load(f)
        else:
            self.client_config = dict(client_config)

        self.creds: Optional[Credentials] = None
        self.service = None
        self.base_drive_id: Optional[str] = None

        # Tente de charger des tokens existants
        self._load_existing_tokens()

        if self.creds and self.creds.expired and self.creds.refresh_token:
            try:
                self.creds.refresh(Request())
                self._persist_tokens()
            except RefreshError as e:
                # Le cas typique : refresh_token révoqué / mismatch client => invalid_grant
                # On invalide les creds pour forcer une ré-auth (oauth_init ou flow web)
                self.creds = None
                self.service = None
                raise e

        if self.creds and self.creds.valid:
            self._build_service()

        # Si base_folder_id fourni et service prêt, récupère driveId (Drive partagé)
        if self.base_folder_id and self.service is not None:
            self.base_drive_id = self.get_drive_id(self.base_folder_id)

    # ---------------------- Auth: Desktop (dev) ----------------------
    @classmethod
    def from_desktop_flow(
        cls,
        client_secrets_file: str,
        user_key: str,
        token_store: FileTokenStore | None = None,
        scopes: List[str] | None = None,
        base_folder_id: Optional[str] = None,
    ) -> "OAuthDriveManager":
        scopes = scopes or DEFAULT_SCOPES
        flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, scopes=scopes)
        creds = flow.run_local_server(port=0, access_type="offline", include_granted_scopes="true", prompt="consent")
        mgr = cls(client_secrets_file, token_store, user_key, scopes=scopes, base_folder_id=base_folder_id)
        mgr.creds = creds
        mgr._persist_tokens()
        mgr._build_service()
        if mgr.base_folder_id:
            mgr.base_drive_id = mgr.get_drive_id(mgr.base_folder_id)
        return mgr

    def _try_refresh(self) -> None:
        if not self.creds:
            return
        if self.creds.expired and self.creds.refresh_token:
            try:
                self.creds.refresh(Request())
                self._persist_tokens()
            except RefreshError as e:
                msg = str(e)
                if "invalid_grant" in msg:
                    try:
                        self.token_store.delete(self.user_key)
                    except Exception:
                        pass
                self.creds = None
                self.service = None
                raise
            """except RefreshError as e:
                # invalid_grant = refresh token mort => il faut refaire le flow
                msg = str(e)
                if "invalid_grant" in msg:
                    # option: supprimer le token local pour éviter boucle infinie
                    try:
                        self.token_store.delete(self.user_key)  # à implémenter si absent
                    except Exception:
                        pass
                raise"""
    # ----------------------- Auth: Web (prod) ------------------------
    def get_authorization_url(
        self,
        redirect_uri: str,
        state: Optional[str] = None,
        login_hint: Optional[str] = None,
        prompt: str = "consent",
        access_type: str = "offline",
        include_granted_scopes: str = "true",
    ) -> Tuple[str, str]:
        """Construit l'URL d'autorisation OAuth (pour rediriger l'utilisateur).
        Retourne (authorization_url, state_effectif).
        """
        flow = Flow.from_client_config(self.client_config, scopes=self.scopes)
        flow.redirect_uri = redirect_uri
        auth_url, state_val = flow.authorization_url(
            access_type=access_type,
            include_granted_scopes=include_granted_scopes,
            prompt=prompt,
            login_hint=login_hint or self.user_email_hint,
        )
        # Persister state (optionnel) si vous souhaitez le valider au callback
        self._pending_state = state_val
        return auth_url, state_val

    def fetch_token(self, authorization_response: str, redirect_uri: str) -> None:
        """Échange le code d'auth contre des tokens, puis construit le service Drive."""
        flow = Flow.from_client_config(self.client_config, scopes=self.scopes)
        flow.redirect_uri = redirect_uri
        flow.fetch_token(authorization_response=authorization_response)
        self.creds = flow.credentials
        self._persist_tokens()
        self._build_service()
        if self.base_folder_id:
            self.base_drive_id = self.get_drive_id(self.base_folder_id)

    def get_auth_url(self, redirect_uri: str, state: str | None = None) -> tuple[str, str]:
        """
        Retourne (auth_url, state). L’utilisateur doit ouvrir auth_url.
        """
        client_cfg = self.client_config
        # attend une structure "web": {client_id, client_secret, auth_uri, token_uri, ...}
        flow = Flow.from_client_config(
            client_cfg,
            scopes=self.scopes,
            redirect_uri=redirect_uri,
        )
        auth_url, new_state = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
            state=state,
        )
        return auth_url, new_state

    def fetch_token_from_code(self, code: str, redirect_uri: str) -> None:
        """
        Échange le code contre des tokens, puis persist et construit le service.
        """
        flow = Flow.from_client_config(
            self.client_config,
            scopes=self.scopes,
            redirect_uri=redirect_uri,
        )
        flow.fetch_token(code=code)
        self.creds = flow.credentials
        self._persist_tokens()
        if self.creds and self.creds.valid:
            self._build_service()

    def revoke(self) -> None:
        if self.creds:
            try:
                self.creds.revoke(Request())
            except Exception:
                pass
        self.creds = None
        self.service = None
        self.token_store.delete(self.user_key)

    # ------------------------ Helpers internes ----------------------
    def _load_existing_tokens(self) -> None:
        tok = self.token_store.load(self.user_key)
        if not tok:
            return
        self.creds = Credentials.from_authorized_user_info(tok, scopes=self.scopes)

    def _persist_tokens(self) -> None:
        if not self.creds:
            return
        data = {
            "token": self.creds.token,
            "refresh_token": self.creds.refresh_token,
            "token_uri": self.creds.token_uri,
            "client_id": self.creds.client_id,
            "client_secret": self.creds.client_secret,
            "scopes": self.creds.scopes,
        }
        self.token_store.save(self.user_key, data)

    def _build_service(self) -> None:
        if not self.creds or not self.creds.valid:
            raise RuntimeError("Crédentiel OAuth invalide : appelez get_authorization_url/fetch_token ou from_desktop_flow.")
        self.service = build("drive", "v3", credentials=self.creds)

    # --------------------------- Contexte Drive ----------------------
    def set_base_folder(self, base_folder_id: str) -> None:
        self.base_folder_id = base_folder_id
        if self.service is not None:
            self.base_drive_id = self.get_drive_id(base_folder_id)

    def get_drive_id(self, file_id: str) -> Optional[str]:
        try:
            meta = self.service.files().get(fileId=file_id, fields="id,name,driveId", **_COMMON_GET_KW).execute()
            return meta.get("driveId")
        except Exception:
            return None

    # ------------------------------ Listage -------------------------
    def _list(self, **kwargs) -> List[Dict]:
        items: List[Dict] = []
        page_token: Optional[str] = None
        while True:
            resp = self.service.files().list(pageToken=page_token, **kwargs).execute()
            items.extend(resp.get("files", []))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        return items

    def find_folder_id(self, folder_name: str, parent_id: Optional[str] = None) -> Optional[str]:
        parent = parent_id or self.base_folder_id
        if not parent:
            raise ValueError("base_folder_id non défini : appelez set_base_folder().")
        q = f"'{parent}' in parents and mimeType='{FOLDER_MIME}' and name='{folder_name}' and trashed=false"
        kwargs = dict(q=q, fields="nextPageToken, files(id,name,parents,driveId)", **_COMMON_LIST_KW)
        drive_scope = self.base_drive_id
        if drive_scope:
            kwargs.update({"corpora": "drive", "driveId": drive_scope})
        files = self._list(**kwargs)
        return files[0]["id"] if files else None

    def ensure_folder(self, name: str, parent_id: Optional[str] = None) -> str:
        fid = self.find_folder_id(name, parent_id)
        if fid:
            return fid
        body: Dict = {"name": name, "mimeType": FOLDER_MIME}
        if parent_id or self.base_folder_id:
            body["parents"] = [parent_id or self.base_folder_id]
        file = self.service.files().create(body=body, fields="id", **_COMMON_CREATE_KW).execute()
        return file["id"]

    def ensure_path(self, path: List[str]) -> str:
        parent = self.base_folder_id
        if not parent:
            raise ValueError("base_folder_id non défini : appelez set_base_folder().")
        for name in path:
            parent = self.ensure_folder(name, parent)
        return parent

    def ensure_valid(self) -> None:
        """Force le refresh si expiré et reconstruit le service si nécessaire."""
        if not self.creds:
            return

        if self.creds.expired and self.creds.refresh_token:
            self.creds.refresh(Request())
            self._persist_tokens()

        # Si le service n'est pas construit ou a été invalidé, on le rebâtit
        if self.creds and self.creds.valid and self.service is None:
            self._build_service()

    def list_pdfs_in_folder(self, folder_id: str) -> List[Dict]:
        q = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
        kwargs = dict(q=q, fields="nextPageToken, files(id,name,mimeType,parents,driveId)", pageSize=1000, **_COMMON_LIST_KW)
        drive_scope = self.base_drive_id
        if drive_scope:
            kwargs.update({"corpora": "drive", "driveId": drive_scope})
        return self._list(**kwargs)

    # ------------------------------ Fichiers -------------------------
    def download_file(self, file_id: str, destination_path: str) -> None:
        request = self.service.files().get_media(fileId=file_id)
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        with open(destination_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

    def upload_file(self, local_path: str, remote_name: str, parent_folder_id: str, overwrite: bool = True) -> Dict:
        if overwrite:
            q = f"'{parent_folder_id}' in parents and name='{remote_name}' and trashed=false"
            existing = self._list(q=q, fields="nextPageToken, files(id)", **_COMMON_LIST_KW)
            for it in existing:
                try:
                    self.service.files().delete(fileId=it["id"], **_COMMON_GET_KW).execute()
                except HttpError:
                    pass
        media = MediaFileUpload(local_path, resumable=True)
        body = {"name": remote_name, "parents": [parent_folder_id]}
        return self.service.files().create(body=body, media_body=media, fields="id", **_COMMON_CREATE_KW).execute()

    def get_file(self, file_id: str) -> Dict:
        return self.service.files().get(fileId=file_id, fields="id,name,mimeType,parents,driveId", **_COMMON_GET_KW).execute()

    def delete_file(self, file_id: str) -> None:
        self.service.files().delete(fileId=file_id, **_COMMON_GET_KW).execute()


# -------------------------- Helpers d'intégration ---------------------------
def build_manager_for_streamlit(
    client_secrets_json: Dict | str,
    token_store: FileTokenStore | None,
    user_email: str,
    base_folder_id: Optional[str] = None,
    scopes: Optional[List[str]] = None,
) -> OAuthDriveManager:
    """Helper pratique pour Streamlit :
    - charge/rafraîchit automatiquement les tokens si présents
    - sinon, il faudra appeler get_authorization_url(...) puis fetch_token(...)
    """
    mgr = OAuthDriveManager(client_secrets_json, token_store, user_key=user_email, scopes=scopes or DEFAULT_SCOPES, base_folder_id=base_folder_id, user_email_hint=user_email)
    return mgr
