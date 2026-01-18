import os
import io
from typing import List, Dict, Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from googleapiclient.errors import HttpError

# --- Constantes partagées (compat Mon Drive & Drive partagés) ---------------
_COMMON_LIST_KW = {"supportsAllDrives": True, "includeItemsFromAllDrives": True}
_COMMON_GET_KW = {"supportsAllDrives": True}
_COMMON_CREATE_KW = {"supportsAllDrives": True}

FOLDER_MIME = "application/vnd.google-apps.folder"


class GoogleDriveManager:
    """Gestion simplifiée de Google Drive (Service Account).

    - Compatible Mon Drive et Drive partagés.
    - Ajoute automatiquement supportsAllDrives/includeItemsFromAllDrives aux requêtes.
    - Fournit des helpers pour lister, télécharger, uploader, créer des dossiers.
    """

    def __init__(self, credentials_path: str, base_folder_id: str):
        self.creds = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=["https://www.googleapis.com/auth/drive"]
        )
        self.service = build("drive", "v3", credentials=self.creds)
        self.base_folder_id = base_folder_id
        self.base_drive_id = self.get_drive_id(base_folder_id)

    # ------------------------------------------------------------------
    # Utilitaires internes
    # ------------------------------------------------------------------
    def get_drive_id(self, file_id: str) -> Optional[str]:
        """Retourne le driveId (None si Mon Drive ou si non détectable)."""
        try:
            meta = self.service.files().get(
                fileId=file_id, fields="id,name,driveId", **_COMMON_GET_KW
            ).execute()
            return meta.get("driveId")
        except Exception:
            return None

    def _list(self, **kwargs) -> List[Dict]:
        """List avec pagination automatique (agrège toutes les pages)."""
        items: List[Dict] = []
        page_token: Optional[str] = None
        while True:
            resp = self.service.files().list(pageToken=page_token, **kwargs).execute()
            items.extend(resp.get("files", []))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        return items

    # ------------------------------------------------------------------
    # Dossiers
    # ------------------------------------------------------------------
    def find_folder_id(self, folder_name: str, parent_id: Optional[str] = None) -> Optional[str]:
        """Trouve l'ID d'un dossier enfant par son nom sous parent_id (par défaut: base_folder_id)."""
        if parent_id is None:
            parent_id = self.base_folder_id
        q = (
            f"'{parent_id}' in parents and "
            f"mimeType='{FOLDER_MIME}' and name='{folder_name}' and trashed=false"
        )
        files = self._list(
            q=q, fields="nextPageToken, files(id,name,parents,driveId)", **_COMMON_LIST_KW
        )
        return files[0]["id"] if files else None

    def ensure_folder(self, name: str, parent_id: Optional[str] = None) -> str:
        """Retourne l'ID du dossier 'name' sous parent_id, en le créant si besoin."""
        fid = self.find_folder_id(name, parent_id)
        if fid:
            return fid
        body: Dict = {"name": name, "mimeType": FOLDER_MIME}
        if parent_id:
            body["parents"] = [parent_id]
        file = self.service.files().create(body=body, fields="id", **_COMMON_CREATE_KW).execute()
        return file["id"]

    def ensure_path(self, path: List[str]) -> str:
        """Crée/résout une arborescence de dossiers sous base_folder_id. Renvoie l'ID final."""
        parent = self.base_folder_id
        for name in path:
            parent = self.ensure_folder(name, parent)
        return parent

    # ------------------------------------------------------------------
    # Fichiers
    # ------------------------------------------------------------------
    def list_pdfs_in_folder(self, folder_id: str, drive_id: Optional[str] = None) -> List[Dict]:
        """Liste tous les PDF d'un dossier (pagination gérée)."""
        query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
        kwargs = dict(
            q=query,
            fields="nextPageToken, files(id,name,mimeType,parents,driveId)",
            pageSize=1000,
            **_COMMON_LIST_KW,
        )
        # Scope explicite au Drive partagé si connu (meilleur pour perfs/fiabilité)
        drive_scope = drive_id or self.base_drive_id
        if drive_scope:
            kwargs.update({"corpora": "drive", "driveId": drive_scope})
        return self._list(**kwargs)

    def download_file(self, file_id: str, destination_path: str) -> None:
        """Télécharge un fichier Drive vers un chemin local."""
        request = self.service.files().get_media(fileId=file_id)
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        with open(destination_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

    def get_file(self, file_id: str) -> Dict:
        return self.service.files().get(
            fileId=file_id,
            fields="id,name,mimeType,parents,driveId",
            **_COMMON_GET_KW,
        ).execute()

    def delete_file(self, file_id: str) -> None:
        self.service.files().delete(fileId=file_id, **_COMMON_GET_KW).execute()

    def upload_file(
        self,
        local_path: str,
        remote_name: str,
        parent_folder_id: str,
        overwrite: bool = True,
    ) -> Dict:
        """Upload d'un fichier. Si overwrite=True, supprime les homonymes dans le dossier cible."""
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
        return self.service.files().create(
            body=body, media_body=media, fields="id", **_COMMON_CREATE_KW
        ).execute()
