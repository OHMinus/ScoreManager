"""
Firebase と Google Drive の連携を担当するモジュール
score_api.py をコアロジック（ローカル処理）とデータ永続化層に分離するため、
ここで外部サービス（Firebase/Google Drive）の処理を集約する
"""
import gc
import os
import json
import glob
import firebase_admin
from firebase_admin import credentials, db as firebase_db
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from PIL import Image
import io
import concurrent.futures

# ==========================================
# Firebase と Google Drive の初期化
# ==========================================

def initialize_firebase():
    """Firebase を初期化する"""
    firebase_cred_path = os.environ.get('FIREBASE_CRED_PATH')
    firebase_database_url = os.environ.get('FIREBASE_DATABASE_URL')
    firebase_cred = os.environ.get('FIREBASE_CRED')
    if (not firebase_cred_path and not firebase_cred) or not firebase_database_url:
        return False
    
    if firebase_admin._apps:
        return True
    
    try:
        if firebase_cred:
            # 環境変数から直接証明書を読み込む場合は一時ファイルに保存してから読み込む
            with open('temp_firebase_cred.json', 'w') as f:
                f.write(firebase_cred)
            firebase_cred_path = 'temp_firebase_cred.json'
        cred = credentials.Certificate(firebase_cred_path)
        firebase_admin.initialize_app(cred, {'databaseURL': firebase_database_url})
        print("✓ Firebase initialized.")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize Firebase: {e}")
        return False


def initialize_google_drive(token_json_str=None):
    """OAuth 2.0 を使用して Google Drive API を初期化する。token_json_str がない場合はサービスアカウントでフォールバックする。"""
    SCOPES = ['https://www.googleapis.com/auth/drive.file', 'https://www.googleapis.com/auth/drive.readonly']
    creds = None
    
    if token_json_str:
        try:
            creds = Credentials.from_authorized_user_info(json.loads(token_json_str), SCOPES)
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
        except Exception as e:
            print(f"✗ Failed to load/refresh token from session string: {e}")
            creds = None
    elif os.path.exists('token.json'):
        try:
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
        except Exception as e:
            print(f"✗ Failed to load/refresh token: {e}")
            # Do not return immediately, fallback to service account
            creds = None

    if not creds or not creds.valid:
        service_account_path = os.environ.get('GOOGLE_DRIVE_SERVICE_ACCOUNT_PATH', 'service_account.json')
        client_secret_path = os.environ.get('GOOGLE_DRIVE_CRED_PATH', 'client_secret.json')

        # Check explicit service account path first, then fallback to checking client secret path as SA
        if os.path.exists(service_account_path):
            try:
                creds = service_account.Credentials.from_service_account_file(
                    service_account_path, scopes=SCOPES)
                print("✓ Google Drive API initialized with Service Account.")
            except Exception as e:
                print(f"✗ Failed to load Service Account: {e}")
        elif os.path.exists(client_secret_path):
             # Some users might configure GOOGLE_DRIVE_CRED_PATH with a service account JSON by mistake or design
             try:
                 creds = service_account.Credentials.from_service_account_file(
                     client_secret_path, scopes=SCOPES)
                 print("✓ Google Drive API initialized with Service Account from client_secret path.")
             except Exception:
                 pass # It's probably an OAuth client secret file, so we ignore the error

    if creds and creds.valid:
        try:
            drive_service = build('drive', 'v3', credentials=creds)
            print("✓ Google Drive API initialized successfully.")
            return drive_service
        except Exception as e:
            print(f"✗ Failed to initialize Google Drive: {e}")

    return None


# ==========================================
# DB 操作（Firebase or JSON ファイル対応）
# ==========================================

def load_db_from_firebase():
    """Firebase Realtime Database からデータを読み込む"""
    if not firebase_admin._apps:
        return None
    
    try:
        ref = firebase_db.reference('scores')
        data = ref.get()
        return data if data else {}
    except Exception as e:
        print(f"✗ Failed to load from Firebase: {e}")
        return None


def save_db_to_firebase(data):
    """Firebase Realtime Database にデータを保存する"""
    if not firebase_admin._apps:
        return False
    
    try:
        ref = firebase_db.reference('scores')
        ref.set(data)
        print("✓ Data saved to Firebase.")
        return True
    except Exception as e:
        print(f"✗ Failed to save to Firebase: {e}")
        return False


def load_db_from_json(db_path="scores_db.json"):
    """JSON ファイルからデータを読み込む"""
    if not os.path.exists(db_path):
        return {}
    
    try:
        with open(db_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"✗ Failed to load from JSON: {e}")
        return {}


def save_db_to_json(data, db_path="scores_db.json"):
    """JSON ファイルにデータを保存する"""
    try:
        with open(db_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        print(f"✗ Failed to save to JSON: {e}")
        return False


# ==========================================
# 統合 DB アダプター
# ==========================================

class DatabaseAdapter:
    """
    Firebase Realtime Database または JSON ファイルを透過的に使える
    アダプターパターン
    """
    
    def __init__(self, use_firebase=True, json_path="scores_db.json"):
        """
        Args:
            use_firebase: Firebase を使うか（初期化されていない場合は JSON に fallback）
            json_path: JSON ファイルのパス
        """
        self.use_firebase = use_firebase and firebase_admin._apps
        self.json_path = json_path
    
    def load(self):
        """データを読み込む"""
        local_data = load_db_from_json(self.json_path)
        if self.use_firebase:
            firebase_data = load_db_from_firebase()
            if firebase_data is not None:
                # Merge logic: priorites Firebase instruments with urls
                # Update local data with Firebase data, keeping local-only files
                merged_data = local_data.copy()
                for score_id, fb_score in firebase_data.items():
                    if score_id not in merged_data:
                        merged_data[score_id] = fb_score
                    else:
                        for fb_inst_name, fb_inst_data in fb_score.get('instruments', {}).items():
                            # If firebase has urls, update local
                            if isinstance(fb_inst_data, list):
                                merged_data[score_id].setdefault('instruments', {})[fb_inst_name] = fb_inst_data
                return merged_data
        return local_data
    
    def save(self, data):
        """データを保存する"""
        success_json = save_db_to_json(data, self.json_path)
        
        if self.use_firebase:
            # Only save items with URLs to firebase
            firebase_data = {}
            for score_id, score in data.items():
                filtered_instruments = {}
                for inst_name, inst_data in score.get('instruments', {}).items():
                    if isinstance(inst_data, list):
                        filtered_instruments[inst_name] = inst_data
                if filtered_instruments:
                    fb_score = score.copy()
                    fb_score['instruments'] = filtered_instruments
                    firebase_data[score_id] = fb_score

            success_firebase = save_db_to_firebase(firebase_data)
            return success_firebase and success_json
        
        return success_json


# ==========================================
# Google Drive 操作
# ==========================================
def _get_folder_by_name(drive_service, parent_id, folder_name):
    """
    指定した親フォルダ内に同名のフォルダが存在するか検索してIDを返す（内部用）
    """
    try:
        # フォルダ名にシングルクォートが含まれる場合のエスケープ処理
        safe_name = folder_name.replace("'", "\\'")
        query = f"name='{safe_name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        
        results = drive_service.files().list(
            q=query, spaces='drive', fields='files(id, name)',
            supportsAllDrives=True, includeItemsFromAllDrives=True
        ).execute()
        
        items = results.get('files', [])
        if items:
            return items[0]['id']
        return None
    except Exception as e:
        print(f"✗ Failed to search folder '{folder_name}': {e}")
        return None

def get_or_create_drive_path(drive_service, root_id, path_components):
    """
    リストで渡されたパス階層を辿り、存在しない場合は作成しながら末端のフォルダIDを返す
    
    Args:
        drive_service: Google Drive service オブジェクト
        root_id: 起点となるルートのフォルダID (GOOGLE_DRIVE_FOLDER_ID)
        path_components: 辿る階層のリスト (例: ["2026春合宿", "宝島", "Alto_Sax"])
        
    Returns:
        末端のフォルダID (文字列)、失敗時は None
    """
    if not drive_service or not root_id or not path_components:
        return root_id
        
    current_parent_id = root_id
    
    for folder_name in path_components:
        # 1. 既存のフォルダを探す
        folder_id = _get_folder_by_name(drive_service, current_parent_id, folder_name)
        
        # 2. 存在しなければ作成する
        if not folder_id:
            try:
                file_metadata = {
                    'name': folder_name,
                    'mimeType': 'application/vnd.google-apps.folder',
                    'parents': [current_parent_id]
                }
                folder = drive_service.files().create(
                    body=file_metadata, fields='id',
                    supportsAllDrives=True
                ).execute()
                
                folder_id = folder.get('id')
                print(f"✓ Created new folder: {folder_name}")
            except Exception as e:
                print(f"✗ Failed to create folder '{folder_name}': {e}")
                return None
                
        # 次の階層の親IDとしてセット
        current_parent_id = folder_id
        
    return current_parent_id


def upload_image_to_google_drive(pil_image, filename, drive_service, folder_id):
    """
    PIL Image を Google Drive にアップロードし、アクセス可能な URL を返す
    """
    if not folder_id:
        return None, None
    
    # 最後に必ず close するため、あらかじめ None で初期化
    img_io = None

    try:
        # PIL Image をバイトストリームに変換
        img_io = io.BytesIO()
        print(f"saving {filename}")
        pil_image.save(img_io, format='PNG')
        img_io.seek(0)
        
        # Google Drive にアップロード用のメタデータ
        file_metadata = {
            'name': filename,
            'parents': [folder_id],
            'mimeType': 'image/png'
        }
        
        # 1. 5MB単位で分割アップロードを設定
        media = MediaIoBaseUpload(img_io, mimetype='image/png', chunksize=256*1024, resumable=True)
        
        # ※ここではまだ .execute() は呼ばない
        request = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id',
            supportsAllDrives=True
        )
        
        # 2. ループを回して実際に分割アップロードを実行する
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                print(f"  -> {filename} Uploading... {int(status.progress() * 100)}%")
        
        # アップロード完了後のファイルID取得
        file_id = response.get('id')
        
        # ファイルを公開設定にする
        drive_service.permissions().create(
            fileId=file_id,
            body={'type': 'anyone', 'role': 'reader'},
            supportsAllDrives=True
        ).execute()
        
        # 公開 URL を生成
        public_url = f"https://drive.google.com/uc?id={file_id}"
        
        print(f"✓ Uploaded to Google Drive: {filename}")
        gc.collect()
        return file_id, public_url
        
    except Exception as e:
        print(f"✗ Failed to upload to Google Drive: {e}")
        return None, None
        
    finally:
        # 3. 正常終了・異常終了に関わらず、メモリ上のストリームを確実に解放する
        if img_io:
            img_io.close()
        
        # 呼び出し元のループ処理でメモリが逼迫するのを防ぐため、
        # この関数内で使い終わった pil_image も明示的にクローズすることを推奨します
        if pil_image:
            pil_image.close()

def upload_score_pages_to_google_drive(score_dir, score_id, instrument, drive_service, folder_id):
    """
    ローカルに保存されたスコアページを Google Drive にアップロードし、URL リストを返す
    
    Args:
        score_dir: ローカルスコアディレクトリパス
        score_id: スコアID
        instrument: 楽器名
        drive_service: Google Drive service オブジェクト
        folder_id: アップロード先フォルダID
    
    Returns:
        URL のリスト、または空リスト（エラー時）
    """
    if not drive_service or not folder_id:
        return []
    
    try:
        # 重複アップロードを避けるため、既存のファイル一覧を取得
        query = f"'{folder_id}' in parents and trashed=false"
        results = drive_service.files().list(
            q=query, spaces='drive', fields='files(id, name)',
            supportsAllDrives=True, includeItemsFromAllDrives=True
        ).execute()
        existing_files = {item['name']: item['id'] for item in results.get('files', [])}

        image_files = sorted(glob.glob(os.path.join(score_dir, "*.png")))
        urls_list = [None] * len(image_files)

        # 🔴 ThreadPoolExecutor を廃止し、安全な通常の for ループに変更
        for i, image_file in enumerate(image_files):
            filename = os.path.basename(image_file)
            
            if filename in existing_files:
                # 既にファイルが存在する場合はスキップ
                file_id = existing_files[filename]
                url = f"https://drive.google.com/uc?id={file_id}"
                print(f"✓ Skipped existing file on Google Drive: {filename}")
                urls_list[i] = url
            else:
                # 存在しない場合のみ、Pillowで開いてアップロード
                pil_image = Image.open(image_file)
                # 前回の修正通り、drive_service を引数として渡す
                _, url = upload_image_to_google_drive(pil_image, filename, drive_service, folder_id)
                urls_list[i] = url
        
        return urls_list
        
    except Exception as e:
        print(f"✗ Failed to upload score pages: {e}")
        return []

# ==========================================
# 便利関数
# ==========================================

def get_db_adapter(use_firebase=True):
    """
    データベースアダプターを取得する
    
    Args:
        use_firebase: Firebase を優先的に使うか
    
    Returns:
        DatabaseAdapter インスタンス
    """
    return DatabaseAdapter(use_firebase=use_firebase, json_path="scores_db.json")


def is_firebase_available():
    """Firebase が利用可能か判定する"""
    return bool(firebase_admin._apps)


def is_google_drive_available():
    """Google Drive が利用可能か判定する"""
    client_secret_path = os.environ.get('GOOGLE_DRIVE_CRED_PATH', 'client_secret.json')
    service_account_path = os.environ.get('GOOGLE_DRIVE_SERVICE_ACCOUNT_PATH', 'service_account.json')
    return os.path.exists('token.json') or os.path.exists(client_secret_path) or os.path.exists(service_account_path)
