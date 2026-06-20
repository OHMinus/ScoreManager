from flask import Flask, send_from_directory, render_template, request, redirect, url_for, flash, send_file, jsonify, session
import os
os.environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = '1'
import glob
from google_auth_oauthlib.flow import Flow
import uuid
import datetime
import time
import threading
import concurrent.futures
from PIL import Image
import cv2
import pytesseract
import zipfile
import io
import urllib.parse
from googleapiclient.http import MediaIoBaseDownload
from dotenv import load_dotenv
from werkzeug.middleware.proxy_fix import ProxyFix
import json
from werkzeug.utils import secure_filename

# dotenv はモジュールの読み込み前に実行する
if os.path.isfile(".env"):
    load_dotenv()

import score_api
import firebase_db

app = Flask(__name__)

progress_store = {}
result_store = {}

@app.route('/progress/<task_id>', methods=['GET'])
def get_progress(task_id):
    progress = progress_store.get(task_id, 0)
    return jsonify({"progress": progress})

app.secret_key = 'score_processor_secret_key'

# Firebase と Google Drive の初期化
firebase_db.initialize_firebase()
# global default drive service (fallback to service account or local token.json)
drive_service = firebase_db.initialize_google_drive()
GOOGLE_DRIVE_FOLDER_ID = os.environ.get('GOOGLE_DRIVE_FOLDER_ID')

def get_drive_service(token):
    """Helper to get drive service utilizing session token if available."""
    return firebase_db.initialize_google_drive(token) if token else drive_service

# データベースアダプターを初期化
db_adapter = firebase_db.get_db_adapter(use_firebase=firebase_db.is_firebase_available())
score_api.set_db_adapter(db_adapter)

TEMP_UPLOAD_DIR = os.path.join('static', 'temp', 'uploads')
TEMP_PREVIEW_DIR = os.path.join('static', 'temp', 'previews')
TEMP_DEBUG_DIR = os.path.join('static', 'temp', 'debug') # 新規追加: デバッグ画像の保存先
TEMP_UNCROPPED_DIR = os.path.join('static', 'temp', 'uncropped')

os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMP_PREVIEW_DIR, exist_ok=True)
os.makedirs(TEMP_DEBUG_DIR, exist_ok=True)
os.makedirs(TEMP_UNCROPPED_DIR, exist_ok=True)

def clear_temp_dir(directory, max_age_hours=1):
    now = time.time()
    for f in glob.glob(os.path.join(directory, '*')):
        try:
            if os.stat(f).st_mtime < now - max_age_hours * 3600: os.remove(f)
        except: pass

def extract_info_from_header(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return "", ""
        h, w = img.shape
        header_img = img[0:int(h * 0.15), 0:w]
        _, thresh = cv2.threshold(header_img, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(thresh, lang='eng').strip()
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        piece_guess = lines[0] if len(lines) > 0 else ""
        inst_guess = lines[1] if len(lines) > 1 else ""
        piece_guess = "".join(c for c in piece_guess if c.isalnum() or c in " -_")
        inst_guess = "".join(c for c in inst_guess if c.isalnum() or c in " -_")
        return piece_guess, inst_guess
    except: return "", ""

@app.route('/login')
def authorize_drive():
    if 'GOOGLE_DRIVE_CRED' in os.environ:
        open('client_secret.json', 'w').write(os.environ['GOOGLE_DRIVE_CRED'])
    client_secret_path = os.environ.get('GOOGLE_DRIVE_CRED_PATH', 'client_secret.json')
    if not os.path.exists(client_secret_path):
        flash('Google Drive API credentials (client_secret.json) not found.')
        return redirect(url_for('index'))

    flow = Flow.from_client_secrets_file(
        client_secret_path,
        scopes=['https://www.googleapis.com/auth/drive.file']
    )

    flow.redirect_uri = url_for('oauth2callback', _external=True)

    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        prompt='consent'
    )

    print(f"set redirect url as {authorization_url} from {flow.redirect_uri}")

    session['state'] = state
    session['code_verifier'] = flow.code_verifier
    return redirect(authorization_url)


@app.route('/oauth2callback')
def oauth2callback():
    client_secret_path = os.environ.get('GOOGLE_DRIVE_CRED_PATH', 'client_secret.json')
    state = session.get('state')

    if not state:
        flash('Invalid OAuth state.')
        return redirect(url_for('index'))

    flow = Flow.from_client_secrets_file(
        client_secret_path,
        scopes=['https://www.googleapis.com/auth/drive.file'],
        state=state
    )
    flow.redirect_uri = url_for('oauth2callback', _external=True)

    # 現在のURLを取得
    authorization_response = request.url

    # PKCE のための code_verifier を復元
    if 'code_verifier' in session:
        flow.code_verifier = session['code_verifier']

    try:
        flow.fetch_token(authorization_response=authorization_response)
        credentials = flow.credentials

        session['google_token'] = credentials.to_json()

        # グローバルの drive_service を再初期化
        global drive_service
        drive_service = firebase_db.initialize_google_drive(session['google_token'])

        session['google_logged_in'] = True
        flash('Google Drive 認証が完了しました！')
    except Exception as e:
        flash(f'Google Drive 認証に失敗しました: {e}')
    print(f"session data of google_logged_in : {session["google_logged_in"]}")
    return redirect(url_for('index'))


@app.route('/session/view')
def SessionView():
    return render_template("session_view.html")

@app.route('/session/get')
def GetSession():
    if os.environ.get('isDebug'):
        jsonStr = json.dumps(
            dict(session),
            ensure_ascii=False,  # Keep non-ASCII characters as-is
            indent=4             # Pretty-print with indentation
        )
        return jsonStr
    else:
        return None


@app.route('/')
def index():
    clear_temp_dir(TEMP_UPLOAD_DIR)
    clear_temp_dir(TEMP_PREVIEW_DIR)
    clear_temp_dir(TEMP_DEBUG_DIR)
    clear_temp_dir(TEMP_UNCROPPED_DIR)

    # ログイン状態の確認
    token = session.get('google_token')
    google_logged_in = False
    if token:
        ds = firebase_db.initialize_google_drive(token)
        if ds:
            google_logged_in = True
        else:
            session.pop('google_token', None)
            session.pop('google_logged_in', None)

    session['google_logged_in'] = google_logged_in

    printers = GetPrinterState()
    if len(printers.keys()) == 0:
        printers = None

    return render_template('index.html',printer=printers, google_logged_in=google_logged_in)

def process_images_in_background(task_id, file_paths):
    first_file_path = file_paths[0] if file_paths else None
    
    try:
        total_files = len(file_paths)
        preview_filenames_list = [[] for _ in range(total_files)]
        progress_lock = threading.Lock()
        completed_files = [0]

        def process_single_file(i, temp_path):
            local_preview_filenames = []

            def progress_cb(p):
                if task_id:
                    with progress_lock:
                        base_progress = (completed_files[0] / total_files) * 100
                        current_file_progress = (p / 100) * (100 / total_files)
                        overall_p = int(base_progress + current_file_progress)
                        if overall_p >= 100 and completed_files[0] < total_files - 1:
                            overall_p = 99
                        progress_store[task_id] = overall_p

            pages_with_uncropped = score_api.process_file_to_1in1(temp_path, score_api.DEFAULT_CONFIG, debug_out_dir=TEMP_DEBUG_DIR, progress_callback=progress_cb)
            for page, uncropped in pages_with_uncropped:
                unique_filename = f"{uuid.uuid4().hex}.png"
                preview_path = os.path.join(TEMP_PREVIEW_DIR, unique_filename)
                page.save(preview_path, optimize=True)

                uncropped_path = os.path.join(TEMP_UNCROPPED_DIR, unique_filename)
                cv2.imwrite(uncropped_path, uncropped)

                local_preview_filenames.append(unique_filename)

            with progress_lock:
                completed_files[0] += 1

            return i, local_preview_filenames

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_single_file, i, temp_path) for i, temp_path in enumerate(file_paths)]
            for future in concurrent.futures.as_completed(futures):
                idx, local_preview_filenames = future.result()
                preview_filenames_list[idx] = local_preview_filenames

        preview_filenames = [filename for sublist in preview_filenames_list for filename in sublist]
                
        piece_guess, inst_guess = "", ""
        if first_file_path:
            piece_guess, inst_guess = extract_info_from_header(first_file_path)
        print(f"success convertion (task #{task_id})")
        result_store[task_id] = {
            "success": True,
            "preview_filenames": preview_filenames,
            "piece_guess": piece_guess,
            "inst_guess": inst_guess
        }
    except Exception as e:
        result_store[task_id] = {
            "success": False,
            "error": str(e)
        }
    finally:
        progress_store[task_id] = 100

@app.route('/process', methods=['POST'])
def process_files():
    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({"error": "No files provided"}), 400

    task_id = request.form.get('task_id')
    if not task_id:
        task_id = uuid.uuid4().hex

    progress_store[task_id] = 0

    clear_temp_dir(TEMP_UPLOAD_DIR)
    clear_temp_dir(TEMP_PREVIEW_DIR)
    clear_temp_dir(TEMP_UNCROPPED_DIR)

    file_paths = []
    for file in files:
        if file.filename == '': continue
        temp_path = os.path.join(TEMP_UPLOAD_DIR, file.filename)
        file.save(temp_path)
        file_paths.append(temp_path)

    thread = threading.Thread(target=process_images_in_background, args=(task_id, file_paths))
    thread.start()
    print(progress_store)
    return jsonify({"task_id": task_id, "status": "processing"})


@app.route('/preview/<task_id>')
def preview_result_new(task_id : str):
    return preview_result(task_id)

@app.route('/preview_result')
def preview_result_old():
    task_id = request.args.get('task_id')
    return preview_result(task_id)

@app.route('/queue/<next>/<task_id>')
def Queue(next : str,task_id : str):
    # リダイレクト先URLを生成
    target_url = url_for(next, task_id=task_id)
    return render_template('queue.html', target_url=target_url)

def preview_result(task_id,isFirst = False):
    if not task_id or not(task_id in result_store):
        print(result_store)
        flash(f'無効なタスクIDです。{task_id}')
        if isFirst:
            return redirect(url_for('Queue',next = "preview_result_new",task_id = task_id))
        else:
            return redirect(url_for('index'))

    result = result_store.pop(task_id)
    if not result.get("success"):
        flash(f'処理エラー: {result.get("error")}')
        return redirect(url_for('index'))

    return render_template('preview.html', previews=result["preview_filenames"], piece_guess=result["piece_guess"], inst_guess=result["inst_guess"],
                           piece_names=score_api.get_unique_piece_names(), event_names=score_api.get_unique_event_names(),
                           composers=score_api.get_unique_composers_arrangers()[0], arrangers=score_api.get_unique_composers_arrangers()[1],
                           current_year=datetime.datetime.now().year, drive_service=drive_service)
    
@app.route('/scan_ui', methods=['POST'])
def scan_ui():
    return render_template('scan.html', device_name=request.form.get('device_name', ''), scanned_files=[])

@app.route('/scan_execute', methods=['POST'])
def scan_execute():
    device_name = request.form.get('device_name', '')
    scanned_files = request.form.getlist('scanned_files[]')
    task_id = request.form.get('task_id')

    if task_id:
        progress_store[task_id] = 0

    try:
        temp_scan_path = os.path.join(TEMP_UPLOAD_DIR, f"scanned_{uuid.uuid4().hex}.png")
        score_api.scan_score_from_epson(temp_scan_path, dpi=score_api.DEFAULT_CONFIG['dpi'], device_name=device_name if device_name else None)
        
        def progress_cb(p):
            if task_id:
                overall_p = int(p)
                if overall_p >= 100:
                    overall_p = 99
                progress_store[task_id] = overall_p

        # デバッグディレクトリを指定して処理を実行
        pages_with_uncropped = score_api.process_file_to_1in1(temp_scan_path, score_api.DEFAULT_CONFIG, debug_out_dir=TEMP_DEBUG_DIR, progress_callback=progress_cb)
        for page, uncropped in pages_with_uncropped:
            unique_filename = f"{uuid.uuid4().hex}.png"
            preview_path = os.path.join(TEMP_PREVIEW_DIR, unique_filename)
            page.save(preview_path, optimize=True)

            uncropped_path = os.path.join(TEMP_UNCROPPED_DIR, unique_filename)
            cv2.imwrite(uncropped_path, uncropped)

            scanned_files.append(unique_filename)
        return render_template('scan.html', device_name=device_name, scanned_files=scanned_files)
    except Exception as e:
        flash(f'スキャンエラー: {str(e)}')
        return render_template('scan.html', device_name=device_name, scanned_files=scanned_files)
    finally:
        if task_id:
            progress_store[task_id] = 100

@app.route('/scan_to_preview', methods=['POST'])
def scan_to_preview():
    scanned_files = request.form.getlist('scanned_files[]')
    if not scanned_files: return redirect(url_for('index'))
    first_file_path = os.path.join(TEMP_PREVIEW_DIR, scanned_files[0])
    piece_guess, inst_guess = extract_info_from_header(first_file_path)
    
    return render_template('preview.html', previews=scanned_files, piece_guess=piece_guess, inst_guess=inst_guess,
                           piece_names=score_api.get_unique_piece_names(), event_names=score_api.get_unique_event_names(),
                           composers=score_api.get_unique_composers_arrangers()[0], arrangers=score_api.get_unique_composers_arrangers()[1],
                           current_year=datetime.datetime.now().year,drive_service=drive_service)

@app.route('/update_order', methods=['POST'])
def update_order():
    piece = request.form.get('piece', '')
    instrument = request.form.get('instrument', '')
    filenames = request.form.getlist('filenames[]')
    orders = request.form.getlist('orders[]')
    try:
        paired = [(int(o), f) for f, o in zip(filenames, orders)]
        paired.sort(key=lambda x: x[0])
        sorted_filenames = [f for _, f in paired]
        flash('ページの順番を更新しました。')
        return render_template('preview.html', previews=sorted_filenames, piece_guess=piece, inst_guess=instrument,
                               piece_names=score_api.get_unique_piece_names(), event_names=score_api.get_unique_event_names(),
                               composers=score_api.get_unique_composers_arrangers()[0], arrangers=score_api.get_unique_composers_arrangers()[1],
                               current_year=datetime.datetime.now().year,drive_service=drive_service)
    except ValueError:
        flash('順序には数値を入力してください。')
        return render_template('preview.html', previews=filenames, piece_guess=piece, inst_guess=instrument,
                               piece_names=score_api.get_unique_piece_names(), event_names=score_api.get_unique_event_names(),
                               composers=score_api.get_unique_composers_arrangers()[0], arrangers=score_api.get_unique_composers_arrangers()[1],
                               current_year=datetime.datetime.now().year,drive_service=drive_service)

@app.route('/api/get_profiles')
def api_get_profiles():
    piece = request.args.get('piece', '')
    return jsonify(score_api.get_profiles_by_piece(piece))

@app.route('/save', methods=['POST'])
def save_score():
    piece = request.form.get('piece')
    instrument = request.form.get('instrument')
    year = request.form.get('year')
    event_name = request.form.get('event_name')
    preview_filenames = request.form.getlist('previews')
    save_mode = request.form.get('save_mode')

    if not save_mode or not piece or not instrument:
        flash('必須項目が入力されていません。')
        return redirect(url_for('index'))

    if save_mode == 'new' and (not year or not event_name):
        flash('必須項目が入力されていません。')
        return redirect(url_for('index'))

    score_id = None
    composer = ""
    arranger = ""

    if save_mode == 'new':
        composer = request.form.get('new_composer', '')
        arranger = request.form.get('new_arranger', '')
    elif save_mode.startswith('existing_'):
        idx = save_mode.split('_')[1]
        score_id = request.form.get(f'ex_id_{idx}')

    try:
        pages = [Image.open(os.path.join(TEMP_PREVIEW_DIR, fname)) for fname in preview_filenames]
        saved_dir, saved_score_id = score_api.save_and_register_score(pages, year, event_name, piece, composer, arranger, instrument, score_id=score_id)
        
        print(f"drive_service : {drive_service}")
        print(f"GOOGLE_DRIVE_FOLDER_ID : {GOOGLE_DRIVE_FOLDER_ID}")

        # Google Drive にアップロード（オプション）
        if drive_service and GOOGLE_DRIVE_FOLDER_ID:
            # 既存のイベントからフォルダを特定するか、新規作成する
            # year / event_name が提供されていない場合、直近のイベントを利用する
            actual_year = year
            actual_event_name = event_name
            if not actual_year or not actual_event_name:
                db = score_api.load_db()
                if saved_score_id in db and db[saved_score_id].get('events'):
                    # events はソートされている前提 (または最新のものを使う)
                    latest_event = db[saved_score_id]['events'][0]
                    actual_year = latest_event.get('year', '')
                    actual_event_name = latest_event.get('event_name', '')

            # Google Drive上に作る階層構造をリストで定義します
            event_dir_name = f"{actual_year}{actual_event_name}"
            path_components = [event_dir_name, piece, instrument]

            # 再帰的にフォルダを確認・生成して、保存先のフォルダIDを取得
            target_folder_id = firebase_db.get_or_create_drive_path(
                drive_service, GOOGLE_DRIVE_FOLDER_ID, path_components
            )

            urls = []
            if target_folder_id:
                token = session.get('google_token')
                urls = firebase_db.upload_score_pages_to_google_drive(
                    saved_dir, saved_score_id, instrument,
                    token, target_folder_id
                )
            # Firebase にも URL を記録
            if urls:
                db = score_api.load_db()
                if saved_score_id in db:
                    if 'instruments' in db[saved_score_id]:
                         db[saved_score_id]['instruments'][instrument] = urls
                         score_api.save_db(db)
        
        flash(f'「{piece}」({instrument}) の登録・追加が完了しました！')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'保存エラー: {str(e)}')
        return redirect(url_for('index'))

@app.route('/list')
def score_list():
    sort_by = request.args.get('sort', 'event')
    if sort_by == 'piece': 
        pieces_list = score_api.get_all_scores_by_piece()
        return render_template('list.html', pieces_list=pieces_list, sort_by=sort_by)
    else: 
        grouped_data = score_api.get_all_scores_grouped()
        return render_template('list.html', grouped_data=grouped_data, sort_by=sort_by)

@app.route('/search', methods=['GET'])
def search_score():
    keyword = request.args.get('keyword', '')
    if not keyword: return redirect(url_for('index'))
    results = score_api.search_pieces_by_keyword(keyword)
    return render_template('list.html', search_results=results, keyword=keyword)

@app.route('/piece')
def piece_details():
    score_id = request.args.get('id', '').strip()
    if not score_id: return redirect(url_for('score_list'))
    
    details = score_api.get_piece_details(score_id)
    if not details: 
        flash(f'システムエラー: 楽譜データが見つかりませんでした。')
        return redirect(url_for('score_list'))
    
    return render_template('piece.html', details=details, event_names=score_api.get_unique_event_names(), current_year=datetime.datetime.now().year)


def download_from_drive_if_missing(score_id, instrument, details):
    """【セキュリティ＆安定性両立版】Driveから必要な分だけ安全に直列ダウンロードする"""
    target_dir = None
    urls = None
    for inst in details['instruments']:
        if inst['name'] == instrument:
            if 'urls' in inst:
                urls = inst['urls']
            else:
                target_dir = inst.get('dir')
            break

    if urls is not None:
        target_dir = os.path.join("score_data", score_id, instrument)
        os.makedirs(target_dir, exist_ok=True)
        
        # すでにローカル（サーバー内）に保存されているファイル数をカウント
        image_files = sorted(glob.glob(os.path.join(target_dir, "*.png")))

        # 枚数が一致しない場合（未ダウンロード、または不足がある場合）のみDriveから取得
        if len(image_files) != len(urls):
            token = session.get('google_token')
            
            # グローバル共有を避け、このユーザー要求のためだけの ds を毎回生成（SAトークンの衝突防止）
            ds = firebase_db.initialize_google_drive(token)

            print(f"🔒 サーバー内に安全にダウンロード中... (全 {len(urls)} ページ)")
            
            # 並列ではなく、1枚ずつ直列で確実にダウンロード
            for i, url in enumerate(urls):
                filename = f"{score_id}_{instrument}_page_{i + 1:03d}.png"
                filepath = os.path.join(target_dir, filename)
                
                if not os.path.exists(filepath):
                    try:
                        parsed = urllib.parse.urlparse(url)
                        file_id = urllib.parse.parse_qs(parsed.query).get('id', [None])[0]
                        
                        if file_id and ds:
                            request_obj = ds.files().get_media(fileId=file_id, supportsAllDrives=True)
                            
                            # with 構文で1枚ごとにバイナリメモリを強制解放
                            with io.BytesIO() as fh:
                                downloader = MediaIoBaseDownload(fh, request_obj)
                                done = False
                                while done is False:
                                    status, done = downloader.next_chunk()
                                
                                with open(filepath, 'wb') as f:
                                    f.write(fh.getvalue())
                                    
                            print(f"  -> Cache completed: {filename}")
                    except Exception as e:
                        print(f"  -> Error caching page {i+1}: {e}")
            
            # メモリを徹底的に掃除
            import gc
            gc.collect()

    elif not target_dir:
         target_dir = os.path.join("score_data", score_id, instrument)

    return target_dir, urls

@app.route('/image/404')
def get_image():
    IMAGE_FOLDER = os.path.join(app.root_path, 'static')
    return send_from_directory(IMAGE_FOLDER, '404.jpg')

@app.route('/support/printer')
def GetPrinterState():
    return score_api.get_printer_dict()

@app.route('/view_score')
def view_score():
    score_id = request.args.get('id', '').strip()
    instrument = request.args.get('instrument', '').strip()
    if not score_id or not instrument: return redirect(url_for('score_list'))

    details = score_api.get_piece_details(score_id)
    if not details: return redirect(url_for('score_list'))

    filenames,target_dir, urlsTmp = GetImagePathes(score_id, instrument, details)
    
    printers = GetPrinterState()
    if len(printers.keys()) == 0:
        printers = None

    urls = [url if url is not None else url_for("get_image") for url in urlsTmp]

    return render_template('view_score.html',
                           printer=printers,
                           details=details, 
                           instrument=instrument, 
                           filenames=filenames, 
                           target_dir=target_dir, 
                           has_urls=urls is not None)


def GetImagePathes(score_id, instrument, details, scoreOrder = None):
    target_dir, urls = download_from_drive_if_missing(score_id, instrument, details)

    if not target_dir: return redirect(url_for('piece_details', id=score_id))
    
    image_files = sorted(glob.glob(os.path.join(target_dir, "*.png")))
    filenames = [os.path.basename(f) for f in image_files]
    
    if scoreOrder:
        filenamesTmp = filenames.copy()
        filenames = [filenamesTmp[i] for i in scoreOrder]

    return filenames,target_dir, urls

@app.route('/score_image/<score_id>/<instrument>/<filename>')
def score_image(score_id, instrument, filename):
    details = score_api.get_piece_details(score_id)
    if not details: return "Not found", 404
    
    target_dir = None
    for inst in details['instruments']:
        if inst['name'] == instrument:
            if 'dir' in inst:
                target_dir = inst['dir']
            elif 'urls' in inst:
                target_dir = os.path.join("score_data", score_id, instrument)
            break
            
    if not target_dir or not os.path.exists(os.path.join(target_dir, filename)):
        return "Not found", 404
        
    return send_file(os.path.join(target_dir, filename))


@app.route('/upload_to_drive', methods=['POST'])
def upload_to_drive():
    score_id = request.form.get('score_id')
    instrument = request.form.get('instrument')

    if not score_id or not instrument:
        flash('必要なパラメータがありません。')
        return redirect(url_for('score_list'))

    token = session.get('google_token')
    ds = get_drive_service(token)
    if not ds or not GOOGLE_DRIVE_FOLDER_ID:
        flash('Google Drive にログインしていません。')
        return redirect(url_for('view_score', id=score_id, instrument=instrument))

    details = score_api.get_piece_details(score_id)
    if not details:
        flash('楽譜が見つかりません。')
        return redirect(url_for('score_list'))

    target_dir = None
    for inst in details['instruments']:
        if inst['name'] == instrument:
            target_dir = inst.get('dir')
            if 'urls' in inst:
                flash('すでにGoogle Driveにアップロードされています。')
                return redirect(url_for('view_score', id=score_id, instrument=instrument))
            break

    if not target_dir or not os.path.exists(target_dir):
        flash('ローカルの楽譜ファイルが見つかりません。')
        return redirect(url_for('view_score', id=score_id, instrument=instrument))

    actual_year = details['events'][0]['year'] if details.get('events') else str(datetime.datetime.now().year)
    actual_event_name = details['events'][0]['event_name'] if details.get('events') else '未設定行事'
    piece_name = details.get('piece', 'UnknownPiece')

    event_dir_name = f"{actual_year}{actual_event_name}"
    path_components = [event_dir_name, piece_name, instrument]

    target_folder_id = firebase_db.get_or_create_drive_path(ds, GOOGLE_DRIVE_FOLDER_ID, path_components)

    urls = []
    if target_folder_id:
        token = session.get('google_token')
        urls = firebase_db.upload_score_pages_to_google_drive(
            target_dir, score_id, instrument,
            token, target_folder_id
        )

    if urls:
        db = score_api.load_db()
        if score_id in db and 'instruments' in db[score_id] and instrument in db[score_id]['instruments']:
            db[score_id]['instruments'][instrument] = urls
            score_api.save_db(db)
            flash(f'「{piece_name}」({instrument}) をGoogle Driveへアップロードしました。')
    else:
        flash('アップロードに失敗しました。')

    return redirect(url_for('view_score', id=score_id, instrument=instrument))

@app.route('/update_db_order', methods=['POST'])
def update_db_order():
    score_id = request.form.get('score_id')
    instrument = request.form.get('instrument')
    filenames = request.form.getlist('filenames[]')
    orders = request.form.getlist('orders[]')

    if not score_id or not instrument:
        flash('必要な情報が不足しています。')
        return redirect(url_for('score_list'))

    try:
        # Sort based on specified orders
        paired = [(int(o), f) for f, o in zip(filenames, orders)]
        paired.sort(key=lambda x: x[0])
        sorted_filenames = [f for _, f in paired]

        db = score_api.load_db()
        if score_id in db:
            score = db[score_id]
            if 'instruments' in score and instrument in score['instruments']:
                target_dir = score['instruments'][instrument]

                if isinstance(target_dir, list):
                    # It's a list of URLs
                    urls_map = {f: url for f, url in zip(filenames, target_dir)}
                    new_urls = [urls_map[f] for f in sorted_filenames]
                    score['instruments'][instrument] = new_urls
                    score_api.save_db(db)

                    # Also rename local cached files if they exist
                    local_dir = os.path.join("score_data", score_id, instrument)
                    if os.path.exists(local_dir):
                        # Create temporary mapping
                        temp_map = {}
                        for i, old_fname in enumerate(sorted_filenames):
                            secure_old_fname = secure_filename(old_fname)
                            ext = os.path.splitext(secure_old_fname)[1] or '.png'
                            old_path = os.path.join(local_dir, secure_old_fname)
                            if os.path.exists(old_path):
                                temp_path = os.path.join(local_dir, f"temp_{uuid.uuid4().hex}{ext}")
                                os.rename(old_path, temp_path)
                                temp_map[temp_path] = f"{score_id}_{instrument}_page_{i + 1:03d}{ext}"

                        # Rename to final names
                        for temp_path, new_fname in temp_map.items():
                            new_path = os.path.join(local_dir, new_fname)
                            os.rename(temp_path, new_path)

                else:
                    # It's a local directory path
                    local_dir = target_dir
                    if os.path.exists(local_dir):
                        # Create temporary mapping
                        temp_map = {}
                        for i, old_fname in enumerate(sorted_filenames):
                            secure_old_fname = secure_filename(old_fname)
                            ext = os.path.splitext(secure_old_fname)[1] or '.png'
                            old_path = os.path.join(local_dir, secure_old_fname)
                            if os.path.exists(old_path):
                                temp_path = os.path.join(local_dir, f"temp_{uuid.uuid4().hex}{ext}")
                                os.rename(old_path, temp_path)
                                temp_map[temp_path] = f"page_{i + 1:03d}{ext}"

                        # Rename to final names
                        for temp_path, new_fname in temp_map.items():
                            new_path = os.path.join(local_dir, new_fname)
                            os.rename(temp_path, new_path)

        flash('ページの順番を更新しました。')
    except ValueError:
        flash('順序には数値を入力してください。')
    except Exception as e:
        flash(f'順番の更新中にエラーが発生しました: {str(e)}')

    return redirect(url_for('view_score', id=score_id, instrument=instrument))

@app.route('/delete_db_page', methods=['POST'])
def delete_db_page():
    score_id = request.form.get('score_id')
    instrument = request.form.get('instrument')
    filename_to_delete = request.form.get('filename')

    if not score_id or not instrument or not filename_to_delete:
        flash('必要な情報が不足しています。')
        return redirect(url_for('score_list'))

    try:
        secure_target_fname = secure_filename(filename_to_delete)
        db = score_api.load_db()
        if score_id in db:
            score = db[score_id]
            if 'instruments' in score and instrument in score['instruments']:
                target_dir = score['instruments'][instrument]

                if isinstance(target_dir, list):
                    # It's a list of URLs
                    # Remove the URL that corresponds to the filename we are deleting
                    # We know the index of the file in the local cache matches the index in the URL list
                    local_dir = os.path.join("score_data", score_id, instrument)
                    if os.path.exists(local_dir):
                        all_files = sorted(glob.glob(os.path.join(local_dir, "*.png")))
                        filenames_only = [os.path.basename(f) for f in all_files]

                        if secure_target_fname in filenames_only:
                            index_to_remove = filenames_only.index(secure_target_fname)

                            if index_to_remove < len(target_dir):
                                target_dir.pop(index_to_remove)
                                score_api.save_db(db)

                            # Delete local file
                            target_path = os.path.join(local_dir, secure_target_fname)
                            if os.path.exists(target_path):
                                os.remove(target_path)

                else:
                    # It's a local directory path
                    local_dir = target_dir
                    if os.path.exists(local_dir):
                        target_path = os.path.join(local_dir, secure_target_fname)
                        if os.path.exists(target_path):
                            os.remove(target_path)

        flash(f'ページ ({secure_target_fname}) を削除しました。')
    except Exception as e:
        flash(f'ページの削除中にエラーが発生しました: {str(e)}')

    return redirect(url_for('view_score', id=score_id, instrument=instrument))


@app.route('/update_piece_info', methods=['POST'])
def update_piece_info():
    score_id = request.form.get('score_id')
    composer = request.form.get('composer', '')
    arranger = request.form.get('arranger', '')
    success = score_api.update_composer_arranger(score_id, composer, arranger)
    if success: flash('📝 作曲者・編曲者の情報を更新しました。')
    else: flash('エラー: 情報の更新に失敗しました。')
    return redirect(url_for('piece_details', id=score_id))

@app.route('/add_event', methods=['POST'])
def add_event():
    score_id = request.form.get('score_id')
    dest_year = request.form.get('dest_year')
    dest_event = request.form.get('dest_event')

    if not dest_year or not dest_event:
        flash('追加先の年度と演奏会名を入力してください。')
        return redirect(url_for('piece_details', id=score_id))
    try:
        success = score_api.add_event_to_score(score_id, dest_year, dest_event)
        if success:
            flash(f'この楽譜を {dest_year}{dest_event} の行事に追加（リンク）しました！')
        else:
            flash('共有元のデータが見つかりませんでした。')
        return redirect(url_for('piece_details', id=score_id))
    except Exception as e:
        flash(f'処理中にエラーが発生しました: {str(e)}')
        return redirect(url_for('piece_details', id=score_id))

@app.route('/output_action', methods=['POST'])
def output_action():
    directory = request.form.get('directory')
    mode = request.form.get('mode')
    action_type = request.form.get('action_type')
    printer = request.form.get('printer', '')
    score_id = request.form.get('score_id')
    piece = request.form.get('piece', 'score')
    inst = request.form.get('instrument', 'inst')

    details = score_api.get_piece_details(score_id)
    if details:
        target_dir, urls = download_from_drive_if_missing(score_id, inst, details)
        if target_dir:
            directory = target_dir

    if not directory or not os.path.exists(directory):
        flash('エラー: 指定されたデータが見つかりません。')
        return redirect(url_for('piece_details', id=score_id))
    try:
        if action_type == 'print':
            score_api.layout_and_print_score(directory=directory, mode=mode, orientation=score_api.DEFAULT_CONFIG['page_orientation'], printer_name=printer if printer else None, dpi=score_api.DEFAULT_CONFIG['dpi'])
            flash(f'[{piece} - {inst}] の印刷ジョブを送信しました！')
            if request.form.get('from_view'):
                return redirect(url_for('view_score', id=score_id, instrument=inst))
            return redirect(url_for('piece_details', id=score_id))
            
        output_pages = score_api.apply_layout(directory=directory, mode=mode, orientation=score_api.DEFAULT_CONFIG['page_orientation'], booklet_dir=score_api.DEFAULT_CONFIG['booklet_direction'], dpi=score_api.DEFAULT_CONFIG['dpi'])
        safe_filename = f"{piece}_{inst}_{mode}".replace(' ', '_')
        
        if action_type == 'pdf':
            pdf_io = io.BytesIO()
            output_pages[0].save(pdf_io, save_all=True, append_images=output_pages[1:], format='PDF', resolution=score_api.DEFAULT_CONFIG['dpi'])
            pdf_io.seek(0)
            return send_file(pdf_io, as_attachment=True, download_name=f"{safe_filename}.pdf", mimetype='application/pdf')
        elif action_type == 'zip':
            zip_io = io.BytesIO()
            with zipfile.ZipFile(zip_io, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
                for i, page in enumerate(output_pages):
                    img_io = io.BytesIO()
                    page.save(img_io, format='PNG', optimize=True)
                    zf.writestr(f"{safe_filename}_{i+1:03d}.png", img_io.getvalue())
            zip_io.seek(0)
            return send_file(zip_io, as_attachment=True, download_name=f"{safe_filename}.zip", mimetype='application/zip')
    except Exception as e:
        flash(f'出力エラー: {str(e)}')
        return redirect(url_for('piece_details', id=score_id))

# ===== ▼ 新規追加: デバッグ画面表示API ▼ =====
@app.route('/debug')
def debug_view():
    debug_files = sorted(glob.glob(os.path.join(TEMP_DEBUG_DIR, "*.jpg")), reverse=True)
    filenames = [os.path.basename(f) for f in debug_files]
    return render_template('debug.html', debug_images=filenames)

@app.route('/re_crop', methods=['POST'])
def re_crop():
    filename = request.form.get('filename')
    try:
        x = float(request.form.get('x'))
        y = float(request.form.get('y'))
        w = float(request.form.get('w'))
        h = float(request.form.get('h'))
    except (TypeError, ValueError):
        return jsonify({'success': False, 'error': '無効な座標です'}), 400

    if not filename:
        return jsonify({'success': False, 'error': 'ファイル名が指定されていません'}), 400

    uncropped_filepath = os.path.join(TEMP_UNCROPPED_DIR, filename)
    preview_filepath = os.path.join(TEMP_PREVIEW_DIR, filename)

    if not os.path.exists(uncropped_filepath):
        return jsonify({'success': False, 'error': '元の画像が見つかりません'}), 404

    try:
        uncropped_img = cv2.imread(uncropped_filepath, cv2.IMREAD_GRAYSCALE)
        if uncropped_img is None:
            return jsonify({'success': False, 'error': '元の画像を読み込めません'}), 500

        cropped_pil = score_api.crop_margins_and_fit(uncropped_img, score_api.DEFAULT_CONFIG, manual_crop=(x, y, w, h))
        cropped_pil.save(preview_filepath, optimize=True)

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/rotate_image', methods=['POST'])
def rotate_image():
    filename = request.form.get('filename')
    direction = request.form.get('direction')
    
    if not filename or not direction:
        return jsonify({'success': False, 'error': 'パラメータが不足しています'}), 400
        
    filepath = os.path.join(TEMP_PREVIEW_DIR, filename)
    uncropped_filepath = os.path.join(TEMP_UNCROPPED_DIR, filename)

    if not os.path.exists(filepath):
        return jsonify({'success': False, 'error': 'ファイルが見つかりません'}), 404
        
    try:
        img = Image.open(filepath)
        if direction == 'left':
            img = img.transpose(Image.ROTATE_90)
        elif direction == 'right':
            img = img.transpose(Image.ROTATE_270)
        elif direction == '180':
            img = img.transpose(Image.ROTATE_180)
            
        img.save(filepath, optimize=True)

        if os.path.exists(uncropped_filepath):
            u_img = Image.open(uncropped_filepath)
            if direction == 'left':
                u_img = u_img.transpose(Image.ROTATE_90)
            elif direction == 'right':
                u_img = u_img.transpose(Image.ROTATE_270)
            elif direction == '180':
                u_img = u_img.transpose(Image.ROTATE_180)
            u_img.save(uncropped_filepath)

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    
if not os.environ.get('isDebug'):
    app.config['PREFERRED_URL_SCHEME'] = 'https'
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
else:
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

if __name__ == '__main__':
    ssl = None;
    if os.environ.get('isDebug'):
        print("⚠️ デバッグモードで起動しています。セキュリティに注意してください。")
        os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
    else:
        app.config['PREFERRED_URL_SCHEME'] = 'https'
        app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
        if os.environ.get("SSH_CERT"):
            ssl = (os.environ["SSH_CERT"], os.environ["SSH_KEY"])
            print("🔒 SSL/TLS 証明書を使用して起動します。")

    app.run(host='0.0.0.0', port=5000, debug=True, ssl_context=ssl)
