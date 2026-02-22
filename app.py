from flask import Flask, render_template, request, redirect, url_for, flash
import os
import glob
import uuid
from PIL import Image
import cv2
import pytesseract
import score_api

app = Flask(__name__)
app.secret_key = 'score_processor_secret_key'

TEMP_UPLOAD_DIR = os.path.join('static', 'temp', 'uploads')
TEMP_PREVIEW_DIR = os.path.join('static', 'temp', 'previews')
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMP_PREVIEW_DIR, exist_ok=True)

def clear_temp_dir(directory):
    for f in glob.glob(os.path.join(directory, '*')):
        try: os.remove(f)
        except: pass

def extract_info_from_header(image_path):
    """
    画像の上部15%を切り出してOCRにかけ、タイトルや楽器名を推測する。
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return "", ""
        
        h, w = img.shape
        # 憶測: 上部15%にタイトル等の情報がある
        header_img = img[0:int(h * 0.15), 0:w]
        
        # OCR精度を上げるための二値化
        _, thresh = cv2.threshold(header_img, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # 英語モードでテキスト抽出 (楽譜のタイトルはアルファベットが多いという憶測)
        text = pytesseract.image_to_string(thresh, lang='eng').strip()
        
        # 抽出したテキストを改行で分割し、空行を除去
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        piece_guess = lines[0] if len(lines) > 0 else ""
        inst_guess = lines[1] if len(lines) > 1 else ""
        
        # 特殊文字を少しクリーニング
        piece_guess = "".join(c for c in piece_guess if c.isalnum() or c in " -_")
        inst_guess = "".join(c for c in inst_guess if c.isalnum() or c in " -_")
        
        return piece_guess, inst_guess
    except Exception as e:
        print(f"OCR Error: {e}")
        return "", ""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_files():
    if 'files' not in request.files:
        flash('ファイルが選択されていません。')
        return redirect(url_for('index'))

    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        flash('ファイルが選択されていません。')
        return redirect(url_for('index'))

    clear_temp_dir(TEMP_UPLOAD_DIR)
    clear_temp_dir(TEMP_PREVIEW_DIR)
    
    preview_filenames = []
    first_file_path = None
    
    try:
        for i, file in enumerate(files):
            if file.filename == '': continue
            
            temp_path = os.path.join(TEMP_UPLOAD_DIR, file.filename)
            file.save(temp_path)
            
            if i == 0:
                first_file_path = temp_path
            
            pages = score_api.process_file_to_1in1(temp_path, score_api.DEFAULT_CONFIG)
            
            for page in pages:
                unique_filename = f"{uuid.uuid4().hex}.png"
                preview_path = os.path.join(TEMP_PREVIEW_DIR, unique_filename)
                page.save(preview_path, optimize=True)
                preview_filenames.append(unique_filename)
                
        # 最初のページからOCRで文字を推測
        piece_guess, inst_guess = "", ""
        if first_file_path:
            piece_guess, inst_guess = extract_info_from_header(first_file_path)
            
        return render_template('preview.html', previews=preview_filenames, piece_guess=piece_guess, inst_guess=inst_guess)
        
    except Exception as e:
        flash(f'処理エラー: {str(e)}')
        return redirect(url_for('index'))

@app.route('/update_order', methods=['POST'])
def update_order():
    """プレビュー画面で順番変更ボタンが押されたときの処理"""
    piece = request.form.get('piece', '')
    instrument = request.form.get('instrument', '')
    filenames = request.form.getlist('filenames[]')
    orders = request.form.getlist('orders[]')
    
    try:
        # 入力された順序(数字)とファイル名のペアを作り、数字の昇順でソート
        paired = []
        for f, o in zip(filenames, orders):
            paired.append((int(o), f))
            
        paired.sort(key=lambda x: x[0])
        sorted_filenames = [f for _, f in paired]
        
        flash('ページの順番を更新しました。')
        # 入力途中の楽曲名等も維持して再描画
        return render_template('preview.html', previews=sorted_filenames, piece_guess=piece, inst_guess=instrument)
    except ValueError:
        flash('順序には数値を入力してください。')
        return render_template('preview.html', previews=filenames, piece_guess=piece, inst_guess=instrument)

@app.route('/save', methods=['POST'])
def save_score():
    piece = request.form.get('piece')
    instrument = request.form.get('instrument')
    preview_filenames = request.form.getlist('previews')
    
    if not piece or not instrument:
        flash('楽曲名と楽器名を入力してください。')
        return render_template('preview.html', previews=preview_filenames, piece_guess=piece, inst_guess=instrument, error="必須項目です")
        
    try:
        pages = []
        for filename in preview_filenames:
            local_path = os.path.join(TEMP_PREVIEW_DIR, filename)
            pages.append(Image.open(local_path))
            
        saved_dir = score_api.save_and_register_score(pages, piece, instrument)
        flash(f'登録が完了しました！ [ 保存先: {saved_dir} ]')
        return redirect(url_for('index'))
        
    except Exception as e:
        flash(f'保存エラー: {str(e)}')
        return redirect(url_for('index'))

@app.route('/print', methods=['POST'])
def print_score():
    # 既存のprint_scoreの実装そのまま
    piece = request.form.get('piece')
    instrument = request.form.get('instrument')
    mode = request.form.get('mode', 'booklet')
    printer = request.form.get('printer', '')
    
    if not piece or not instrument:
        flash('印刷対象の楽曲名と楽器名を入力してください。')
        return redirect(url_for('index'))
        
    try:
        target_dir = score_api.search_score_directory(piece, instrument)
        if not target_dir:
            flash(f'エラー: 対象の楽譜 [{piece} - {instrument}] が見つかりません。')
            return redirect(url_for('index'))
            
        printer_name = printer if printer else None
        score_api.layout_and_print_score(
            directory=target_dir,
            mode=mode,
            orientation=score_api.DEFAULT_CONFIG['page_orientation'],
            printer_name=printer_name,
            dpi=score_api.DEFAULT_CONFIG['dpi']
        )
        flash('印刷ジョブをプリンタに送信しました！')
        
    except Exception as e:
        flash(f'印刷エラー: {str(e)}')
        
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)