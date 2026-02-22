from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import os
import glob
import uuid
import datetime
import time
from PIL import Image
import cv2
import pytesseract
import zipfile
import io
import score_api

app = Flask(__name__)
app.secret_key = 'score_processor_secret_key'

TEMP_UPLOAD_DIR = os.path.join('static', 'temp', 'uploads')
TEMP_PREVIEW_DIR = os.path.join('static', 'temp', 'previews')
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMP_PREVIEW_DIR, exist_ok=True)

def clear_temp_dir(directory, max_age_hours=1):
    """複数回のスキャン中にファイルが消えないよう、1時間以上古いファイルのみ削除する"""
    now = time.time()
    for f in glob.glob(os.path.join(directory, '*')):
        try:
            if os.stat(f).st_mtime < now - max_age_hours * 3600:
                os.remove(f)
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
    except:
        return "", ""

@app.route('/')
def index():
    clear_temp_dir(TEMP_UPLOAD_DIR)
    clear_temp_dir(TEMP_PREVIEW_DIR)
    event_names = score_api.get_unique_event_names()
    current_year = datetime.datetime.now().year
    return render_template('index.html', event_names=event_names, current_year=current_year)

@app.route('/process', methods=['POST'])
def process_files():
    if 'files' not in request.files:
        flash('ファイルが選択されていません。')
        return redirect(url_for('index'))
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        flash('ファイルが選択されていません。')
        return redirect(url_for('index'))

    year = request.form.get('year', '')
    event_name = request.form.get('event_name', '')

    preview_filenames = []
    first_file_path = None
    
    try:
        for i, file in enumerate(files):
            if file.filename == '': continue
            temp_path = os.path.join(TEMP_UPLOAD_DIR, file.filename)
            file.save(temp_path)
            if i == 0: first_file_path = temp_path
            
            pages = score_api.process_file_to_1in1(temp_path, score_api.DEFAULT_CONFIG)
            for page in pages:
                unique_filename = f"{uuid.uuid4().hex}.png"
                preview_path = os.path.join(TEMP_PREVIEW_DIR, unique_filename)
                page.save(preview_path, optimize=True)
                preview_filenames.append(unique_filename)
                
        piece_guess, inst_guess = "", ""
        if first_file_path:
            piece_guess, inst_guess = extract_info_from_header(first_file_path)
            
        piece_names = score_api.get_unique_piece_names()
            
        return render_template('preview.html', previews=preview_filenames, 
                               year=year, event_name=event_name,
                               piece_guess=piece_guess, inst_guess=inst_guess,
                               piece_names=piece_names)
    except Exception as e:
        flash(f'処理エラー: {str(e)}')
        return redirect(url_for('index'))

@app.route('/update_order', methods=['POST'])
def update_order():
    year = request.form.get('year', '')
    event_name = request.form.get('event_name', '')
    piece = request.form.get('piece', '')
    instrument = request.form.get('instrument', '')
    device_name = request.form.get('device_name', '')
    filenames = request.form.getlist('filenames[]')
    orders = request.form.getlist('orders[]')
    
    piece_names = score_api.get_unique_piece_names()
    
    try:
        paired = [(int(o), f) for f, o in zip(filenames, orders)]
        paired.sort(key=lambda x: x[0])
        sorted_filenames = [f for _, f in paired]
        flash('ページの順番を更新しました。')
        return render_template('preview.html', previews=sorted_filenames, year=year, event_name=event_name, piece_guess=piece, inst_guess=instrument, piece_names=piece_names, device_name=device_name)
    except ValueError:
        flash('順序には数値を入力してください。')
        return render_template('preview.html', previews=filenames, year=year, event_name=event_name, piece_guess=piece, inst_guess=instrument, piece_names=piece_names, device_name=device_name)

@app.route('/save', methods=['POST'])
def save_score():
    year = request.form.get('year')
    event_name = request.form.get('event_name')
    piece = request.form.get('piece')
    instrument = request.form.get('instrument')
    preview_filenames = request.form.getlist('previews')
    
    if not piece or not instrument or not event_name or not year:
        flash('年度、演奏会名、楽曲名、楽器名をすべて入力してください。')
        piece_names = score_api.get_unique_piece_names()
        return render_template('preview.html', previews=preview_filenames, year=year, event_name=event_name, piece_guess=piece, inst_guess=instrument, piece_names=piece_names, error="必須項目です")
    try:
        pages = [Image.open(os.path.join(TEMP_PREVIEW_DIR, fname)) for fname in preview_filenames]
        saved_dir = score_api.save_and_register_score(pages, year, event_name, piece, instrument)
        flash(f'登録が完了しました！ [ 保存先: {saved_dir} ]')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'保存エラー: {str(e)}')
        return redirect(url_for('index'))

# --- 以下、前回の /list, /search, /piece, /output_action など続く ---
@app.route('/list')
def score_list():
    grouped_data = score_api.get_all_scores_grouped()
    return render_template('list.html', grouped_data=grouped_data)

@app.route('/search', methods=['GET'])
def search_score():
    keyword = request.args.get('keyword', '')
    if not keyword:
        flash('検索キーワードを入力してください。')
        return redirect(url_for('index'))
    results = score_api.search_pieces_by_keyword(keyword)
    return render_template('list.html', search_results=results, keyword=keyword)

@app.route('/piece')
def piece_details():
    year = request.args.get('year')
    event_name = request.args.get('event_name')
    piece = request.args.get('piece')
    if not year or not event_name or not piece: return redirect(url_for('score_list'))
    instruments = score_api.get_piece_details(year, event_name, piece)
    return render_template('piece.html', year=year, event_name=event_name, piece=piece, instruments=instruments)

@app.route('/output_action', methods=['POST'])
def output_action():
    directory = request.form.get('directory')
    mode = request.form.get('mode')
    action_type = request.form.get('action_type')
    printer = request.form.get('printer', '')
    year = request.form.get('year', '')
    event_name = request.form.get('event_name', '')
    piece = request.form.get('piece', 'score')
    inst = request.form.get('instrument', 'inst')

    if not directory or not os.path.exists(directory):
        flash('エラー: 指定されたデータがサーバー上に見つかりません。')
        return redirect(url_for('piece_details', year=year, event_name=event_name, piece=piece))
    try:
        if action_type == 'print':
            score_api.layout_and_print_score(directory=directory, mode=mode, orientation=score_api.DEFAULT_CONFIG['page_orientation'], printer_name=printer if printer else None, dpi=score_api.DEFAULT_CONFIG['dpi'])
            flash(f'[{piece} - {inst}] の印刷ジョブを送信しました！')
            return redirect(url_for('piece_details', year=year, event_name=event_name, piece=piece))
            
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
        return redirect(url_for('piece_details', year=year, event_name=event_name, piece=piece))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)