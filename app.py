from flask import Flask, render_template, request, redirect, url_for, flash
import os
import glob
import uuid
from PIL import Image
import score_api
import argparse

app = Flask(__name__)
# flashメッセージを使用するためのシークレットキー（推測されにくい文字列に変更可能です）
app.secret_key = 'score_processor_secret_key'

# 一時ファイルの保存先設定
TEMP_UPLOAD_DIR = os.path.join('temp', 'uploads')
TEMP_PREVIEW_DIR = os.path.join('temp', 'previews')
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMP_PREVIEW_DIR, exist_ok=True)

def clear_temp_dir(directory):
    """一時フォルダ内の古いファイルを削除する"""
    for f in glob.glob(os.path.join(directory, '*')):
        try: os.remove(f)
        except: pass

@app.route('/')
def index():
    """メイン画面（アップロード＆印刷フォーム）を表示"""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_files():
    """画像を受け取り、処理を行ってプレビュー画面へ遷移する"""
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
    
    try:
        for file in files:
            if file.filename == '': continue
            
            temp_path = os.path.join(TEMP_UPLOAD_DIR, file.filename)
            file.save(temp_path)
            
            # APIを呼び出して処理
            pages = score_api.process_file_to_1in1(temp_path, score_api.DEFAULT_CONFIG)
            
            # プレビュー用に保存
            for page in pages:
                unique_filename = f"{uuid.uuid4().hex}.png"
                preview_path = os.path.join(TEMP_PREVIEW_DIR, unique_filename)
                page.save(preview_path, optimize=True)
                preview_filenames.append(unique_filename)
                
        # 処理が成功したら、プレビュー画面を描画して返す
        return render_template('preview.html', previews=preview_filenames)
        
    except Exception as e:
        flash(f'処理エラー: {str(e)}')
        return redirect(url_for('index'))

@app.route('/save', methods=['POST'])
def save_score():
    """プレビュー画面からの保存要求を受け取り、DB登録してメイン画面へ戻る"""
    piece = request.form.get('piece')
    instrument = request.form.get('instrument')
    # HTMLの hidden input からプレビューファイル名のリストを受け取る
    preview_filenames = request.form.getlist('previews')
    
    if not piece or not instrument:
        flash('楽曲名と楽器名を入力してください。')
        # 入力不足の場合は再度プレビュー画面を返す
        return render_template('preview.html', previews=preview_filenames, error="楽曲名と楽器名は必須です")
        
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
    """メイン画面からの印刷要求を受け取り、印刷実行してメイン画面へ戻る"""
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
   
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="楽譜画像処理・管理・印刷 CUIツール")
    subparsers = parser.add_subparsers(dest="command", help="実行するコマンドを選択してください")

    parser_gui = subparsers.add_parser("gui", help="WebベースのGUIを起動します (Flaskサーバー)")

    parser_scan = subparsers.add_parser("scan", help="スキャナーから画像を取り込みます")
    parser_scan.add_argument("-o", "--output", required=True, help="保存先のファイル名 (例: temp.png)")
    parser_scan.add_argument("-d", "--device", help="スキャナーのデバイス名 (省略時はデフォルト)")

    parser_reg = subparsers.add_parser("register", help="入力画像を処理し、1in1形式でDBに登録・保存します")
    parser_reg.add_argument("-i", "--input", required=True, help="入力画像ディレクトリ、または単一ファイル")
    parser_reg.add_argument("-p", "--piece", required=True, help="楽曲名")
    parser_reg.add_argument("-inst", "--instrument", required=True, help="楽器名")

    parser_print = subparsers.add_parser("print", help="DBから楽譜を検索して印刷します")
    parser_print.add_argument("-p", "--piece", required=True, help="楽曲名")
    parser_print.add_argument("-inst", "--instrument", required=True, help="楽器名")
    parser_print.add_argument("-m", "--mode", choices=['1in1', '2in1', 'booklet'], default='booklet', help="印刷モード")
    parser_print.add_argument("-d", "--printer", help="対象のプリンタ名")

    args = parser.parse_args()

    if args.command == "gui":
        app.run(host='0.0.0.0', port=5000, debug=True)

    if args.command == "scan":
        score_api.scan_score_from_epson(args.output, dpi=score_api.GetDefaultConfig()['dpi'], device_name=args.device)
        print(f"スキャンが完了しました: {args.output}")

    elif args.command == "register":
        input_path = args.input
        files_to_process = []
        
        if os.path.isdir(input_path):
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.bmp']:
                files_to_process.extend(glob.glob(os.path.join(input_path, ext)))
            files_to_process.sort()
        elif os.path.isfile(input_path):
            files_to_process.append(input_path)

        if not files_to_process:
            print("処理対象の画像が見つかりませんでした。")
            exit(1)

        all_processed_pages = []
        for file in files_to_process:
            print(f"処理中: {os.path.basename(file)}")
            try:
                pages = score_api.process_file_to_1in1(file, score_api.GetDefaultConfig())
                all_processed_pages.extend(pages)
            except Exception as e:
                print(f"エラー発生 ({os.path.basename(file)}): {e}")

        if all_processed_pages:
            saved_dir = score_api.save_and_register_score(all_processed_pages, args.piece, args.instrument)
            print(f"完了: {len(all_processed_pages)} ページを処理し保存しました -> {saved_dir}")

    elif args.command == "print":
        target_dir = score_api.search_score_directory(args.piece, args.instrument)
        if not target_dir:
            print("エラー: 該当する楽譜データがDBに見つかりませんでした。")
            exit(1)
            
        try:
            score_api.layout_and_print_score(
                directory=target_dir,
                mode=args.mode,
                orientation=score_api.GetDefaultConfig()['page_orientation'],
                printer_name=args.printer,
                dpi=score_api.GetDefaultConfig()['dpi']
            )
            print("印刷ジョブの送信が完了しました。")
        except Exception as e:
            print(f"印刷エラー: {e}")
            exit(1)
    else:
        parser.print_help()