import cv2
import numpy as np
from PIL import Image
import os
import glob
import csv
import subprocess

# ==========================================
# デフォルト設定 (GUI/CUIから上書き可能)
# ==========================================
DEFAULT_CONFIG = {
    'dpi': 300,
    'margin_top': 5,
    'margin_bottom': 5,
    'margin_left': 5,
    'margin_right': 5,
    'binary_threshold': 30,
    'noise_min_area': 50,
    'noise_min_density': 0.05,
    'enable_deskew': True,
    'page_orientation': 'portrait',
    'spread_width_ratio': 0.40,
    'auto_white_point': True,
    'white_point_offset': 5,
    'manual_white_point': 210,
    'black_point': 50,
    'crop_padding_mm': 2.0,
    'enable_final_closing': True,
    'final_closing_kernel': 2,
    'booklet_direction': 'left' # 小冊子の開き方向を追加
}


# ==========================================
# 内部処理用ヘルパー関数
# ==========================================

def mm_to_px(mm, dpi):
    return int(mm * dpi / 25.4)

def adjust_dynamic_range(cv_img, config):
    gray = cv_img if len(cv_img.shape) == 2 else cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    white_point = config['manual_white_point']
    if config['auto_white_point']:
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        peak_val = int(np.argmax(hist))
        white_point = max(0, peak_val - config['white_point_offset'])
        
    gray = np.clip(gray, config['black_point'], white_point)
    if white_point <= config['black_point']: return gray
    gray = ((gray - config['black_point']) * (255.0 / (white_point - config['black_point']))).astype(np.uint8)
    return gray

def deskew_and_orient_score(cv_img, config):
    if not config['enable_deskew']: return cv_img
    
    gray = cv_img if len(cv_img.shape) == 2 else cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    if lines is not None:
        angles = [np.degrees(np.arctan2(l[0][3] - l[0][1], l[0][2] - l[0][0])) for l in lines]
        vertical_lines = [a for a in angles if abs(a) > 70]
        horizontal_lines = [a for a in angles if abs(a) < 20]
        
        if len(vertical_lines) > len(horizontal_lines) * 2:
            cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_CLOCKWISE)
            gray = cv_img if len(cv_img.shape) == 2 else cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

    if lines is not None:
        angles_fine = [np.degrees(np.arctan2(l[0][3] - l[0][1], l[0][2] - l[0][0])) for l in lines]
        angles_fine = [a for a in angles_fine if -15 < a < 15]
        if angles_fine:
            median_angle = np.median(angles_fine)
            if abs(median_angle) >= 0.1:
                (h, w) = cv_img.shape[:2]
                M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
                bg_color = (255, 255, 255) if len(cv_img.shape) == 3 else 255
                cv_img = cv2.warpAffine(cv_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)
    return cv_img

def detect_and_split_candidates(cv_img, config):
    gray = cv_img if len(cv_img.shape) == 2 else cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, config['binary_threshold'], 255, cv2.THRESH_BINARY_INV)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    
    intervals = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > config['noise_min_area'] and stats[i, cv2.CC_STAT_WIDTH] > 10 and stats[i, cv2.CC_STAT_HEIGHT] > 10:
            left = stats[i, cv2.CC_STAT_LEFT]
            intervals.append([left, left + stats[i, cv2.CC_STAT_WIDTH]])
            
    if not intervals: return [cv_img]

    intervals.sort(key=lambda x: x[0])
    merged_intervals = [intervals[0]]
    merge_distance = mm_to_px(5, config['dpi']) 
    
    for current in intervals[1:]:
        if current[0] <= merged_intervals[-1][1] + merge_distance:
            merged_intervals[-1][1] = max(merged_intervals[-1][1], current[1])
        else:
            merged_intervals.append(current)

    candidates = [{'x': l, 'w': r - l, 'right': r} for l, r in merged_intervals]
    candidates.sort(key=lambda c: c['w'], reverse=True)

    if len(candidates) >= 2 and (candidates[1]['w'] / candidates[0]['w']) >= config['spread_width_ratio']:
        c1, c2 = (candidates[0], candidates[1]) if candidates[0]['x'] < candidates[1]['x'] else (candidates[1], candidates[0])
        cleaned_img = cv_img.copy()
        mask = np.zeros_like(gray)
        mask[:, c1['x']:c1['right']] = 255
        mask[:, c2['x']:c2['right']] = 255
        cleaned_img[mask == 0] = 255
        split_x = c1['right'] + (c2['x'] - c1['right']) // 2
        return [cleaned_img[:, :split_x], cleaned_img[:, split_x:]]
    elif len(candidates) >= 1:
        cand1 = candidates[0]
        cleaned_img = cv_img.copy()
        mask = np.zeros_like(gray)
        mask[:, cand1['x']:cand1['right']] = 255
        cleaned_img[mask == 0] = 255
        return [cleaned_img]
    return [cv_img]

def crop_margins_and_fit(cv_img, config):
    gray = cv_img if len(cv_img.shape) == 2 else cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, config['binary_threshold'], 255, cv2.THRESH_BINARY_INV)
    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    clean_mask = np.zeros_like(thresh)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w_st = stats[i, cv2.CC_STAT_WIDTH]
        h_st = stats[i, cv2.CC_STAT_HEIGHT]
        density = area / (w_st * h_st) if (w_st * h_st) > 0 else 0
        if area >= config['noise_min_area'] and density >= config['noise_min_density']:
            clean_mask[labels == i] = 255

    points = cv2.findNonZero(clean_mask)
    if points is None:
        cropped = np.ones((mm_to_px(10, config['dpi']), mm_to_px(10, config['dpi'])), dtype=np.uint8) * 255
    else:
        x, y, w, h = cv2.boundingRect(points)
        pad_px = mm_to_px(config['crop_padding_mm'], config['dpi'])
        img_h, img_w = cv_img.shape[:2]
        x1, y1 = max(0, x - pad_px), max(0, y - pad_px)
        x2, y2 = min(img_w, x + w + pad_px), min(img_h, y + h + pad_px)
        cropped = cv_img[y1:y2, x1:x2]

    pil_img = Image.fromarray(cropped)
    dpi = config['dpi']
    if config['page_orientation'] == 'portrait':
        a4_w, a4_h = mm_to_px(210, dpi), mm_to_px(297, dpi)
    else:
        a4_w, a4_h = mm_to_px(297, dpi), mm_to_px(210, dpi)

    valid_w = a4_w - mm_to_px(config['margin_left'] + config['margin_right'], dpi)
    valid_h = a4_h - mm_to_px(config['margin_top'] + config['margin_bottom'], dpi)

    img_w, img_h = pil_img.size
    scale = min(valid_w / img_w, valid_h / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    resized_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    a4_canvas = Image.new('L', (a4_w, a4_h), 255) 
    paste_x = mm_to_px(config['margin_left'], dpi) + (valid_w - new_w) // 2
    paste_y = mm_to_px(config['margin_top'], dpi) + (valid_h - new_h) // 2
    a4_canvas.paste(resized_img, (paste_x, paste_y))

    img_bin = (np.array(a4_canvas, 'f') > 128).astype(np.uint8) * 255
    if config['enable_final_closing']:
        img_inv = 255 - img_bin
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (config['final_closing_kernel'], config['final_closing_kernel']))
        img_bin = 255 - cv2.morphologyEx(img_inv, cv2.MORPH_CLOSE, kernel)
        
    return Image.fromarray(img_bin)

# ==========================================
# GUI/CUI向け 公開API
# ==========================================

def scan_score_from_epson(output_filepath, dpi=300, device_name=None):
    cmd = ["scanimage"]
    if device_name:
        cmd.extend(["-d", device_name])
        
    cmd.extend([
        "--resolution", str(dpi),
        "--format=png",       
        "--mode", "Gray"      
    ])
    
    try:
        with open(output_filepath, "wb") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE)
            
        if result.returncode != 0:
            error_msg = result.stderr.decode('utf-8', errors='replace')
            raise RuntimeError(f"スキャンに失敗しました:\n{error_msg}")
            
        return output_filepath
    except FileNotFoundError:
        raise FileNotFoundError("scanimage コマンドが見つかりません。SANEパッケージがインストールされているか確認してください。")

def process_file_to_1in1(file_path, config=DEFAULT_CONFIG):
    file_bytes = np.fromfile(file_path, dtype=np.uint8)
    cv_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if cv_img is None:
        raise ValueError(f"画像を読み込めません: {file_path}")

    cv_img = adjust_dynamic_range(cv_img, config)
    cv_img = deskew_and_orient_score(cv_img, config)
    split_images = detect_and_split_candidates(cv_img, config)
    
    processed_pages = []
    for sub_img in split_images:
        a4_pil = crop_margins_and_fit(sub_img, config)
        processed_pages.append(a4_pil)
        
    return processed_pages

# --- score_api.py の save_and_register_score 以降を以下に差し替え ---

def save_and_register_score(processed_pages_list, year, event_name, piece_name, instrument, db_csv_path="scores_db.csv", base_save_dir="score_data"):
    # 年度と演奏会名を結合したフォルダ名を作成 (例: 2026定期演奏会)
    combined_event = f"{year}{event_name}"
    save_dir = os.path.join(base_save_dir, combined_event, piece_name, instrument)
    os.makedirs(save_dir, exist_ok=True)
    
    for i, page in enumerate(processed_pages_list):
        save_path = os.path.join(save_dir, f"page_{i+1:03d}.png")
        page.save(save_path, optimize=True)
        
    db_exists = os.path.exists(db_csv_path)
    records = []
    updated = False
    
    if db_exists:
        with open(db_csv_path, mode='r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                # 互換性維持: 古いデータ('Event')がある場合はそれを EventName に回す
                r_year = row.get('Year', '')
                r_ev = row.get('EventName', row.get('Event', ''))
                
                if r_year == year and r_ev == event_name and row.get('Piece') == piece_name and row.get('Instrument') == instrument:
                    row['Directory'] = save_dir
                    updated = True
                
                records.append({
                    'Year': r_year,
                    'EventName': r_ev,
                    'Piece': row.get('Piece', ''),
                    'Instrument': row.get('Instrument', ''),
                    'Directory': row.get('Directory', '')
                })
                
    if not updated:
        records.append({'Year': year, 'EventName': event_name, 'Piece': piece_name, 'Instrument': instrument, 'Directory': save_dir})
        
    with open(db_csv_path, mode='w', encoding='utf-8', newline='') as f:
        # DBのカラム構造を刷新
        writer = csv.DictWriter(f, fieldnames=['Year', 'EventName', 'Piece', 'Instrument', 'Directory'])
        writer.writeheader()
        writer.writerows(records)
        
    return save_dir

def get_unique_event_names(db_csv_path="scores_db.csv"):
    """
    サジェスト用に、DBに存在するユニークな演奏会名のリストを取得する。
    """
    if not os.path.exists(db_csv_path): return []
    events = set()
    with open(db_csv_path, mode='r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            ev = row.get('EventName', row.get('Event', ''))
            if ev: events.add(ev)
    return sorted(list(events))

def get_unique_piece_names(db_csv_path="scores_db.csv"):
    """
    サジェスト用に、DBに存在するユニークな楽曲名のリストを取得する。
    """
    if not os.path.exists(db_csv_path): return []
    pieces = set()
    with open(db_csv_path, mode='r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            pi = row.get('Piece', '')
            if pi: pieces.add(pi)
    return sorted(list(pieces))

def get_all_scores_grouped(db_csv_path="scores_db.csv"):
    """
    (Year, EventName) のタプルでグループ化し、辞書で返す
    """
    if not os.path.exists(db_csv_path): return {}
    data = {}
    with open(db_csv_path, mode='r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            y = row.get('Year', '')
            ev = row.get('EventName', row.get('Event', ''))
            pi = row.get('Piece', '')
            if not ev and not pi: continue
            
            key = (y, ev)
            if key not in data:
                data[key] = set()
            data[key].add(pi)
            
    # 年度と行事名でソート（降順・昇順の制御）
    sorted_data = {}
    for key in sorted(data.keys(), key=lambda k: (k[0], k[1])):
        sorted_data[key] = sorted(list(data[key]))
    return sorted_data

def search_pieces_by_keyword(keyword, db_csv_path="scores_db.csv"):
    if not os.path.exists(db_csv_path): return []
    results = set()
    keyword = keyword.lower()
    
    with open(db_csv_path, mode='r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            y = row.get('Year', '')
            ev = row.get('EventName', row.get('Event', ''))
            pi = row.get('Piece', '')
            combined = f"{y}{ev}"
            
            if keyword in combined.lower() or keyword in pi.lower():
                results.add((y, ev, pi))
                
    return sorted(list(results), key=lambda x: (x[0], x[1], x[2]))

def get_piece_details(year, event_name, piece, db_csv_path="scores_db.csv"):
    if not os.path.exists(db_csv_path): return []
    instruments = []
    
    with open(db_csv_path, mode='r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            r_year = row.get('Year', '')
            r_ev = row.get('EventName', row.get('Event', ''))
            if r_year == year and r_ev == event_name and row.get('Piece') == piece:
                instruments.append({
                    'name': row.get('Instrument', ''),
                    'dir': row.get('Directory', '')
                })
                
    return sorted(instruments, key=lambda x: x['name'])

def apply_layout(directory, mode='booklet', orientation='portrait', booklet_dir='left', dpi=300):
    if not os.path.exists(directory): raise FileNotFoundError(f"ディレクトリが見つかりません: {directory}")
    image_files = sorted(glob.glob(os.path.join(directory, "*.png")))
    if not image_files: raise ValueError(f"画像がありません: {directory}")
        
    pages = [Image.open(f) for f in image_files]
    output_pages = []

    def create_a3(p1, p2):
        if orientation == 'portrait':
            a3_w, a3_h = mm_to_px(420, dpi), mm_to_px(297, dpi)
            offset_x, offset_y = mm_to_px(210, dpi), 0
        else:
            a3_w, a3_h = mm_to_px(297, dpi), mm_to_px(420, dpi)
            offset_x, offset_y = 0, mm_to_px(210, dpi)
        canvas = Image.new('L', (a3_w, a3_h), 255)
        if p1: canvas.paste(p1, (0, 0))
        if p2: canvas.paste(p2, (offset_x, offset_y))
        return canvas

    if mode == '1in1':
        output_pages = pages
    elif mode == '2in1':
        for i in range(0, len(pages), 2):
            p1 = pages[i]
            p2 = pages[i+1] if (i + 1) < len(pages) else None
            output_pages.append(create_a3(p1, p2))
    elif mode == 'booklet':
        bw, bh = (mm_to_px(210, dpi), mm_to_px(297, dpi)) if orientation == 'portrait' else (mm_to_px(297, dpi), mm_to_px(210, dpi))
        blank = Image.new('L', (bw, bh), 255)
        while len(pages) % 4 != 0: pages.append(blank)
        total = len(pages)
        for i in range(total // 2):
            if booklet_dir == 'left':
                idx1, idx2 = (total - 1 - i, i) if i % 2 == 0 else (i, total - 1 - i)
            else:
                idx1, idx2 = (i, total - 1 - i) if i % 2 == 0 else (total - 1 - i, i)
            output_pages.append(create_a3(pages[idx1], pages[idx2]))

    return output_pages

def layout_and_print_score(directory, mode='booklet', orientation='portrait', printer_name=None, booklet_dir='left', dpi=300):
    output_pages = apply_layout(directory, mode, orientation, booklet_dir, dpi)
    temp_pdf_path = "/tmp/score_print_temp.pdf"
    if output_pages:
        output_pages[0].save(temp_pdf_path, save_all=True, append_images=output_pages[1:], format='PDF', resolution=dpi)
        print_cmd = ["lp"]
        if printer_name: print_cmd.extend(["-d", printer_name])
        if mode in ['2in1', 'booklet']: print_cmd.extend(["-o", "media=A3"])
        else: print_cmd.extend(["-o", "media=A4"])
        print_cmd.append(temp_pdf_path)
        
        result = subprocess.run(print_cmd, capture_output=True, text=True)
        if result.returncode != 0: raise RuntimeError(f"印刷コマンドに失敗しました:\n{result.stderr}")
        return True
    return False

def GetDefaultConfig():
    return DEFAULT_CONFIG.copy()