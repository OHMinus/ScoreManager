import cv2
import numpy as np
from PIL import Image
import os
import glob
import json
import uuid
import subprocess

DEFAULT_CONFIG = {
    'dpi': 300, 'margin_top': 5, 'margin_bottom': 5, 'margin_left': 5, 'margin_right': 5,
    'binary_threshold': 30, 'noise_min_area': 50, 'noise_min_density': 0.05,
    'enable_deskew': True, 'page_orientation': 'portrait', 'spread_width_ratio': 0.40,
    'auto_white_point': True, 'white_point_offset': 5, 'manual_white_point': 210,
    'black_point': 50, 'crop_padding_mm': 2.0, 'enable_final_closing': True,
    'final_closing_kernel': 2, 'booklet_direction': 'left'
}

def generate_debug_summary(steps, out_path):
    target_width = 1000
    canvas_parts = []
    for title, img in steps:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        scale = target_width / float(w)
        new_h = int(h * scale)
        resized = cv2.resize(img, (target_width, new_h))
        title_bar = np.zeros((50, target_width, 3), dtype=np.uint8)
        cv2.putText(title_bar, title, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        canvas_parts.append(title_bar)
        canvas_parts.append(resized)
    final_img = np.vstack(canvas_parts)
    cv2.imwrite(out_path, final_img)

def ensure_horizontal(cv_img):
    """
    【Split前処理】
    ピクセル分布の分散から五線譜が「縦か横か」を判定し、
    縦向き(90度寝ている)の場合は水平になるように回転させる。
    """
    print("\n--- [DEBUG] 水平判定を開始 ---")
    scale = 800.0 / max(cv_img.shape[0], cv_img.shape[1])
    small = cv2.resize(cv_img, (int(cv_img.shape[1] * scale), int(cv_img.shape[0] * scale)))
    gray = small if len(small.shape) == 2 else cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    proj_h = np.sum(thresh, axis=1)
    proj_v = np.sum(thresh, axis=0)
    var_h = np.var(proj_h)
    var_v = np.var(proj_v)
    
    if var_h > var_v:
        print(f"[DEBUG] 判定: 五線譜は水平方向です (Var_H:{var_h:.0f} > Var_V:{var_v:.0f})")
        return cv_img
    else:
        print(f"[DEBUG] 判定: 五線譜は垂直方向です (Var_H:{var_h:.0f} < Var_V:{var_v:.0f}) -> 90度回転します")
        return cv2.rotate(cv_img, cv2.ROTATE_90_CLOCKWISE)

def ensure_upright(cv_img, return_debug=False):
    """
    【Split後処理】
    単一ページに対して「左重みの法則」を適用し、
    右側が重い場合は逆さま(180度)と判定して補正する。
    """
    print("--- [DEBUG] 上下判定(180度)を開始 ---")
    scale = 800.0 / max(cv_img.shape[0], cv_img.shape[1])
    small = cv2.resize(cv_img, (int(cv_img.shape[1] * scale), int(cv_img.shape[0] * scale)))
    gray = small if len(small.shape) == 2 else cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # 余白を切り落とし、楽譜の「コンテンツ部分」だけを抽出
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        content = thresh[y:y+h, x:x+w]
    else:
        content = thresh
        w = content.shape[1]
        
    # コンテンツ内の「縦線（音部記号・括弧・小節線）」だけを抽出
    v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    verticals = cv2.morphologyEx(content, cv2.MORPH_OPEN, v_k)
    
    # 左端 15% と 右端 15% のインク（縦線）密度を比較
    margin = max(10, int(w * 0.15))
    left_density = np.sum(verticals[:, :margin])
    right_density = np.sum(verticals[:, -margin:])
    
    print(f"[DEBUG] 左端インク量: {left_density:.0f} | 右端インク量: {right_density:.0f}")
    
    is_upside_down = False
    if right_density > left_density * 1.1:
        is_upside_down = True
        print("[DEBUG] -> 右側の方が重いため、逆さま(180度反転)と判断しました。\n")
        res_img = cv2.rotate(cv_img, cv2.ROTATE_180)
    else:
        print("[DEBUG] -> 左側の方が重いため、正位置と判断しました。\n")
        res_img = cv_img
        
    # デバッグ描画
    if return_debug:
        debug_bg = cv2.bitwise_not(verticals)
        debug_color = cv2.cvtColor(debug_bg, cv2.COLOR_GRAY2BGR)
        overlay = debug_color.copy()
        cv2.rectangle(overlay, (0, 0), (margin, content.shape[0]), (255, 0, 0), -1)  # 青(左)
        cv2.rectangle(overlay, (content.shape[1] - margin, 0), (content.shape[1], content.shape[0]), (0, 0, 255), -1) # 赤(右)
        cv2.addWeighted(overlay, 0.2, debug_color, 0.8, 0, debug_color)
        
        cv2.putText(debug_color, f"LEFT: {left_density:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2)
        (tw, _), _ = cv2.getTextSize(f"RIGHT: {right_density:.0f}", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(debug_color, f"RIGHT: {right_density:.0f}", (content.shape[1] - margin - tw - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
        
        res_text = "ROTATED 180" if is_upside_down else "UPRIGHT (OK)"
        color = (0, 0, 255) if is_upside_down else (0, 255, 0)
        (tw, th), _ = cv2.getTextSize(res_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        cv2.rectangle(debug_color, (content.shape[1]//2 - tw//2 - 10, 10), (content.shape[1]//2 + tw//2 + 10, 20 + th + 10), (0, 0, 0), -1)
        cv2.putText(debug_color, res_text, (content.shape[1]//2 - tw//2, 20 + th), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        
        return res_img, debug_color
        
    return res_img

def mm_to_px(mm, dpi): return int(mm * dpi / 25.4)
def adjust_dynamic_range(cv_img, config):
    gray = cv_img if len(cv_img.shape) == 2 else cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    white_point = config['manual_white_point']
    if config['auto_white_point']:
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        peak_val = int(np.argmax(hist))
        white_point = max(0, peak_val - config['white_point_offset'])
    gray = np.clip(gray, config['black_point'], white_point)
    if white_point <= config['black_point']: return gray
    return ((gray - config['black_point']) * (255.0 / (white_point - config['black_point']))).astype(np.uint8)

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
        angles_fine = [a for a in [np.degrees(np.arctan2(l[0][3] - l[0][1], l[0][2] - l[0][0])) for l in lines] if -15 < a < 15]
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
        if current[0] <= merged_intervals[-1][1] + merge_distance: merged_intervals[-1][1] = max(merged_intervals[-1][1], current[1])
        else: merged_intervals.append(current)
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
        if area >= config['noise_min_area'] and density >= config['noise_min_density']: clean_mask[labels == i] = 255
    points = cv2.findNonZero(clean_mask)
    if points is None: cropped = np.ones((mm_to_px(10, config['dpi']), mm_to_px(10, config['dpi'])), dtype=np.uint8) * 255
    else:
        x, y, w, h = cv2.boundingRect(points)
        pad_px = mm_to_px(config['crop_padding_mm'], config['dpi'])
        img_h, img_w = cv_img.shape[:2]
        x1, y1 = max(0, x - pad_px), max(0, y - pad_px)
        x2, y2 = min(img_w, x + w + pad_px), min(img_h, y + h + pad_px)
        cropped = cv_img[y1:y2, x1:x2]
    pil_img = Image.fromarray(cropped)
    dpi = config['dpi']
    if config['page_orientation'] == 'portrait': a4_w, a4_h = mm_to_px(210, dpi), mm_to_px(297, dpi)
    else: a4_w, a4_h = mm_to_px(297, dpi), mm_to_px(210, dpi)
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

def scan_score_from_epson(output_filepath, dpi=300, device_name=None):
    cmd = ["scanimage"]
    if device_name: cmd.extend(["-d", device_name])
    cmd.extend(["--resolution", str(dpi), "--format=png", "--mode", "Gray"])
    try:
        with open(output_filepath, "wb") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE)
        if result.returncode != 0:
            error_msg = result.stderr.decode('utf-8', errors='replace')
            raise RuntimeError(f"スキャンに失敗しました:\n{error_msg}")
        return output_filepath
    except FileNotFoundError:
        raise FileNotFoundError("scanimage コマンドが見つかりません。")

def process_file_to_1in1(file_path, config=DEFAULT_CONFIG, debug_out_dir=None):
    file_bytes = np.fromfile(file_path, dtype=np.uint8)
    cv_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if cv_img is None: raise ValueError(f"画像を読み込めません: {file_path}")
    
    debug_steps = []
    def add_debug(title, img):
        if debug_out_dir:
            if isinstance(img, Image.Image):
                img_cv = np.array(img)
                if len(img_cv.shape) == 3: img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
            else:
                img_cv = img.copy()
            debug_steps.append((title, img_cv))

    add_debug("1_Original", cv_img)

    cv_img = adjust_dynamic_range(cv_img, config)
    add_debug("2_Adjusted_Dynamic_Range", cv_img)

    # ★ ステップ1: Split前に画像が「水平」になるようにだけ補正する
    cv_img = ensure_horizontal(cv_img)
    add_debug("3_Ensure_Horizontal", cv_img)
        
    cv_img = deskew_and_orient_score(cv_img, config)
    add_debug("4_Deskew", cv_img)

    # 画像を分割する
    split_images = detect_and_split_candidates(cv_img, config)
    processed_pages = []
    
    for idx, sub_img in enumerate(split_images):
        add_debug(f"5_Split_Candidate_{idx+1}", sub_img)
        
        # ★ ステップ2: Splitされた「1ページ」に対して、上下の向きを判定・補正する
        if debug_out_dir:
            sub_img, debug_upright = ensure_upright(sub_img, return_debug=True)
            add_debug(f"6_Ensure_Upright_{idx+1}", debug_upright)
        else:
            sub_img = ensure_upright(sub_img)
            
        cropped = crop_margins_and_fit(sub_img, config)
        add_debug(f"7_Cropped_Final_Page_{idx+1}", cropped)
        processed_pages.append(cropped)
        
    if debug_out_dir:
        os.makedirs(debug_out_dir, exist_ok=True)
        out_name = os.path.join(debug_out_dir, f"debug_{uuid.uuid4().hex[:8]}.jpg")
        generate_debug_summary(debug_steps, out_name)

    return processed_pages

# ==========================================
# JSON データベース操作 API
# ==========================================
DB_PATH = "scores_db.json"

def load_db(db_path=DB_PATH):
    if not os.path.exists(db_path): return {}
    with open(db_path, 'r', encoding='utf-8') as f:
        try: return json.load(f)
        except: return {}

def save_db(data, db_path=DB_PATH):
    with open(db_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def save_and_register_score(processed_pages_list, year, event_name, piece_name, composer, arranger, instrument, score_id=None, base_save_dir="score_data"):
    db = load_db()
    
    if score_id and score_id in db:
        score = db[score_id]
        event_exists = any(e.get('year') == str(year) and e.get('event_name') == str(event_name) for e in score.get('events', []))
        if not event_exists:
            score.setdefault('events', []).append({'year': str(year), 'event_name': str(event_name)})
            
        if instrument not in score.get('instruments', {}):
            score.setdefault('instruments', {})[instrument] = os.path.join(base_save_dir, score_id, instrument)
            
        save_dir = score['instruments'][instrument]
    else:
        score_id = str(uuid.uuid4())
        save_dir = os.path.join(base_save_dir, score_id, instrument)
        db[score_id] = {
            'piece': str(piece_name),
            'composer': str(composer),
            'arranger': str(arranger),
            'events': [{'year': str(year), 'event_name': str(event_name)}],
            'instruments': {
                instrument: save_dir
            }
        }
        
    os.makedirs(save_dir, exist_ok=True)
    existing_files = glob.glob(os.path.join(save_dir, "page_*.png"))
    start_idx = len(existing_files)
    
    for i, page in enumerate(processed_pages_list):
        save_path = os.path.join(save_dir, f"page_{start_idx + i + 1:03d}.png")
        page.save(save_path, optimize=True)

    save_db(db)
    return save_dir

def get_profiles_by_piece(piece):
    db = load_db()
    profiles = []
    for sid, s in db.items():
        if s.get('piece', '') == piece:
            profiles.append({
                'id': sid,
                'composer': s.get('composer', ''),
                'arranger': s.get('arranger', ''),
                'events': sorted(s.get('events', []), key=lambda x: (x.get('year',''), x.get('event_name','')), reverse=True)
            })
    return profiles

def get_unique_event_names():
    db = load_db()
    events = set()
    for s in db.values():
        for e in s.get('events', []):
            if e.get('event_name'): events.add(e['event_name'])
    return sorted(list(events))

def get_unique_piece_names():
    db = load_db()
    pieces = set(s.get('piece') for s in db.values() if s.get('piece'))
    return sorted(list(pieces))

def get_unique_composers_arrangers():
    db = load_db()
    composers = set(s.get('composer') for s in db.values() if s.get('composer'))
    arrangers = set(s.get('arranger') for s in db.values() if s.get('arranger'))
    return sorted(list(composers)), sorted(list(arrangers))

def get_piece_details(score_id):
    db = load_db()
    if score_id not in db: return None
    s = db[score_id]
    
    insts = []
    for inst_name, d in s.get('instruments', {}).items():
        insts.append({'name': inst_name, 'dir': d})
    insts = sorted(insts, key=lambda x: x['name'])
    
    return {
        'id': score_id,
        'piece': s.get('piece', ''),
        'composer': s.get('composer', ''),
        'arranger': s.get('arranger', ''),
        'events': sorted(s.get('events', []), key=lambda x: (x.get('year',''), x.get('event_name','')), reverse=True),
        'instruments': insts
    }

def update_composer_arranger(score_id, composer, arranger):
    db = load_db()
    if score_id in db:
        db[score_id]['composer'] = composer
        db[score_id]['arranger'] = arranger
        save_db(db)
        return True
    return False

def add_event_to_score(score_id, dest_year, dest_event):
    db = load_db()
    if score_id not in db: return False
    
    events = db[score_id].setdefault('events', [])
    if any(e.get('year') == str(dest_year) and e.get('event_name') == str(dest_event) for e in events):
        return True 
        
    events.append({'year': str(dest_year), 'event_name': str(dest_event)})
    save_db(db)
    return True

def get_all_scores_grouped():
    db = load_db()
    data = {}
    for sid, s in db.items():
        for e in s.get('events', []):
            y = e.get('year', '')
            ev = e.get('event_name', '')
            key = (y, ev)
            if key not in data: data[key] = []
            data[key].append({
                'id': sid, 'piece': s.get('piece', ''),
                'composer': s.get('composer', ''), 'arranger': s.get('arranger', '')
            })
    
    sorted_data = {}
    for key in sorted(data.keys(), key=lambda k: (k[0], k[1]), reverse=True):
        sorted_data[key] = sorted(data[key], key=lambda x: x['piece'])
    return sorted_data

def get_all_scores_by_piece():
    db = load_db()
    pieces_list = []
    for sid, s in db.items():
        pieces_list.append({
            'id': sid,
            'piece': s.get('piece', ''),
            'composer': s.get('composer', ''),
            'arranger': s.get('arranger', ''),
            'events': sorted(s.get('events', []), key=lambda x: (x.get('year',''), x.get('event_name','')), reverse=True)
        })
    return sorted(pieces_list, key=lambda x: x['piece'])

def search_pieces_by_keyword(keyword):
    db = load_db()
    results = []
    keyword = keyword.lower()
    for sid, s in db.items():
        pi = s.get('piece', '')
        comp = s.get('composer', '')
        arr = s.get('arranger', '')
        
        match = False
        if keyword in pi.lower() or keyword in comp.lower() or keyword in arr.lower():
            match = True
            
        for e in s.get('events', []):
            combined = f"{e.get('year', '')}{e.get('event_name', '')}"
            if keyword in combined.lower():
                match = True
                break
                
        if match:
            results.append({
                'id': sid, 'piece': pi, 'composer': comp, 'arranger': arr,
                'events': sorted(s.get('events', []), key=lambda x: (x.get('year',''), x.get('event_name','')), reverse=True)
            })
    return sorted(results, key=lambda x: x['piece'])

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

    if mode == '1in1': output_pages = pages
    elif mode == '2in1':
        for i in range(0, len(pages), 2):
            output_pages.append(create_a3(pages[i], pages[i+1] if (i + 1) < len(pages) else None))
    elif mode == 'booklet':
        bw, bh = (mm_to_px(210, dpi), mm_to_px(297, dpi)) if orientation == 'portrait' else (mm_to_px(297, dpi), mm_to_px(210, dpi))
        blank = Image.new('L', (bw, bh), 255)
        while len(pages) % 4 != 0: pages.append(blank)
        total = len(pages)
        for i in range(total // 2):
            if booklet_dir == 'left': idx1, idx2 = (total - 1 - i, i) if i % 2 == 0 else (i, total - 1 - i)
            else: idx1, idx2 = (i, total - 1 - i) if i % 2 == 0 else (total - 1 - i, i)
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
        if result.returncode != 0: raise RuntimeError(f"印刷コマンドに失敗: {result.stderr}")
        return True
    return False

def GetDefaultConfig():
    return DEFAULT_CONFIG.copy()