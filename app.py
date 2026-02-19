import cv2
import numpy as np
from PIL import Image
import os
import glob

# ==========================================
# ユーザー設定エリア
# ==========================================

INPUT_FOLDER = 'input_scores'
OUTPUT_FOLDER = 'output_cl_alvamer'
DEBUG_FOLDER = 'debug_output'

DPI = 300

MARGIN_TOP = 5
MARGIN_BOTTOM = 5
MARGIN_LEFT = 5
MARGIN_RIGHT = 5

BINARY_THRESHOLD = 30
WIDTH_THRESHOLD = 10
HEIGHT_THRESHOLD = 10

NOISE_MIN_AREA = 10   
NOISE_MIN_DENSITY = 0.02

# ==========================================
# 機能設定エリア
# ==========================================

# 1. 回転・向き補正
ENABLE_DESKEW = True

# 2. 楽譜原稿の縦横指定 ('portrait' または 'landscape')
PAGE_ORIENTATION = 'portrait'

# 3. 出力モード ('1in1', '2in1', 'booklet')
OUTPUT_MODE = '1in1'
BOOKLET_DIRECTION = 'left'

# 4. デバッグ画像出力
ENABLE_DEBUG = True

# 5. 見開き判定のしきい値
SPREAD_WIDTH_RATIO = 0.40

# 6. ダイナミックレンジ補正 (Auto White Point対応)
AUTO_WHITE_POINT = True     
WHITE_POINT_OFFSET = -20  
MANUAL_WHITE_POINT = 210    
BLACK_POINT = 50

# 7. クロップ（切り取り）時の余裕パディング
CROP_PADDING_MM = 2.0

# 8. 最終出力の線かすれ補修 (クロージング) (New!)
# かすれて途切れた五線譜や音符の線を綺麗に繋ぎ合わせます。
ENABLE_FINAL_CLOSING = True
FINAL_CLOSING_KERNEL = 2 # 線の繋ぎ具合 (目安: 1〜3)

# ==========================================

def mm_to_px(mm, dpi):
    return int(mm * dpi / 25.4)

def calculate_dynamic_white_point(cv_img):
    gray = cv_img if len(cv_img.shape) == 2 else cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    peak_val = int(np.argmax(hist))
    dynamic_wp = max(0, peak_val - WHITE_POINT_OFFSET)
    
    print(f"    -> ヒストグラム解析: ピーク={peak_val}, 動的WHITE_POINT={dynamic_wp} (オフセット={WHITE_POINT_OFFSET})")
    return dynamic_wp, hist

def draw_histogram_image(hist, white_point, black_point):
    h, w = 600, 1024 
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 240
    cv2.rectangle(canvas, (0, 0), (w-1, h-1), (100, 100, 100), 2)
    
    hist_norm = np.zeros_like(hist)
    cv2.normalize(hist, hist_norm, 0, h - 50, cv2.NORM_MINMAX)
    
    for x in range(256):
        val = int(hist_norm[x, 0])
        cv2.line(canvas, (x * 4, h), (x * 4, h - val), (150, 150, 150), 3)
        
    bp_x = black_point * 4
    cv2.line(canvas, (bp_x, 0), (bp_x, h), (255, 0, 0), 2)
    cv2.putText(canvas, f"BP:{black_point}", (max(0, bp_x - 60), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    wp_x = white_point * 4
    cv2.line(canvas, (wp_x, 0), (wp_x, h), (0, 0, 255), 2)
    cv2.putText(canvas, f"WP:{white_point}", (min(w - 100, wp_x + 10), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    return canvas

def adjust_dynamic_range(cv_img, white_point):
    gray = cv_img if len(cv_img.shape) == 2 else cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    gray = np.clip(gray, BLACK_POINT, white_point)
    if white_point <= BLACK_POINT:
        return gray
    gray = ((gray - BLACK_POINT) * (255.0 / (white_point - BLACK_POINT))).astype(np.uint8)
    return gray

def deskew_and_orient_score(cv_img):
    if len(cv_img.shape) == 3:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv_img.copy()
        
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    rotated_90 = False
    angle_fine = 0.0
    
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)
            
        vertical_lines = [a for a in angles if abs(a) > 70]
        horizontal_lines = [a for a in angles if abs(a) < 20]
        
        if len(vertical_lines) > len(horizontal_lines) * 2:
            print("    -> 五線譜が縦方向であると検知しました。90度回転(時計回り)させます。")
            cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_CLOCKWISE)
            rotated_90 = True
            
            gray = cv_img if len(cv_img.shape) == 2 else cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

    if lines is not None:
        angles_fine = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -15 < a < 15:
                angles_fine.append(a)
                
        if angles_fine:
            median_angle = np.median(angles_fine)
            if abs(median_angle) >= 0.1:
                print(f"    -> 回転補正を適用: {median_angle:.2f}度")
                angle_fine = median_angle
                (h, w) = cv_img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                bg_color = (255, 255, 255) if len(cv_img.shape) == 3 else 255
                cv_img = cv2.warpAffine(cv_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)
                
    return cv_img, angle_fine, rotated_90

def detect_and_split_candidates(cv_img):
    gray = cv_img if len(cv_img.shape) == 2 else cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    
    intervals = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        
        if area > NOISE_MIN_AREA and w > 10 and h > 10:
            left = stats[i, cv2.CC_STAT_LEFT]
            right = left + w
            intervals.append([left, right])
            
    if not intervals:
        return [cv_img]

    intervals.sort(key=lambda x: x[0])
    merged_intervals = [intervals[0]]
    merge_distance = mm_to_px(5, DPI) 
    
    for current in intervals[1:]:
        previous = merged_intervals[-1]
        if current[0] <= previous[1] + merge_distance:
            previous[1] = max(previous[1], current[1])
        else:
            merged_intervals.append(current)

    candidates = []
    for idx, (left, right) in enumerate(merged_intervals):
        w = right - left
        candidates.append({'x': left, 'w': w, 'right': right})
        
    candidates.sort(key=lambda c: c['w'], reverse=True)

    if len(candidates) >= 2:
        cand1, cand2 = candidates[0], candidates[1]
        ratio = cand2['w'] / cand1['w']
        
        if ratio >= SPREAD_WIDTH_RATIO:
            print(f"    -> 候補領域の横幅比率({ratio*100:.1f}%)が閾値以上。見開きと判定し分割します。")
            left_cand, right_cand = (cand1, cand2) if cand1['x'] < cand2['x'] else (cand2, cand1)
            
            cleaned_img = cv_img.copy()
            mask = np.zeros_like(gray)
            mask[:, left_cand['x']:left_cand['right']] = 255
            mask[:, right_cand['x']:right_cand['right']] = 255
            cleaned_img[mask == 0] = 255
            
            split_x = left_cand['right'] + (right_cand['x'] - left_cand['right']) // 2
            
            if split_x <= left_cand['x'] or split_x >= right_cand['right']:
                split_x = cv_img.shape[1] // 2
                
            return [cleaned_img[:, :split_x], cleaned_img[:, split_x:]]
            
        else:
            print(f"    -> 候補領域の横幅比率({ratio*100:.1f}%)が閾値未満。小さい領域をノイズとして消去します。")
            cleaned_img = cv_img.copy()
            mask = np.zeros_like(gray)
            mask[:, cand1['x']:cand1['right']] = 255
            cleaned_img[mask == 0] = 255
            return [cleaned_img]
            
    elif len(candidates) == 1:
        cand1 = candidates[0]
        cleaned_img = cv_img.copy()
        mask = np.zeros_like(gray)
        mask[:, cand1['x']:cand1['right']] = 255
        cleaned_img[mask == 0] = 255
        return [cleaned_img]

    return [cv_img]

def crop_margins(cv2_img):
    gray = cv2_img if len(cv2_img.shape) == 2 else cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    
    # 微細ノイズの除去 (オープニング)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    # 途切れた線の結合 (クロージング)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    
    clean_mask = np.zeros_like(thresh)
    component_info = []
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w_st = stats[i, cv2.CC_STAT_WIDTH]
        h_st = stats[i, cv2.CC_STAT_HEIGHT]
        x_st = stats[i, cv2.CC_STAT_LEFT]
        y_st = stats[i, cv2.CC_STAT_TOP]
        
        density = area / (w_st * h_st) if (w_st * h_st) > 0 else 0
        
        is_kept = False
        if area >= NOISE_MIN_AREA and density >= NOISE_MIN_DENSITY:
            clean_mask[labels == i] = 255
            is_kept = True
            
        if w_st > 5 and h_st > 5:
            component_info.append({
                'rect': (x_st, y_st, w_st, h_st),
                'area': area,
                'density': density,
                'is_kept': is_kept
            })

    points = cv2.findNonZero(clean_mask)

    if points is None:
        return cv2_img, None, None, component_info
    
    x, y, w, h = cv2.boundingRect(points)
    
    pts = points.reshape(-1, 2)
    reasonPoints = [
        tuple(pts[pts[:, 0].argmin()]), 
        tuple(pts[pts[:, 1].argmin()]), 
        tuple(pts[pts[:, 0].argmax()]), 
        tuple(pts[pts[:, 1].argmax()])  
    ]

    pad_px = mm_to_px(CROP_PADDING_MM, DPI)
    img_h, img_w = cv2_img.shape[:2]
    
    x1 = max(0, x - pad_px)
    y1 = max(0, y - pad_px)
    x2 = min(img_w, x + w + pad_px)
    y2 = min(img_h, y + h + pad_px)
    
    pad_w = x2 - x1
    pad_h = y2 - y1

    print(f"    -> クロップ領域: x={x1}, y={y1}, width={pad_w}, height={pad_h} (パディング適用)")
    cropped = cv2_img[y1:y2, x1:x2]
    
    return cropped, (x1, y1, pad_w, pad_h), reasonPoints, component_info

def fit_to_a4(pil_img, dpi, orientation):
    if orientation == 'portrait':
        a4_w, a4_h = mm_to_px(210, dpi), mm_to_px(297, dpi)
    else:
        a4_w, a4_h = mm_to_px(297, dpi), mm_to_px(210, dpi)

    m_top, m_bottom = mm_to_px(MARGIN_TOP, dpi), mm_to_px(MARGIN_BOTTOM, dpi)
    m_left, m_right = mm_to_px(MARGIN_LEFT, dpi), mm_to_px(MARGIN_RIGHT, dpi)

    valid_w = a4_w - (m_left + m_right)
    valid_h = a4_h - (m_top + m_bottom)

    img_w, img_h = pil_img.size
    scale = min(valid_w / img_w, valid_h / img_h)

    new_w, new_h = int(img_w * scale), int(img_h * scale)
    resized_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    a4_canvas = Image.new('L', (a4_w, a4_h), 255) 
    paste_x = m_left + (valid_w - new_w) // 2
    paste_y = m_top + (valid_h - new_h) // 2

    a4_canvas.paste(resized_img, (paste_x, paste_y))
    return a4_canvas

def create_a3_2in1(page_1, page_2, dpi, orientation):
    if orientation == 'portrait':
        a3_w, a3_h = mm_to_px(420, dpi), mm_to_px(297, dpi)
        offset_x, offset_y = mm_to_px(210, dpi), 0
    else:
        a3_w, a3_h = mm_to_px(297, dpi), mm_to_px(420, dpi)
        offset_x, offset_y = 0, mm_to_px(210, dpi)

    a3_canvas = Image.new('L', (a3_w, a3_h), 255)
    if page_1: a3_canvas.paste(page_1, (0, 0))
    if page_2: a3_canvas.paste(page_2, (offset_x, offset_y))
    return a3_canvas

def ToMonochrome(pil_img : Image.Image) -> Image.Image:
    """PIL画像をモノクロ2値化し、必要に応じてクロージングを適用する"""
    img = np.array(pil_img.convert('L'), 'f')
    img_bin = (img > 128).astype(np.uint8) * 255
    
    if ENABLE_FINAL_CLOSING:
        # OpenCVのクロージングは対象を白(255)として計算するため、一時的に色を反転
        img_inv = 255 - img_bin
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (FINAL_CLOSING_KERNEL, FINAL_CLOSING_KERNEL))
        closed_inv = cv2.morphologyEx(img_inv, cv2.MORPH_CLOSE, kernel)
        img_bin = 255 - closed_inv
        
    return Image.fromarray(img_bin)

def save_debug_image(file_path, orig_img, processed_img, angle, bbox, a4_pil, part_suffix="", reasonPoints=None, component_info=None, hist_img=None):
    if not os.path.exists(DEBUG_FOLDER):
        os.makedirs(DEBUG_FOLDER)

    canvas = np.ones((2700, 2400, 3), dtype=np.uint8) * 240

    def put_img_with_info(img, x_offset, y_offset, max_w, max_h, title, info_lines):
        disp_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
        h, w = disp_img.shape[:2]
        scale = min(max_w / w, max_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(disp_img, (new_w, new_h))

        px = x_offset + (max_w - new_w) // 2
        py = y_offset + (max_h - new_h) // 2 + 50
        canvas[py:py+new_h, px:px+new_w] = resized
        cv2.rectangle(canvas, (px, py), (px+new_w, py+new_h), (100, 100, 100), 2)

        y_text = y_offset + 30
        cv2.putText(canvas, title, (x_offset + 10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
        
        y_text += 35
        for line in info_lines:
            cv2.putText(canvas, line, (x_offset + 20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 50, 50), 2, cv2.LINE_AA)
            y_text += 30

    w_area, h_area = 1150, 800

    put_img_with_info(orig_img, 50, 50, w_area, h_area, "1. Original Image", [f"File: {os.path.basename(file_path)}"])
    
    rot_info = [f"Angle: {angle:.2f} deg"] if ENABLE_DESKEW else ["Deskew: Disabled"]
    put_img_with_info(processed_img, 1200, 50, w_area, h_area, "2. Deskewed & Adjusted", rot_info)

    bbox_img = processed_img.copy()
    if len(bbox_img.shape) == 2: bbox_img = cv2.cvtColor(bbox_img, cv2.COLOR_GRAY2BGR)
    
    if component_info:
        for comp in component_info:
            cx, cy, cw, ch = comp['rect']
            area = comp['area']
            is_kept = comp['is_kept']
            
            color = (0, 255, 0) if is_kept else (0, 0, 255)
            cv2.rectangle(bbox_img, (cx, cy), (cx+cw, cy+ch), color, 2)
            
            if not is_kept:
                cv2.putText(bbox_img, f"A:{area}", (cx, max(cy-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if bbox:
        bx, by, bw, bh = bbox
        cv2.rectangle(bbox_img, (bx, by), (bx+bw, by+bh), (255, 0, 0), 5) 
        crop_info = [f"Final BBox (Padded): x={bx}, y={by}, w={bw}, h={bh}"]
        
        if reasonPoints is not None:
            for idx, point in enumerate(reasonPoints):
                cv2.circle(bbox_img, point, 30, (255, 0, 0), -1) 
    else:
        crop_info = ["BBox: Not Found"]
    
    put_img_with_info(bbox_img, 50, 900, w_area, h_area, "3. Cropping (Green:Kept, Red:Noise)", crop_info)

    a4_cv = np.array(a4_pil)
    a4_info = [f"Orientation: {PAGE_ORIENTATION}", f"Final Closing: {'Enabled' if ENABLE_FINAL_CLOSING else 'Disabled'}"]
    put_img_with_info(a4_cv, 1200, 900, w_area, h_area, "4. Resized & Padded (A4 Canvas)", a4_info)

    if hist_img is not None:
        put_img_with_info(hist_img, 50, 1750, 1150, 800, "5. Pixel Histogram", ["Blue: Black Point, Red: White Point (Background)"])

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    out_path = os.path.join(DEBUG_FOLDER, f"{base_name}{part_suffix}_debug.jpg")
    cv2.imwrite(out_path, canvas)

def main():
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)

    extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.bmp']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))
    files.sort()

    if not files:
        print(f"エラー: '{INPUT_FOLDER}' フォルダに画像が見つかりません。")
        return

    print(f"{len(files)} 枚の画像を処理します...")
    processed_a4_pages = []

    for file_path in files:
        print(f"Processing : {os.path.basename(file_path)}")
        file_bytes = np.fromfile(file_path, dtype=np.uint8)
        cv_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        if cv_img is None: continue
        orig_img = cv_img.copy()

        # STEP 0: ヒストグラム解析とダイナミックレンジ補正
        white_point = MANUAL_WHITE_POINT
        hist_img = None
        
        if AUTO_WHITE_POINT:
            white_point, hist = calculate_dynamic_white_point(cv_img)
            hist_img = draw_histogram_image(hist, white_point, BLACK_POINT)
            
        cv_img = adjust_dynamic_range(cv_img, white_point)

        # STEP 1: 向きと傾きの調整
        angle = 0.0
        if ENABLE_DESKEW:
            cv_img, angle, _ = deskew_and_orient_score(cv_img)
            
        # STEP 2: 候補領域の算定と見開き判定・ノイズ消去
        split_images = detect_and_split_candidates(cv_img)

        # STEP 3 & 4: 各ページ領域ごとの処理
        for idx, sub_img in enumerate(split_images):
            cropped_cv, bbox, reasonPoints, component_info = crop_margins(sub_img)
            
            if bbox is None or bbox[2] < WIDTH_THRESHOLD or bbox[3] < HEIGHT_THRESHOLD:
                print(f"    -> クロップ領域が小さすぎるため、空白ページとして処理します。")
                cropped_cv = np.ones((mm_to_px(10, DPI), mm_to_px(10, DPI)), dtype=np.uint8) * 255
            
            pil_img = Image.fromarray(cropped_cv)
            a4_img = fit_to_a4(pil_img, DPI, PAGE_ORIENTATION)
            
            if ENABLE_DEBUG:
                part_suffix = f"_part{idx+1}" if len(split_images) > 1 else ""
                save_debug_image(file_path, orig_img, sub_img, angle, bbox, a4_img, part_suffix, reasonPoints, component_info, hist_img)
                
            processed_a4_pages.append(a4_img)

    print(f"\n出力モード '{OUTPUT_MODE}' ({PAGE_ORIENTATION}) で画像を生成します...")
    total_pages = len(processed_a4_pages)

    if OUTPUT_MODE == '1in1':
        for i, page in enumerate(processed_a4_pages):
            page = ToMonochrome(page)
            out_name = f"score_1in1_{i+1:03d}.png"
            page.save(os.path.join(OUTPUT_FOLDER, out_name), optimize=True)

    elif OUTPUT_MODE == '2in1':
        for i in range(0, total_pages, 2):
            p1 = ToMonochrome(processed_a4_pages[i])
            p2 = ToMonochrome(processed_a4_pages[i+1]) if (i + 1) < total_pages else None
            a3_img = create_a3_2in1(p1, p2, DPI, PAGE_ORIENTATION)
            out_name = f"score_2in1_{i//2 + 1:03d}.png"
            a3_img.save(os.path.join(OUTPUT_FOLDER, out_name), optimize=True)

    elif OUTPUT_MODE == 'booklet':
        bw, bh = (mm_to_px(210, DPI), mm_to_px(297, DPI)) if PAGE_ORIENTATION == 'portrait' else (mm_to_px(297, DPI), mm_to_px(210, DPI))
        blank_a4 = Image.new('L', (bw, bh), 255)
        
        while len(processed_a4_pages) % 4 != 0:
            processed_a4_pages.append(blank_a4)
            
        total_pages = len(processed_a4_pages)
        half = total_pages // 2
        
        for i in range(half):
            if BOOKLET_DIRECTION == 'left':
                idx_1, idx_2 = (total_pages - 1 - i, i) if i % 2 == 0 else (i, total_pages - 1 - i)
            else:
                idx_1, idx_2 = (i, total_pages - 1 - i) if i % 2 == 0 else (total_pages - 1 - i, i)
                    
            p1 = ToMonochrome(processed_a4_pages[idx_1])
            p2 = ToMonochrome(processed_a4_pages[idx_2])
            a3_img = create_a3_2in1(p1, p2, DPI, PAGE_ORIENTATION)
            
            side = "front" if i % 2 == 0 else "back"
            out_name = f"score_booklet_sheet{(i // 2) + 1:02d}_{side}.png"
            a3_img.save(os.path.join(OUTPUT_FOLDER, out_name), optimize=True)

    print("\nすべての処理が完了しました。")

if __name__ == "__main__":
    main()