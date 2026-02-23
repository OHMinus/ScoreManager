# ScoreManager

楽譜をきれいに電子化（PDF化）・管理・印刷するための補助アプリケーションです。
Flask (Python) によるWebサーバーと、Electron によるデスクトップアプリケーションとして構成されています。

## 主な機能

- **楽譜の取り込み**: 画像ファイルのアップロード、または接続されたスキャナーからの直接取り込み（`scanimage` コマンド使用）。
- **自動画像処理**: 傾き補正、余白の自動削除、レベル補正、五線譜の向き判定などを自動で行います。
- **データベース管理**: 曲名、作曲者、編曲者、楽器パート、演奏行事（年度・イベント名）などの情報を紐付けて管理します。
- **プレビューと編集**: 取り込んだ画像の確認、ページの並べ替え、回転操作が可能です。
- **出力・印刷**: A4/A3用紙への印刷、小冊子形式（Booklet）での配置、PDF/ZIP形式でのダウンロードに対応しています。

## 動作環境・前提条件

- **OS**: Linux (Ubuntu等) を推奨（スキャナー機能が `scanimage` コマンドに依存しているため）
- **Python**: 3.8 以上
- **Node.js**: Electronの実行に必要
- **外部ライブラリ**:
    - **Tesseract OCR**: 文字認識（タイトル推測など）に使用
    - **SANE**: スキャナー制御に使用

### Ubuntuでのセットアップ例

```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-eng sane-utils
```

## インストール手順

1.  **リポジトリのクローン**
    ```bash
    git clone https://github.com/OHMinus/ScoreManager.git
    cd ScoreManager
    ```

2.  **Python依存パッケージのインストール**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Node.js依存パッケージのインストール**
    ```bash
    npm install
    ```

## 使用方法

### デスクトップアプリとして起動 (推奨)

Electronを使用して、デスクトップアプリケーションとして起動します。

```bash
npm start
```

### Webサーバーとして起動

ブラウザからアクセスして使用する場合や、サーバーとしてホストする場合に使用します。

```bash
python app.py
```
起動後、ブラウザで `http://localhost:5000` にアクセスしてください。

## ファイル構成

- `app.py`: Flaskアプリケーションのエントリーポイント。Webサーバー機能を提供します。
- `score_api.py`: 画像処理、データベース操作、印刷レイアウト生成などのコアロジック。
- `main.js`: Electronのメインプロセス用スクリプト。
- `templates/`: HTMLテンプレートファイル。
- `static/`: 静的ファイルディレクトリ。
    - `css/style.css`: アプリケーション全体のスタイルシート。
    - `js/ui.js`: フロントエンドのインタラクション制御ロジック。
- `scores_db.json`: 登録された楽譜データを保存するJSONデータベース（初回保存時に自動生成されます）。

## ライセンス

ISC License
