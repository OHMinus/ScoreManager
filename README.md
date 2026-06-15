提供されたコード（`app.py`、`score_api.py`、`firebase_db.py`）に基づく、システムのセットアップと使用方法をまとめた `README.md` を作成しました。プロジェクトのルートディレクトリに配置してご活用ください。

---

# Score Manager (楽譜処理・管理システム)

本プロジェクトは、紙の楽譜や画像データをスキャン・最適化し、データベースで一元管理・出力するための統合システムです。ブラウザ上で動作するWebインターフェースと、ターミナルから直接実行できるCLIツールの両方を提供します。

## 主な機能

* **高度な楽譜画像処理**: 画像の傾き補正、見開きページの自動分割、インク密度による上下（正位置）の自動判定、余白の自動トリミング、二値化とノイズ除去を行います。
* **Web UIによる直感的な管理**: ファイルのアップロード、プレビュー確認、ページの並び替え、曲名・作曲者・楽器・イベントごとの管理と検索が可能です。
* **外部デバイス・コマンド連携**: `scanimage` を用いたスキャナからの直接取り込みや、`lp` コマンドを用いたプリンターへの直接印刷（A3見開きや小冊子レイアウト対応）が可能です。
* **クラウドデータ統合**: Google Driveへの画像アップロードおよびFirebase Realtime Databaseを用いたメタデータの保存に対応しています。※設定されていない場合はローカルのJSONファイル（`scores_db.json`）で動作します。
* **多様なエクスポート**: 処理済みの楽譜をPDFやZIPファイルとしてダウンロードできます。

---

## 動作環境とインストール

### 1. 必要なシステム要件

画像処理やOCR、外部機器連携のために以下のシステムパッケージが必要です。

* **Tesseract OCR**: 楽譜ヘッダーからのタイトル・楽器名自動抽出に使用します。
* **SANE (scanimage)**: スキャナから直接読み込む機能を使用する場合に必要です。
* **CUPS (lp)**: システムから直接印刷を行う機能を使用する場合に必要です。

### 2. Pythonパッケージのインストール

以下のコマンドで、必要なPythonライブラリをインストールしてください。

```bash
pip install Flask Pillow opencv-python pytesseract firebase-admin google-api-python-client google-auth requests numpy

```

### 3. 環境変数の設定 (クラウド連携を使用する場合)

FirebaseおよびGoogle Driveと連携する場合は、以下の環境変数を設定してください。

* `FIREBASE_CRED_PATH`: FirebaseサービスアカウントのJSONファイルパス
* `FIREBASE_DATABASE_URL`: Firebase Realtime DatabaseのURL
* `GOOGLE_DRIVE_CRED_PATH`: Google DriveサービスアカウントのJSONファイルパス
* `GOOGLE_DRIVE_FOLDER_ID`: 楽譜画像をアップロードするGoogle DriveのフォルダID

---

## 使い方

### On GUI

以下のコマンドでFlaskサーバーを起動します。

```bash
python app.py

```

起動後、ブラウザで `http://localhost:5000` にアクセスしてください。

**Web画面の主な操作フロー:**

1. **ホーム画面**: スキャン済みの画像/PDFをアップロード、またはスキャナ機能（「スキャンUI」）から直接取り込みます。
2. **プレビュー画面**: 自動分割・最適化されたページの順序を確認・変更し、画像の回転などを行います。ヘッダーから推測された曲名と楽器名が自動入力されます。
3. **保存**: 曲名、楽器、演奏会（年度/イベント名）を指定して保存します。
4. **一覧/検索画面**: 登録済みの楽譜を検索・閲覧し、PDF/ZIP形式での出力や印刷を実行します。

### On CUI

`score_api.py` は単独のCUIツールとしても動作します。

**1. 画像ファイルの最適化処理**

```bash
python score_api.py process input.jpg -o output.png

```

※デバッグ用の詳細な処理過程画像を出力したい場合は `--debug ./debug_out` オプションを付与してください。

**2. スキャナからの直接取り込みと処理**

```bash
python score_api.py scan -o scanned.png --dpi 300

```

※特定のデバイスを指定する場合は `--device <デバイス名>` を追加します。

**3. 登録データの確認**

```bash
# 登録されている楽譜一覧を表示
python score_api.py list-pieces

# 登録されているイベント一覧を表示
python score_api.py list-events

```

---

## ディレクトリ構造の概要

* `app.py`: WebアプリケーションのメインルーティングおよびUI制御。
* `score_api.py`: 画像処理のコアロジック、レイアウト生成、印刷制御、CLIインターフェース。
* `firebase_db.py`: Firebase DBおよびGoogle Drive APIとの通信・認証を行うデータ永続化層のアダプター。
* `scores_db.json`: ローカル環境でのデータベース（自動生成）。
* `static/temp/`: Webアップロード用の一時ファイル、プレビュー画像、デバッグ画像の保存先（定期的にクリーンアップされます）。