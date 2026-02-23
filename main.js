const { app, BrowserWindow } = require('electron');
const { spawn } = require('child_process');
const http = require('http');

let mainWindow;
let flaskProcess;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 900,
    title: "Score Processor",
    webPreferences: {
      nodeIntegration: false
    }
  });

  // メニューバーを隠す（アプリっぽくするため）
  mainWindow.setMenuBarVisibility(false);

  // Flaskサーバーが立ち上がるまでポーリング（監視）して待つ
  const checkServer = () => {
    http.get('http://127.0.0.1:5000', (res) => {
      // 応答があればサーバー起動完了とみなし、URLをロード
      mainWindow.loadURL('http://127.0.0.1:5000');
    }).on('error', (err) => {
      // まだ立ち上がっていない場合は200ミリ秒後に再チェック
      setTimeout(checkServer, 200);
    });
  };
  
  checkServer();

  mainWindow.on('closed', function () {
    mainWindow = null;
  });
}

app.on('ready', () => {
  // Flaskアプリを裏プロセスとして起動
  // ※仮想環境を使っている場合は、'python' を 'venv/bin/python' などに変更してください
  flaskProcess = spawn('python', ['app.py']);

  flaskProcess.stdout.on('data', (data) => {
    console.log(`Flask: ${data}`);
  });

  flaskProcess.stderr.on('data', (data) => {
    console.error(`Flask Error: ${data}`);
  });

  createWindow();
});

// すべてのウィンドウが閉じられたときの処理
app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit();
});

// アプリが終了する直前に、裏で動いているFlaskサーバーもキルする
app.on('will-quit', () => {
  if (flaskProcess) {
    flaskProcess.kill();
  }
});