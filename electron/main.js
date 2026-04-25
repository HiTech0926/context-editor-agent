const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const { spawn } = require('node:child_process');
const fs = require('node:fs');
const http = require('node:http');
const net = require('node:net');
const path = require('node:path');

let mainWindow = null;
let backendProcess = null;
let zoomFactor = 1;

const BACKEND_HOST = '127.0.0.1';
const MIN_ZOOM_FACTOR = 0.67;
const MAX_ZOOM_FACTOR = 1.5;
const ZOOM_STEP = 0.1;

app.setPath('userData', path.join(app.getPath('appData'), app.isPackaged ? 'hashcode' : 'hashcode-dev'));

function getAppRoot() {
  return app.isPackaged ? app.getAppPath() : path.resolve(__dirname, '..');
}

function getReactIndexPath(appRoot) {
  return path.join(appRoot, 'react_app', 'dist', 'index.html');
}

function getAppIconPath(appRoot) {
  const iconName = process.platform === 'win32' ? 'hash-icon.ico' : 'hash-icon.png';
  return path.join(appRoot, 'electron', 'assets', iconName);
}

function getPythonCandidates(appRoot) {
  const candidates = [];

  if (process.platform === 'win32') {
    candidates.push({
      command: path.join(appRoot, '.venv', 'Scripts', 'python.exe'),
      argsPrefix: [],
      mustExist: true,
    });
    candidates.push({ command: 'python', argsPrefix: [], mustExist: false });
    candidates.push({ command: 'py', argsPrefix: ['-3'], mustExist: false });
  } else {
    candidates.push({
      command: path.join(appRoot, '.venv', 'bin', 'python'),
      argsPrefix: [],
      mustExist: true,
    });
    candidates.push({ command: 'python3', argsPrefix: [], mustExist: false });
    candidates.push({ command: 'python', argsPrefix: [], mustExist: false });
  }

  return candidates;
}

function getFreePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.once('error', reject);
    server.listen(0, BACKEND_HOST, () => {
      const address = server.address();
      const port = typeof address === 'object' && address ? address.port : 8765;
      server.close(() => resolve(port));
    });
  });
}

function waitForBackend(port, timeoutMs = 30000) {
  const startedAt = Date.now();

  return new Promise((resolve, reject) => {
    const tick = () => {
      const request = http.get(
        {
          hostname: BACKEND_HOST,
          port,
          path: '/api/init',
          timeout: 1200,
        },
        (response) => {
          response.resume();
          if (response.statusCode && response.statusCode < 500) {
            resolve();
            return;
          }
          retry();
        },
      );

      request.on('error', retry);
      request.on('timeout', () => {
        request.destroy();
        retry();
      });
    };

    const retry = () => {
      if (Date.now() - startedAt > timeoutMs) {
        reject(new Error('Python backend did not become ready in time.'));
        return;
      }
      setTimeout(tick, 300);
    };

    tick();
  });
}

function spawnBackendWithCandidate(candidate, appRoot, port) {
  const scriptPath = path.join(appRoot, 'web_server.py');
  const dataDir = path.join(app.getPath('userData'), 'data');
  const env = {
    ...process.env,
    HASH_WEB_HOST: BACKEND_HOST,
    HASH_WEB_PORT: String(port),
    HASH_DATA_DIR: dataDir,
    HASH_ALLOW_LEGACY_SETTINGS: app.isPackaged ? '0' : '1',
    PYTHONIOENCODING: 'utf-8',
  };

  const args = [...candidate.argsPrefix, scriptPath];
  return spawn(candidate.command, args, {
    cwd: appRoot,
    env,
    windowsHide: true,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
}

async function startBackend() {
  const appRoot = getAppRoot();
  const scriptPath = path.join(appRoot, 'web_server.py');
  const reactIndexPath = getReactIndexPath(appRoot);

  if (!fs.existsSync(scriptPath)) {
    throw new Error(`Cannot find backend script: ${scriptPath}`);
  }

  if (!fs.existsSync(reactIndexPath)) {
    throw new Error('React build not found. Run npm run build:react first.');
  }

  const port = await getFreePort();
  const candidates = getPythonCandidates(appRoot);
  let lastError = null;

  for (const candidate of candidates) {
    if (candidate.mustExist && !fs.existsSync(candidate.command)) {
      continue;
    }

    try {
      const child = spawnBackendWithCandidate(candidate, appRoot, port);
      backendProcess = child;

      child.stdout.on('data', (chunk) => {
        console.log(`[backend] ${chunk.toString().trim()}`);
      });

      child.stderr.on('data', (chunk) => {
        console.error(`[backend] ${chunk.toString().trim()}`);
      });

      child.once('exit', (code, signal) => {
        if (backendProcess === child) {
          backendProcess = null;
        }
        console.log(`[backend] exited with code ${code ?? 'null'} signal ${signal ?? 'null'}`);
      });

      await waitForBackend(port);
      return { port };
    } catch (error) {
      lastError = error;
      if (backendProcess) {
        backendProcess.kill();
        backendProcess = null;
      }
    }
  }

  throw lastError || new Error('Cannot start Python backend. Make sure Python 3.10+ is installed.');
}

function createWindow(port) {
  const appRoot = getAppRoot();

  mainWindow = new BrowserWindow({
    width: 1440,
    height: 960,
    minWidth: 1100,
    minHeight: 720,
    icon: getAppIconPath(appRoot),
    frame: false,
    titleBarStyle: 'hidden',
    backgroundColor: '#0b0f17',
    show: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  mainWindow.webContents.on('before-input-event', (event, input) => {
    if (input.type !== 'keyDown' || (!input.control && !input.meta)) {
      return;
    }

    const key = (input.key || '').toLowerCase();
    const code = input.code || '';
    const isZoomIn = key === '+' || key === '=' || code === 'Equal' || code === 'NumpadAdd';
    const isZoomOut = key === '-' || key === '_' || code === 'Minus' || code === 'NumpadSubtract';
    const isZoomReset = key === '0' || code === 'Digit0' || code === 'Numpad0';

    if (!isZoomIn && !isZoomOut && !isZoomReset) {
      return;
    }

    event.preventDefault();

    if (isZoomReset) {
      zoomFactor = 1;
    } else if (isZoomIn) {
      zoomFactor = Math.min(MAX_ZOOM_FACTOR, Number((zoomFactor + ZOOM_STEP).toFixed(2)));
    } else {
      zoomFactor = Math.max(MIN_ZOOM_FACTOR, Number((zoomFactor - ZOOM_STEP).toFixed(2)));
    }

    mainWindow?.webContents.setZoomFactor(zoomFactor);
  });

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  void mainWindow.loadURL(`http://${BACKEND_HOST}:${port}/react`);

  if (!app.isPackaged && process.env.ELECTRON_OPEN_DEVTOOLS === '1') {
    mainWindow.webContents.openDevTools({ mode: 'detach' });
  }
}

function stopBackend() {
  if (!backendProcess) {
    return;
  }

  const child = backendProcess;
  backendProcess = null;
  child.kill();
}

ipcMain.on('window:minimize', () => {
  mainWindow?.minimize();
});

ipcMain.on('window:maximize', () => {
  if (!mainWindow) {
    return;
  }

  if (mainWindow.isMaximized()) {
    mainWindow.unmaximize();
  } else {
    mainWindow.maximize();
  }
});

ipcMain.on('window:close', () => {
  mainWindow?.close();
});

ipcMain.handle('dialog:select-folder', async () => {
  if (!mainWindow) {
    return { canceled: true };
  }

  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory'],
  });

  if (result.canceled || !result.filePaths.length) {
    return { canceled: true };
  }

  const selectedPath = result.filePaths[0];
  return {
    canceled: false,
    path: selectedPath,
    name: path.basename(selectedPath),
  };
});

ipcMain.handle('shell:open-project-parent-folder', async (_event, targetPath) => {
  const safeTargetPath = typeof targetPath === 'string' ? targetPath : '';
  if (!safeTargetPath) {
    return { ok: false, error: '项目没有本地文件夹路径。' };
  }

  try {
    const resolvedPath = path.resolve(safeTargetPath);
    if (!fs.existsSync(resolvedPath)) {
      return { ok: false, error: '本地文件夹不存在。' };
    }

    const parentPath = path.dirname(resolvedPath);
    const errorMessage = await shell.openPath(parentPath);
    return errorMessage ? { ok: false, error: errorMessage } : { ok: true };
  } catch (error) {
    return {
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    };
  }
});

app.whenReady().then(async () => {
  try {
    const { port } = await startBackend();
    createWindow(port);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    dialog.showErrorBox('hashcode 启动失败', message);
    app.quit();
  }
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', async () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    const { port } = await startBackend();
    createWindow(port);
  }
});

app.on('before-quit', () => {
  stopBackend();
});
