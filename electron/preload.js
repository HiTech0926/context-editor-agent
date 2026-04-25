const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  minimize: () => ipcRenderer.send('window:minimize'),
  maximize: () => ipcRenderer.send('window:maximize'),
  close: () => ipcRenderer.send('window:close'),
  selectFolder: () => ipcRenderer.invoke('dialog:select-folder'),
  openProjectParentFolder: (targetPath) => ipcRenderer.invoke('shell:open-project-parent-folder', targetPath),
  isElectron: true,
});
