/// <reference types="vite/client" />

interface Window {
  electronAPI?: {
    minimize?: () => void;
    maximize?: () => void;
    close?: () => void;
    selectFolder?: () => Promise<{ canceled: boolean; path?: string; name?: string }>;
    openProjectParentFolder?: (path: string) => Promise<{ ok: boolean; error?: string }>;
    isElectron?: boolean;
  };
}
