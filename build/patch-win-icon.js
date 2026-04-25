const fs = require('node:fs');
const path = require('node:path');
const ResEdit = require('resedit');

function patchExecutableIcon(exePath, iconPath) {
  if (!fs.existsSync(exePath)) {
    throw new Error(`Cannot find packaged executable: ${exePath}`);
  }

  if (!fs.existsSync(iconPath)) {
    throw new Error(`Cannot find Windows icon: ${iconPath}`);
  }

  const exe = ResEdit.NtExecutable.from(fs.readFileSync(exePath), { ignoreCert: true });
  const resources = ResEdit.NtExecutableResource.from(exe);
  const iconFile = ResEdit.Data.IconFile.from(fs.readFileSync(iconPath));
  const icons = iconFile.icons.map((item) => item.data);
  const groups = ResEdit.Resource.IconGroupEntry.fromEntries(resources.entries);
  const targets = groups.length > 0 ? groups : [{ id: 1, lang: 1033 }];

  for (const group of targets) {
    ResEdit.Resource.IconGroupEntry.replaceIconsForResource(resources.entries, group.id, group.lang, icons);
  }

  resources.outputResource(exe);
  fs.writeFileSync(exePath, Buffer.from(exe.generate()));
}

exports.default = async function patchWinIcon(context) {
  if (context.electronPlatformName !== 'win32') {
    return;
  }

  const productFilename = context.packager.appInfo.productFilename;
  const exePath = path.join(context.appOutDir, `${productFilename}.exe`);
  const iconPath = path.join(context.packager.projectDir, 'electron', 'assets', 'hash-icon.ico');

  patchExecutableIcon(exePath, iconPath);
  console.log(`[patch-win-icon] Applied ${iconPath} to ${exePath}`);
};
