#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP="$ROOT/dist/PAMalytics.app"
ICON="$ROOT/packaging/macos/PAMalytics.icns"
DMG_ROOT="$ROOT/release/dmg-root"
DMG_OUT="$ROOT/release/PAMalytics-v23-unsigned.dmg"

if [[ ! -d "$APP" ]]; then
  echo "Missing app bundle: $APP" >&2
  echo "Run: python packaging/macos/build_app.py" >&2
  exit 1
fi

mkdir -p "$ROOT/release"
rm -rf "$DMG_ROOT" "$DMG_OUT"
mkdir -p "$DMG_ROOT"

cp -R "$APP" "$DMG_ROOT/"
cp "$ICON" "$DMG_ROOT/.VolumeIcon.icns"
ln -s /Applications "$DMG_ROOT/Applications"

if command -v SetFile >/dev/null 2>&1; then
  SetFile -a C "$DMG_ROOT"
else
  echo "Warning: SetFile not found; DMG volume icon may not display. Install Xcode command line tools if needed." >&2
fi

hdiutil create \
  -volname "PAMalytics" \
  -srcfolder "$DMG_ROOT" \
  -ov \
  -format UDZO \
  "$DMG_OUT"

echo "Created: $DMG_OUT"
