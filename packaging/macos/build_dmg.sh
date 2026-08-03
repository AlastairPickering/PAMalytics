#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERSION="${1:-}"

if [[ -z "$VERSION" ]]; then
  echo "Usage: bash packaging/macos/build_dmg.sh <version>" >&2
  echo "Example: bash packaging/macos/build_dmg.sh v1.0.0-rc5" >&2
  exit 1
fi

APP="$ROOT/dist/PAMalytics.app"
ICON="$ROOT/packaging/macos/PAMalytics.icns"
DMG_ROOT="$ROOT/release/dmg-root"
DMG_OUT="$ROOT/release/PAMalytics-${VERSION}-macOS-arm64.dmg"

if [[ ! -d "$APP" ]]; then
  echo "Missing app bundle: $APP" >&2
  exit 1
fi

if [[ ! -f "$ICON" ]]; then
  echo "Missing volume icon: $ICON" >&2
  exit 1
fi

echo "Verifying application bundle..."
codesign --verify --deep --strict --verbose=2 "$APP"

mkdir -p "$ROOT/release"
rm -rf "$DMG_ROOT"
rm -f "$DMG_OUT"
mkdir -p "$DMG_ROOT"

ditto "$APP" "$DMG_ROOT/PAMalytics.app"
cp "$ICON" "$DMG_ROOT/.VolumeIcon.icns"
ln -s /Applications "$DMG_ROOT/Applications"

if command -v SetFile >/dev/null 2>&1; then
  SetFile -a C "$DMG_ROOT"
else
  echo "Warning: SetFile is unavailable; the custom volume icon may not display." >&2
fi

hdiutil create \
  -volname "PAMalytics" \
  -srcfolder "$DMG_ROOT" \
  -ov \
  -format UDZO \
  "$DMG_OUT"

echo
echo "Created:"
echo "$DMG_OUT"