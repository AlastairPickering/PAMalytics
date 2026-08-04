#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERSION="${1:-}"
SIGNING_IDENTITY="${PAMALYTICS_CODESIGN_IDENTITY:-Developer ID Application: Alastair Pickering (FKXA4C7S9Z)}"
NOTARY_PROFILE="${PAMALYTICS_NOTARY_PROFILE:-PAMalytics-notary}"
APP="$ROOT/dist/PAMalytics.app"
ICON="$ROOT/packaging/macos/PAMalytics.icns"
DMG_ROOT="$ROOT/release/dmg-root"
DMG_OUT="$ROOT/release/PAMalytics-${VERSION}-macOS-arm64.dmg"
CHECKSUM_OUT="$DMG_OUT.sha256"

if [[ -z "$VERSION" ]]; then
  echo "Usage: bash packaging/macos/build_dmg.sh <version>" >&2
  echo "Example: bash packaging/macos/build_dmg.sh v1.0.0" >&2
  exit 1
fi

if [[ ! -d "$APP" ]]; then
  echo "Missing app bundle: $APP" >&2
  echo "Run: python3.12 packaging/macos/build_app.py" >&2
  exit 1
fi

if [[ ! -f "$ICON" ]]; then
  echo "Missing volume icon: $ICON" >&2
  exit 1
fi

security find-identity -v -p codesigning | grep -Fq "$SIGNING_IDENTITY" || {
  echo "Signing identity not found: $SIGNING_IDENTITY" >&2
  exit 1
}

codesign --verify --deep --strict --verbose=2 "$APP"

mkdir -p "$ROOT/release"
rm -rf "$DMG_ROOT"
rm -f "$DMG_OUT" "$CHECKSUM_OUT"
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

codesign \
  --force \
  --timestamp \
  --sign "$SIGNING_IDENTITY" \
  "$DMG_OUT"

codesign --verify --strict --verbose=2 "$DMG_OUT"

xcrun notarytool submit "$DMG_OUT" \
  --keychain-profile "$NOTARY_PROFILE" \
  --wait

xcrun stapler staple "$DMG_OUT"
xcrun stapler validate "$DMG_OUT"
hdiutil verify "$DMG_OUT"
spctl --assess --type open --context context:primary-signature --verbose=4 "$DMG_OUT"

shasum -a 256 "$DMG_OUT" | tee "$CHECKSUM_OUT"

rm -rf "$DMG_ROOT"

echo
echo "Signed, notarised and stapled release created:"
echo "$DMG_OUT"
echo "$CHECKSUM_OUT"
