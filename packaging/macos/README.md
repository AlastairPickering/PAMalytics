# macOS packaging

This folder contains the macOS packaged-app build route for PAMalytics.

The source/no-code launchers at the repository root are intentionally kept there for users who run PAMalytics from a GitHub zip. The scripts in this folder are for release engineering.

## Build unsigned app and DMG

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-build-macos.txt
python -m code.scripts.smoke_test
python packaging/macos/build_app.py
bash packaging/macos/build_dmg.sh
```

Outputs:

```text
dist/PAMalytics.app
release/PAMalytics-v23-unsigned.dmg
```

The DMG includes the PAMalytics icon and an Applications shortcut.

## Signing and notarisation

Public macOS distribution requires an Apple Developer Program account, a Developer ID Application certificate, hardened-runtime signing and notarisation.

Conceptual sequence:

```bash
codesign --force --deep --options runtime --timestamp \
  --sign "Developer ID Application: <name> (<TEAMID>)" \
  dist/PAMalytics.app

bash packaging/macos/build_dmg.sh

codesign --force --timestamp \
  --sign "Developer ID Application: <name> (<TEAMID>)" \
  release/PAMalytics-v23-unsigned.dmg

xcrun notarytool submit release/PAMalytics-v23-unsigned.dmg \
  --keychain-profile <profile> \
  --wait

xcrun stapler staple release/PAMalytics-v23-unsigned.dmg
spctl -a -vvv -t install release/PAMalytics-v23-unsigned.dmg
```

Use `docs/release-testing.md` before signing and again after downloading the notarised DMG.

## Emergency stop for test builds

```bash
pkill -f PAMalytics
pkill -f streamlit
rm -rf dist/PAMalytics.app build/PAMalytics
```
