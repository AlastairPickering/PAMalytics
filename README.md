# PAMalytics — no-code PAM classifier validation and analytics
[![CI (macOS & Windows)](https://github.com/AlastairPickering/PAMalytics/actions/workflows/ci.yaml/badge.svg)](https://github.com/AlastairPickering/PAMalytics/actions/workflows/ci.yaml)

PAMalytics is an open-source, no-code, classifier-agnostic dashboard for reviewing bioacoustic classifier outputs. It supports detection summaries, audio/spectrogram review, validation edits, uncertainty flags, threshold recalculation and export of validated results.

User guide: https://alastairpickering.github.io/PAMalytics/

<img width="256" height="256" alt="PAMalytics logo" src="docs/assets/brand/pamalytics-logo.png" />

## Install PAMalytics

Pre-built desktop applications are available from the latest GitHub release.

### macOS

Download:

```text
PAMalytics-v1.0.1-macOS-arm64.dmg
```

This build is for Apple Silicon Macs and is signed and notarised by Apple.

Open the DMG, drag PAMalytics into Applications, then launch it from Applications.

### Windows

Download:

```text
PAMalytics-v1.0.1-windows-x64-setup.exe
```

Run the installer, then launch PAMalytics from the Windows Start menu.

The Windows installer is currently unsigned, so Windows may display a security warning. Select **More info**, then **Run anyway**, after confirming that the installer was downloaded from the official PAMalytics GitHub release.

### Checksums

SHA-256 checksum files are provided alongside both installers:

- `PAMalytics-v1.0.1-macOS-arm64.dmg.sha256`
- `PAMalytics-v1.0.1-windows-x64-setup.exe.sha256`

## Building the macOS app from source

A packaged macOS app/DMG is built from the same source code.

To build the macOS app locally:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-build-macos.txt
python -m code.scripts.smoke_test
python packaging/macos/build_app.py
bash packaging/macos/build_dmg.sh
```

The DMG will be written to `release/`. 

## Features

- Interactive dashboard for summarising, validating and exporting classifier outputs
- Classifier-agnostic import workflow, including BirdNET and BatDetect2 adapters
- Audio playback and high-resolution spectrogram review
- Card-based validation workflow with uncertainty flags and species/presence correction
- Changed and uncertain detections tracked separately
- Recalculate page for exploring threshold impacts
- Non-destructive validation outputs that preserve original classifier predictions
- Local-first operation: projects and validation state are stored on the user’s machine

## Local data location

By default, user projects and app state are stored outside the source folder:

```text
~/Documents/PAMalytics/
```

For testing, this can be overridden with:

```bash
export PAMALYTICS_HOME=/tmp/pamalytics_test_home
```

## Developer source launch

Manual launch on macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m streamlit run code/Home.py --server.port 8510
```

Manual launch on Windows:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m streamlit run code\Home.py --server.port 8510
```

## Testing

```bash
python -m code.scripts.smoke_test
```

See `docs/release-testing.md` and `packaging/macos/README.md` for release testing and macOS packaging notes.

## Repository layout

```text
code/                    Shared PAMalytics Streamlit application code
docs/                    User documentation site
packaging/macos/         macOS app/DMG build scripts and icons
packaging/windows/       Windows application and installer build files
pamalytics_*_launcher*   Alternative source launchers for development use
VC_redist.x64.exe        Windows runtime prerequisite used by launchers
.uv-bin/                 Bundled uv binaries used by uv launchers
```

## Licence

PAMalytics is released under the GNU General Public License v3.0. See `LICENSE`.
