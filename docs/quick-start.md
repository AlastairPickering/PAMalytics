# Quick start

## Recommended installation

For most users, the recommended way to run PAMalytics is to install the packaged desktop application.

### macOS

1. Download `PAMalytics-v1.0.4-macOS-arm64.dmg`.
2. Open the DMG.
3. Drag **PAMalytics** into **Applications**.
4. Launch PAMalytics from Applications.

The macOS application is signed and notarised by Apple.

### Windows

1. Download `PAMalytics-v1.0.4-windows-x64-setup.exe`.
2. Run the installer.
3. Launch PAMalytics from the Windows Start menu.

The Windows installer is unsigned, so Windows may display a security warning. After confirming that the installer came from the official PAMalytics release, select **More info** and then **Run anyway**.

## Getting started video

<div class="video-wrapper">
  <iframe
    src="https://www.youtube.com/embed/lZZuCw-uo2o"
    title="PAMalytics getting started"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
</div>

## Alternative source launch

The packaged desktop applications are the recommended way to use PAMalytics.

The source launchers below are retained as a backup for users who cannot use the packaged installers, and for development or testing. Python 3.12 is required.

### macOS source launch

Double-click `pamalytics_macos_launcher_uv.command`.

It will:

- create `.venv`
- install `requirements.txt`
- launch Streamlit on port 8510
- open the app at `http://localhost:8510`

User projects and login state are stored outside the code folder at `~/Documents/PAMalytics/`.

### Windows source launch

Double-click `pamalytics_windows_launcher_uv.bat`.

### Manual launch

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run code/Home.py --server.port 8510
```

### Clean test launch

```bash
export PAMALYTICS_HOME=/tmp/pamalytics_clean_test
python -m streamlit run code/Home.py --server.port 8510
```
