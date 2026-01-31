# Quick start

## Prerequisites

Python 3.9–3.12, macOS or Windows. :contentReference[oaicite:4]{index=4}

## macOS

Double-click `pamalytics_macos_launcher.command`.

On first run you may need to allow it in:
System Settings → Privacy & Security → Security → “Allow applications downloaded from App Store and identified developers”. :contentReference[oaicite:5]{index=5}

It will:
- create `.venv`
- install `requirements.txt`
- launch Streamlit on port 8510
- open the app at `http://localhost:8510` :contentReference[oaicite:6]{index=6}

## Windows

Double-click `pamalytics_windows_launcher.bat`
(or run `python scripts\launch_dashboard.py`).

Same behaviour: venv + requirements + Streamlit on port 8510. :contentReference[oaicite:7]{index=7}

## Manual launch

```bash
python -m venv .venv
. .venv/bin/activate          # macOS
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run scripts/Dashboard.py --server.port 8503
