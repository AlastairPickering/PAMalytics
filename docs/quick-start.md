# Quick start

## Prerequisites

Python 3.9–3.12, macOS or Windows.

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

## macOS

Double-click `pamalytics_macos_launcher.command`.

It will:

- create `.venv`
- install `requirements.txt`
- launch Streamlit on port 8510
- open the app at `http://localhost:8510`

On the first run Mac security settings will likely block launch. You'll need to navigate to
System Settings → Privacy & Security → Security → “Allow applications downloaded from App Store and identified developers”.

## Windows

Double-click `pamalytics_windows_launcher.bat`
(or run `python scripts\launch_dashboard.py`).

Same behaviour: venv + requirements + Streamlit on port 8510.

## Manual launch

## Manual launch

```bash
python -m venv .venv
. .venv/bin/activate          # macOS
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run scripts/Dashboard.py --server.port 8510
```

