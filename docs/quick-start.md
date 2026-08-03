# Quick start

## Prerequisites

Python 3.12 for source installs. macOS or Windows.

For ordinary users, use a packaged PAMalytics release where available. The source launchers are retained for development and testing.

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

## macOS source launch

Double-click `pamalytics_macos_launcher_uv.command`.

It will:

- create `.venv`
- install `requirements.txt`
- launch Streamlit on port 8510
- open the app at `http://localhost:8510`

User projects and login state are stored outside the code folder at `~/Documents/PAMalytics/`.

## Windows source launch

Double-click `pamalytics_windows_launcher_uv.bat`.

## Manual launch

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run code/Home.py --server.port 8510
```

## Clean test launch

```bash
export PAMALYTICS_HOME=/tmp/pamalytics_clean_test
python -m streamlit run code/Home.py --server.port 8510
```
