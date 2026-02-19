import sys
from pathlib import Path

import streamlit.web.cli as stcli

PKG_ROOT = Path(__file__).parent


def run():
    sys.argv = [
        "streamlit",
        "run",
        str(PKG_ROOT / "app.py"),
        "--global.developmentMode=false",
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    run()
