from __future__ import annotations

import atexit
import os
import signal
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path
from urllib.request import urlopen

try:
    import fcntl
except ImportError:
    fcntl = None


APP_NAME = "PAMalytics"
DEFAULT_PORT = 8510


def _resource_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[2]


def _user_root() -> Path:
    configured = os.environ.get("PAMALYTICS_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.home() / "Documents" / APP_NAME


def _runtime_root() -> Path:
    root = Path.home() / "Library" / "Application Support" / APP_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def _choose_port() -> int:
    raw = os.environ.get("PAMALYTICS_PORT", str(DEFAULT_PORT)).strip()
    try:
        port = int(raw)
    except ValueError:
        port = DEFAULT_PORT
    if port < 1024 or port > 65535:
        port = DEFAULT_PORT
    return port


def _health_ready(port: int) -> bool:
    try:
        with urlopen(
            f"http://127.0.0.1:{port}/_stcore/health",
            timeout=0.5,
        ) as response:
            return response.status == 200
    except Exception:
        return False


def _app_ready(port: int) -> bool:
    try:
        with urlopen(
            f"http://127.0.0.1:{port}/",
            timeout=1.0,
        ) as response:
            return response.status == 200
    except Exception:
        return False


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.25):
            return True
    except OSError:
        return False


def _open_when_ready(port: int) -> None:
    url = f"http://127.0.0.1:{port}/"
    for _ in range(120):
        if _health_ready(port) and _app_ready(port):
            webbrowser.open(url, new=1)
            return
        time.sleep(0.5)


def _read_lock_pid(lock_file) -> int | None:
    try:
        lock_file.seek(0)
        raw = lock_file.read().strip()
        return int(raw) if raw else None
    except (OSError, ValueError):
        return None


def _stop_stale_process(pid: int | None) -> None:
    if pid is None or pid == os.getpid():
        return

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        return

    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)

    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _acquire_lock(port: int):
    lock_path = _runtime_root() / "pamalytics.lock"

    for _ in range(2):
        lock_file = open(lock_path, "a+")

        if fcntl is None:
            lock_file.seek(0)
            lock_file.truncate()
            lock_file.write(str(os.getpid()))
            lock_file.flush()
            return lock_file

        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_file.seek(0)
            lock_file.truncate()
            lock_file.write(str(os.getpid()))
            lock_file.flush()
            return lock_file
        except BlockingIOError:
            stale_pid = _read_lock_pid(lock_file)

            if _health_ready(port) and _app_ready(port):
                lock_file.close()
                webbrowser.open(f"http://127.0.0.1:{port}/", new=1)
                raise SystemExit(0)

            lock_file.close()
            _stop_stale_process(stale_pid)

            deadline = time.time() + 5.0
            while time.time() < deadline and _port_open(port):
                time.sleep(0.1)

    raise RuntimeError(
        f"PAMalytics could not clear the previous process on port {port}."
    )


def _configure_environment(home: Path) -> None:
    home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PAMALYTICS_HOME", str(home))
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")


def _run_streamlit(home_py: Path, port: int) -> int:
    import streamlit.web.cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        str(home_py),
        "--server.port",
        str(port),
        "--server.address",
        "127.0.0.1",
        "--server.headless",
        "true",
        "--server.runOnSave",
        "false",
        "--server.fileWatcherType",
        "none",
        "--browser.gatherUsageStats",
        "false",
        "--global.developmentMode",
        "false",
    ]
    stcli.main()
    return 0


def main() -> int:
    root = _resource_root()
    home = _user_root()
    port = _choose_port()
    lock_file = _acquire_lock(port)
    atexit.register(lock_file.close)
    _configure_environment(home)

    home_py = root / "code" / "Home.py"
    if not home_py.exists():
        raise FileNotFoundError(f"Cannot find PAMalytics entrypoint: {home_py}")

    opener = threading.Thread(
        target=_open_when_ready,
        args=(port,),
        daemon=True,
    )
    opener.start()
    return _run_streamlit(home_py, port)


if __name__ == "__main__":
    raise SystemExit(main())
