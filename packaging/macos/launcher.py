from __future__ import annotations

import atexit
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path
from urllib.request import urlopen

try:
    import fcntl
except ImportError:
    fcntl = None


APP_NAME = "PAMalytics"
DEFAULT_PORT = 8510
SERVER_MODE_ARGUMENT = "--pamalytics-streamlit-server"
STARTUP_TIMEOUT_SECONDS = 90.0

_server_process: subprocess.Popen | None = None
_server_lock = threading.Lock()
_shutdown_started = False
_instance_lock_file = None


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


def _log_path() -> Path:
    return _runtime_root() / "launcher.log"


def _server_log_path() -> Path:
    return _runtime_root() / "streamlit.log"


def _log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with _log_path().open("a", encoding="utf-8") as handle:
            handle.write(f"[{stamp}] {message}\n")
    except OSError:
        pass


def _choose_port() -> int:
    raw = os.environ.get("PAMALYTICS_PORT", str(DEFAULT_PORT)).strip()
    try:
        port = int(raw)
    except ValueError:
        port = DEFAULT_PORT
    if port < 1024 or port > 65535:
        port = DEFAULT_PORT
    return port


def _app_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/"


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
        with urlopen(_app_url(port), timeout=1.0) as response:
            return response.status == 200
    except Exception:
        return False


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.25):
            return True
    except OSError:
        return False


def _wait_until_ready(port: int, timeout: float = STARTUP_TIMEOUT_SECONDS) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with _server_lock:
            process = _server_process
        if process is not None and process.poll() is not None:
            _log(f"Streamlit server exited during startup with code {process.returncode}.")
            return False
        if _health_ready(port) and _app_ready(port):
            return True
        time.sleep(0.25)
    _log(f"Timed out waiting for Streamlit on port {port}.")
    return False


def _open_browser(port: int) -> None:
    url = _app_url(port)
    if _health_ready(port) and _app_ready(port):
        webbrowser.open(url, new=1)
        return

    def wait_and_open() -> None:
        if _wait_until_ready(port):
            webbrowser.open(url, new=1)

    threading.Thread(target=wait_and_open, daemon=True).start()


def _configure_environment(home: Path) -> None:
    home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PAMALYTICS_HOME", str(home))
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")


def _acquire_instance_lock() -> bool:
    global _instance_lock_file

    lock_path = _runtime_root() / "pamalytics.lock"
    lock_file = lock_path.open("a+")

    if fcntl is not None:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_file.close()
            return False

    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(str(os.getpid()))
    lock_file.flush()
    _instance_lock_file = lock_file
    return True


def _release_instance_lock() -> None:
    global _instance_lock_file

    lock_file = _instance_lock_file
    _instance_lock_file = None
    if lock_file is None:
        return
    try:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        lock_file.close()
    except OSError:
        pass


def _server_command() -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, SERVER_MODE_ARGUMENT]
    return [sys.executable, str(Path(__file__).resolve()), SERVER_MODE_ARGUMENT]


def _start_server(port: int) -> None:
    global _server_process

    with _server_lock:
        if _server_process is not None and _server_process.poll() is None:
            return

        if _port_open(port):
            raise RuntimeError(
                f"Port {port} is already occupied by another process. "
                f"Quit that process or set PAMALYTICS_PORT to another port."
            )

        env = os.environ.copy()
        env["PAMALYTICS_PORT"] = str(port)
        env["ARROW_DEFAULT_MEMORY_POOL"] = "system"
        server_log = _server_log_path().open("a", encoding="utf-8")
        try:
            _server_process = subprocess.Popen(
                _server_command(),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            server_log.close()

        _log(f"Started Streamlit server process {_server_process.pid} on port {port}.")


def _stop_server() -> None:
    global _server_process, _shutdown_started

    with _server_lock:
        if _shutdown_started:
            return
        _shutdown_started = True
        process = _server_process
        _server_process = None

    if process is None or process.poll() is not None:
        return

    _log(f"Stopping Streamlit server process {process.pid}.")
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=8.0)
    except subprocess.TimeoutExpired:
        _log(f"Streamlit server process {process.pid} did not stop; sending SIGKILL.")
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass
    except ProcessLookupError:
        pass


def _run_streamlit_server() -> int:
    root = _resource_root()
    home = _user_root()
    port = _choose_port()
    _configure_environment(home)

    home_py = root / "code" / "Home.py"
    if not home_py.exists():
        raise FileNotFoundError(f"Cannot find PAMalytics entrypoint: {home_py}")

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


def _show_error(message: str) -> None:
    _log(message)
    try:
        from AppKit import NSAlert, NSAlertStyleCritical

        alert = NSAlert.alloc().init()
        alert.setMessageText_("PAMalytics could not start")
        alert.setInformativeText_(message)
        alert.setAlertStyle_(NSAlertStyleCritical)
        alert.runModal()
    except Exception:
        pass


def _run_macos_application() -> int:
    from AppKit import (
        NSApplication,
        NSApplicationActivationPolicyRegular,
        NSObject,
    )

    port = _choose_port()
    _configure_environment(_user_root())

    if not _acquire_instance_lock():
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if _health_ready(port) and _app_ready(port):
                webbrowser.open(_app_url(port), new=1)
                return 0
            time.sleep(0.25)
        _show_error(
            "Another PAMalytics process is running, but its local server did not become available. "
            "Quit PAMalytics from the Dock or Activity Monitor and open it again."
        )
        return 1

    class AppDelegate(NSObject):
        def applicationDidFinishLaunching_(self, notification) -> None:
            try:
                _start_server(port)
                _open_browser(port)
            except Exception as exc:
                _show_error(str(exc))
                NSApplication.sharedApplication().terminate_(None)

        def applicationShouldHandleReopen_hasVisibleWindows_(
            self,
            sender,
            has_visible_windows,
        ) -> bool:
            try:
                with _server_lock:
                    process = _server_process
                if process is None or process.poll() is not None:
                    global _shutdown_started
                    _shutdown_started = False
                    _start_server(port)
                _open_browser(port)
            except Exception as exc:
                _show_error(str(exc))
            return True

        def applicationWillTerminate_(self, notification) -> None:
            _stop_server()
            _release_instance_lock()

    app = NSApplication.sharedApplication()
    delegate = AppDelegate.alloc().init()
    app.setDelegate_(delegate)
    app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
    app.run()
    return 0


def _handle_signal(signum, frame) -> None:
    _stop_server()
    _release_instance_lock()
    raise SystemExit(0)


def main() -> int:
    if SERVER_MODE_ARGUMENT in sys.argv[1:]:
        return _run_streamlit_server()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    atexit.register(_stop_server)
    atexit.register(_release_instance_lock)

    try:
        return _run_macos_application()
    except Exception:
        _log(traceback.format_exc())
        raise


if __name__ == "__main__":
    raise SystemExit(main())
