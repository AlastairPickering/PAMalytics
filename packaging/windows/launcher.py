from __future__ import annotations

import atexit
import ctypes
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path
from urllib.request import urlopen

APP_NAME = "PAMalytics"
APP_USER_MODEL_ID = "AlastairPickering.PAMalytics.Desktop.1"
DEFAULT_PORT = 8510
SERVER_MODE_ARGUMENT = "--pamalytics-streamlit-server"
STARTUP_TIMEOUT_SECONDS = 90.0

_server_process: subprocess.Popen | None = None
_server_lock = threading.Lock()
_shutdown_started = False
_mutex_handle = None


def _set_windows_app_identity() -> None:
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except Exception:
        pass


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
    base = os.environ.get("LOCALAPPDATA")
    root = Path(base) / APP_NAME if base else Path.home() / "AppData" / "Local" / APP_NAME
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


def _acquire_instance_mutex() -> bool:
    global _mutex_handle
    kernel32 = ctypes.windll.kernel32
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    handle = kernel32.CreateMutexW(None, False, "Local\\PAMalyticsDesktopController")
    if not handle:
        raise ctypes.WinError()
    _mutex_handle = handle
    return kernel32.GetLastError() != 183


def _release_instance_mutex() -> None:
    global _mutex_handle
    if _mutex_handle:
        ctypes.windll.kernel32.CloseHandle(_mutex_handle)
        _mutex_handle = None


def _server_command() -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, SERVER_MODE_ARGUMENT]
    return [sys.executable, str(Path(__file__).resolve()), SERVER_MODE_ARGUMENT]


def _start_server(port: int) -> None:
    global _server_process, _shutdown_started

    with _server_lock:
        if _server_process is not None and _server_process.poll() is None:
            return
        if _port_open(port):
            raise RuntimeError(
                f"Port {port} is already occupied by another process. "
                "Close that process or set PAMALYTICS_PORT to another port."
            )

        _shutdown_started = False
        env = os.environ.copy()
        env["PAMALYTICS_PORT"] = str(port)
        env["ARROW_DEFAULT_MEMORY_POOL"] = "system"
        server_log = _server_log_path().open("a", encoding="utf-8")
        creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        try:
            _server_process = subprocess.Popen(
                _server_command(),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                creationflags=creation_flags,
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
        process.terminate()
        process.wait(timeout=8.0)
    except subprocess.TimeoutExpired:
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
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
        import tkinter.messagebox as messagebox

        messagebox.showerror("PAMalytics could not start", message)
    except Exception:
        pass


def _run_windows_application() -> int:
    import tkinter as tk
    from tkinter import ttk

    port = _choose_port()
    _configure_environment(_user_root())

    if not _acquire_instance_mutex():
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if _health_ready(port) and _app_ready(port):
                webbrowser.open(_app_url(port), new=1)
                return 0
            time.sleep(0.25)
        _show_error(
            "Another PAMalytics process is running, but its local server did not become available. "
            "Close PAMalytics in Task Manager and open it again."
        )
        return 1

    _set_windows_app_identity()

    root = tk.Tk()
    root.title(APP_NAME)

    resource_root = _resource_root()
    icon_ico = resource_root / "packaging" / "windows" / "PAMalytics.ico"
    icon_png = resource_root / "packaging" / "windows" / "PAMalytics-icon.png"

    if icon_png.is_file():
        try:
            icon_image = tk.PhotoImage(file=str(icon_png))
            root.iconphoto(True, icon_image)
            root._pamalytics_icon_image = icon_image
        except Exception:
            _log(f"Could not apply window PNG icon from {icon_png}.")

    if icon_ico.is_file():
        try:
            root.iconbitmap(default=str(icon_ico))
        except Exception:
            _log(f"Could not apply window ICO icon from {icon_ico}.")
    root.geometry("380x170")
    root.resizable(False, False)

    status = tk.StringVar(value="Starting PAMalytics…")

    frame = ttk.Frame(root, padding=20)
    frame.pack(fill="both", expand=True)

    ttk.Label(frame, text="PAMalytics", font=("Segoe UI", 16, "bold")).pack(anchor="w")
    ttk.Label(frame, textvariable=status, wraplength=335).pack(anchor="w", pady=(10, 18))

    buttons = ttk.Frame(frame)
    buttons.pack(fill="x")

    open_button = ttk.Button(
        buttons,
        text="Open in browser",
        state="disabled",
        command=lambda: _open_browser(port),
    )
    open_button.pack(side="left")

    def quit_application() -> None:
        _stop_server()
        _release_instance_mutex()
        root.destroy()

    ttk.Button(buttons, text="Quit PAMalytics", command=quit_application).pack(side="right")
    root.protocol("WM_DELETE_WINDOW", quit_application)

    try:
        _start_server(port)
    except Exception as exc:
        _show_error(str(exc))
        quit_application()
        return 1

    def mark_ready() -> None:
        if _wait_until_ready(port):
            status.set("PAMalytics is running in your browser.")
            open_button.configure(state="normal")
            _open_browser(port)
        else:
            status.set(
                "PAMalytics did not start correctly. "
                f"See {_server_log_path()} for details."
            )

    threading.Thread(target=mark_ready, daemon=True).start()
    root.mainloop()
    return 0


def main() -> int:
    if SERVER_MODE_ARGUMENT in sys.argv[1:]:
        return _run_streamlit_server()

    atexit.register(_stop_server)
    atexit.register(_release_instance_mutex)

    try:
        return _run_windows_application()
    except Exception:
        _log(traceback.format_exc())
        raise


if __name__ == "__main__":
    raise SystemExit(main())
