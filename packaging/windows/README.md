# Windows support

The current Windows no-code route is through the root-level launchers:

```text
pamalytics_windows_launcher_uv.bat
pamalytics_windows_launcher.bat
```

They intentionally remain in the repository root so users who download the GitHub zip can start PAMalytics without navigating through developer folders.

`VC_redist.x64.exe` is retained at the repository root because some Windows scientific/audio dependencies require the Microsoft Visual C++ runtime. The launcher installs it passively when present.

A fully packaged Windows installer is planned later. Until that is tested, do not advertise a standalone Windows packaged release.
