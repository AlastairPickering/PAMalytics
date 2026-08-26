#ifndef MyAppVersion
  #define MyAppVersion "1.0.4"
#endif

#ifndef MySourceDir
  #define MySourceDir "..\..\dist\PAMalytics"
#endif

#ifndef MyOutputDir
  #define MyOutputDir "..\..\release"
#endif

[Setup]
AppId={{E8FC0544-1D93-4A2B-B95A-41AE2CB3713E}
AppName=PAMalytics
AppVersion={#MyAppVersion}
AppPublisher=Alastair Pickering
DefaultDirName={autopf}\PAMalytics
DefaultGroupName=PAMalytics
DisableProgramGroupPage=yes
OutputDir={#MyOutputDir}
OutputBaseFilename=PAMalytics-v{#MyAppVersion}-windows-x64-setup
SetupIconFile=PAMalytics.ico
UninstallDisplayIcon={app}\PAMalytics.exe
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
CloseApplications=yes
RestartApplications=no
VersionInfoVersion=1.0.4.0
VersionInfoCompany=Alastair Pickering
VersionInfoDescription=PAMalytics Windows installer
VersionInfoProductName=PAMalytics
VersionInfoProductVersion=1.0.4.0

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "{#MySourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\PAMalytics"; Filename: "{app}\PAMalytics.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\PAMalytics"; Filename: "{app}\PAMalytics.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\PAMalytics.exe"; Description: "Launch PAMalytics"; Flags: nowait postinstall skipifsilent
