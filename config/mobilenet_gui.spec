a = Analysis(
    ["../src/mobilenet_gui.py"],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="MobileNet Object Detector",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

app = BUNDLE(
    exe,
    name="MobileNet Object Detector.app",
    icon=None,
    bundle_identifier="com.greenhouseinfo.mobilenet-object-detector",
    info_plist={
        "CFBundleDisplayName": "MobileNet Object Detector",
        "CFBundleName": "MobileNet Object Detector",
        "NSCameraUsageDescription": "This app needs camera access to capture frames for realtime MobileNet object detection.",
        "NSHighResolutionCapable": True,
    },
)
