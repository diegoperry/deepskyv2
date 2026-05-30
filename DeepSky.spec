# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


ROOT = Path.cwd()
SOURCE = ROOT / "DeepSky" / "launcher.py"


a = Analysis(
    [str(SOURCE)],
    pathex=[str(ROOT / "DeepSky")],
    binaries=[],
    datas=[],
    hiddenimports=[
        "astropy.io.fits",
        "tifffile",
        "cv2",
        "PIL.Image",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "pandas",
        "scipy",
        "sklearn",
        "IPython",
        "notebook",
        "jupyter",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DeepSky",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="DeepSky",
)
