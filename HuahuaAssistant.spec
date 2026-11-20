# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['run_app.py'],
    pathex=[],
    binaries=[],
    datas=[('huahua_assistant/assets', 'huahua_assistant/assets')],
    # windows_capture 是在 _WgcSession 里按需导入的，静态分析看不到，必须显式声明，
    # 否则打包后的 exe 会退回到「截屏幕」那条路（工具窗口一盖就截到自己）。
    hiddenimports=['huahua_assistant.multi_user', 'windows_capture'],
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
    name='HuahuaAssistant',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
