# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['D:\\small_tools_python\\GSViewer\\main.py'],
    pathex=[],
    binaries=[('D:\\Anaconda3\\envs\\pytorch\\Lib\\site-packages\\glfw\\glfw3.dll', '.')],
    datas=[('D:\\small_tools_python\\GSViewer', 'GSViewer/'), ('D:\\small_tools_python\\GSViewer\\gui', 'gui/'), ('D:\\small_tools_python\\GSViewer\\render', 'render/'), ('D:\\small_tools_python\\GSViewer\\shaders', 'shaders/'), ('D:\\small_tools_python\\GSViewer\\tools', 'tools/'), ('D:\\small_tools_python\\GSViewer\\tools\\gsconverter', 'gsconverter/'), ('D:\\small_tools_python\\GSViewer\\tools\\gsconverter\\utils', 'utils/')],
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
    name='GSViewer',
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
    icon=['D:\\small_tools_python\\GSViewer\\EXElogo.ico'],
)
