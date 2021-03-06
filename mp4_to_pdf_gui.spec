# -*- mode: python ; coding: utf-8 -*-

# from https://stackoverflow.com/questions/60057003/copying-license-dependencies-for-pyinstaller
matches = ["LICENSE.txt","METADATA","PKG-INFO"]
lics = []
print("Find 3rd party dependency license files")
for root, dir, files in os.walk("env\Lib\site-packages"):
    for file in files:
            if file in matches:
               src = f"{root}/{file}"
               dest = f"licenses/{os.path.basename(root)}"
               lics.append((src,dest))
               print(f"\tLicense file: {root}/{file}")
print(f"{len(lics)} dependency licenses found. Copying to /license folder in distribution")

block_cipher = None


a = Analysis(['mp4_to_pdf_gui.py'],
             pathex=['D:\\Development\\Python\\PyMp4ToPDF'],
             binaries=[],
             datas=lics,
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='mp4_to_pdf_gui',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='mp4_to_pdf_gui')
