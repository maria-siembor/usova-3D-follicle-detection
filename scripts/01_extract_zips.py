import zipfile
import os

os.makedirs('data', exist_ok=True)

zips = ['Training_Set_2019.zip', 'Test_Set_2019.zip', 'USOVA3D_tools.zip']
for z in zips:
    path = os.path.join('data', z)
    if not os.path.exists(path):
        print(f"MISSING: {path}, place the zip file there first")
        continue
    with zipfile.ZipFile(path) as zf:
        zf.extractall('data')
    print(f"Extracted {z}")
