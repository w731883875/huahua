$ErrorActionPreference = "Stop"

python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m PyInstaller --noconfirm --onefile --windowed --name HuahuaAssistant run_app.py

Write-Host "Built: dist\HuahuaAssistant.exe"
