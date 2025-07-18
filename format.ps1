# Windows PowerShell version (format.ps1)
# Run .\format.ps1 in bash
.\.fraudvenv\Scripts\black . 
.\.fraudvenv\Scripts\isort . 
.\.fraudvenv\Scripts\python.exe -m flake8 . --exclude=.fraudvenv
if (Get-ChildItem -Recurse -Filter *.ipynb) {
  .\.fraudvenv\Scripts\python.exe -m nbqa black . --line-length=88 --exclude .fraudvenv
  .\.fraudvenv\Scripts\python.exe -m nbqa isort . --skip .fraudvenv
  .\.fraudvenv\Scripts\python.exe -m nbqa flake8 . --exclude .fraudvenv --exit-zero
} else {
  Write-Host "No notebooks found - skipping nbQA."
}
