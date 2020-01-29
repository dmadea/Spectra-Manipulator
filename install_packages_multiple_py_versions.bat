
REM Change this path according to your Python 3 distribution Stripts folder location
cd "C:\Users\[--user--]\AppData\Local\Programs\Python\Python37\Scripts"

..\python.exe -m pip install --upgrade pip
pip.exe install --upgrade setuptools
pip.exe install lmfit
pip.exe install PyQt5
pip.exe install numba
pip.exe install matplotlib
pip.exe install qtconsole

@echo off
cls
CALL
pause