
python -m pip install --upgrade pip --user
pip install --upgrade setuptools --user
pip install git+https://github.com/pyqtgraph/pyqtgraph@master --user
pip install lmfit --user
pip install PyQt5 --user

rm pip install numba --user

pip install matplotlib --user
pip install qtconsole --user

rm Associate project files
python associate_project_file.py

@echo off
pause