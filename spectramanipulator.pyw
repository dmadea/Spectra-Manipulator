#!python3.8
#
# from PyQt5.QtWidgets import QApplication
# from PyQt5.QtCore import QCoreApplication
from spectramanipulator.__main__ import main
import sys
import os

if sys.executable.endswith("pythonw.exe"):
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.path.join(os.getenv("TEMP"), "stderr-" + os.path.basename(sys.argv[0])), "w")

if __name__ == '__main__':
    main()

