
import requests
from distutils.version import StrictVersion
from distutils.version import LooseVersion
import sys
import os
import subprocess
from . import __version__
from . import windows
from .qt_task import Task

from PyQt5.QtWidgets import QMessageBox


# from https://gist.github.com/trinitronx/026574839d96a0c0efe1e9f2fd300f03
def get_versions(PKG_NAME='spectramanipulator'):
    """Get all versions of a package from PyPI"""
    r = requests.get(f'https://pypi.org/pypi/{PKG_NAME}/json')
    data = r.json()
    versions = data['releases'].keys()
    # remove dev versions
    versions = list(filter(lambda v: 'rc' not in v and 'dev' not in v and 'post' not in v, versions))
    try:
        versions.sort(key=StrictVersion)
    except ValueError as e:
        if 'invalid version number' in str(e):
            versions.sort(key=LooseVersion)
        else:
            raise e

    return versions


def get_latest_version():
    return get_versions()[-1]

# https://stackoverflow.com/questions/11887762/how-do-i-compare-version-numbers-in-python
def is_latest_version(ver):
    return LooseVersion(__version__) >= LooseVersion(ver)
    # return LooseVersion("0.1.0") >= LooseVersion(ver)


# # https://stackoverflow.com/questions/636561/how-can-i-run-an-external-command-asynchronously-from-python
class TaskUpdate(Task):
    def __init__(self, parent=None):
        super(TaskUpdate, self).__init__(parent)
        self.parent = parent
        self.latest_version = __version__

    def can_update_program(self):
        try:
            self.latest_version = get_latest_version()
        except requests.exceptions.RequestException as e:
            QMessageBox.critical(self.parent, 'Update', f"Connection error. {str(e)}")
            return False

        if is_latest_version(self.latest_version):
            QMessageBox.information(self.parent, 'Update', "The program is already up to date.")
            return False

        reply = QMessageBox.question(self.parent, 'Update',
                                     f"New version {self.latest_version} was found. Current version is {__version__}. "
                                     + "Do you want to update the program?", QMessageBox.Yes |
                                     QMessageBox.No)

        if reply == QMessageBox.No:
            return False

        if sys.platform == 'win32':
            windows.set_attached_console_visible(True)

        return True

    def run(self):
        python_dir = os.path.dirname(sys.executable)
        pip_path = os.path.join(python_dir, "Scripts", "pip.exe")

        subprocess.run(f"{pip_path} install spectramanipulator=={self.latest_version}")

    def postRun(self):
        if sys.platform == 'win32':
            windows.set_attached_console_visible(False)
        QMessageBox.information(self.parent, 'Update',
                                "The program was updated. Please restart your application (all instances have to be closed).")

