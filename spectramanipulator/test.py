
import sys
from PyQt5 import QtWidgets
from config_sel.configtreemodel import ConfigTreeModel
from config_sel.configtreeview import ConfigTreeView

app = QtWidgets.QApplication(sys.argv)

tm = ConfigTreeModel()
tv = ConfigTreeView(tm)
tv.show()

sys.exit(app.exec())

