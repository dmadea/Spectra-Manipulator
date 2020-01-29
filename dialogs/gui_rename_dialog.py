# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dialogs\gui_rename_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(403, 333)
        self.widget = QtWidgets.QWidget(Dialog)
        self.widget.setGeometry(QtCore.QRect(20, 20, 361, 291))
        self.widget.setObjectName("widget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 0, 1, 1)
        self.sbOffset = QtWidgets.QSpinBox(self.widget)
        self.sbOffset.setObjectName("sbOffset")
        self.gridLayout.addWidget(self.sbOffset, 1, 2, 1, 1)
        self.leExpression = QtWidgets.QLineEdit(self.widget)
        self.leExpression.setObjectName("leExpression")
        self.gridLayout.addWidget(self.leExpression, 0, 2, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.cbTakeNamesFromList = QtWidgets.QCheckBox(self.widget)
        self.cbTakeNamesFromList.setObjectName("cbTakeNamesFromList")
        self.verticalLayout_2.addWidget(self.cbTakeNamesFromList)
        self.leList = QtWidgets.QLineEdit(self.widget)
        self.leList.setObjectName("leList")
        self.verticalLayout_2.addWidget(self.leList)
        self.buttonBox = QtWidgets.QDialogButtonBox(self.widget)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout_2.addWidget(self.buttonBox)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        Dialog.setTabOrder(self.leExpression, self.sbOffset)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "<html><head/><body><p>Smartly rename selected items. {:02d} - integer counter, for more see https://pyformat.info/. For original name slicing, use {start_idx:end_idx}, same from python slicing rules. Eg. expression is \'{:02d}: t = {:} us\', current name is \'167\' and current counter is 16. Resulting name will be \'16: t = 167 us\'.</p></body></html>"))
        self.label_2.setText(_translate("Dialog", "expression"))
        self.label_3.setText(_translate("Dialog", "counter offset"))
        self.cbTakeNamesFromList.setText(_translate("Dialog", "Take names from list (separate values by comma):"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
