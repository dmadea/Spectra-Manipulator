# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dialogs\gui_export_spectra_as.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(416, 266)
        self.gridLayout_2 = QtWidgets.QGridLayout(Dialog)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.lblTitle = QtWidgets.QLabel(Dialog)
        self.lblTitle.setWordWrap(True)
        self.lblTitle.setObjectName("lblTitle")
        self.verticalLayout.addWidget(self.lblTitle)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.leDir = QtWidgets.QLineEdit(Dialog)
        self.leDir.setObjectName("leDir")
        self.horizontalLayout.addWidget(self.leDir)
        self.btnDir = QtWidgets.QToolButton(Dialog)
        self.btnDir.setObjectName("btnDir")
        self.horizontalLayout.addWidget(self.btnDir)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.leFileExt = QtWidgets.QLineEdit(Dialog)
        self.leFileExt.setObjectName("leFileExt")
        self.gridLayout.addWidget(self.leFileExt, 0, 2, 1, 1)
        self.leDelimiter = QtWidgets.QLineEdit(Dialog)
        self.leDelimiter.setObjectName("leDelimiter")
        self.gridLayout.addWidget(self.leDelimiter, 1, 2, 1, 1)
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 1, 1, 1)
        self.leDecimalSeparator = QtWidgets.QLineEdit(Dialog)
        self.leDecimalSeparator.setObjectName("leDecimalSeparator")
        self.gridLayout.addWidget(self.leDecimalSeparator, 3, 2, 1, 1)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 3, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 0, 3, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Save)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)
        self.gridLayout_2.addLayout(self.verticalLayout, 0, 0, 1, 1)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        Dialog.setTabOrder(self.btnDir, self.leDir)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.lblTitle.setText(_translate("Dialog", "Each selected top-level item will be saved in separate file with its name as filename. Spectra in groups will be concatenated. Select directory where spectra will be saved and specify the file extension. Non existing directories will be created. Top level items with same name will be overwritten. Delimiter field: use \\t for tabulator."))
        self.btnDir.setText(_translate("Dialog", "..."))
        self.label_2.setText(_translate("Dialog", "Delimiter:"))
        self.label.setText(_translate("Dialog", "File extension:"))
        self.label_3.setText(_translate("Dialog", "Decimal separator:"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
