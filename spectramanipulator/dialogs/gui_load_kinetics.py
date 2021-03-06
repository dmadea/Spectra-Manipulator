# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'spectramanipulator/dialogs\gui_load_kinetics.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(475, 493)
        self.gridLayout_2 = QtWidgets.QGridLayout(Dialog)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_3.addWidget(self.label_4)
        self.btnChooseDirs = QtWidgets.QToolButton(Dialog)
        self.btnChooseDirs.setObjectName("btnChooseDirs")
        self.horizontalLayout_3.addWidget(self.btnChooseDirs)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.lwGridLayout = QtWidgets.QGridLayout()
        self.lwGridLayout.setObjectName("lwGridLayout")
        self.verticalLayout.addLayout(self.lwGridLayout)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 1)
        self.leSpectra = QtWidgets.QLineEdit(Dialog)
        self.leSpectra.setObjectName("leSpectra")
        self.gridLayout.addWidget(self.leSpectra, 1, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 0, 1, 1)
        self.leBlank = QtWidgets.QLineEdit(Dialog)
        self.leBlank.setObjectName("leBlank")
        self.gridLayout.addWidget(self.leBlank, 2, 1, 1, 1)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 7, 0, 1, 2)
        self.leTimes = QtWidgets.QLineEdit(Dialog)
        self.leTimes.setObjectName("leTimes")
        self.gridLayout.addWidget(self.leTimes, 4, 1, 1, 1)
        self.cbCut = QtWidgets.QCheckBox(Dialog)
        self.cbCut.setChecked(False)
        self.cbCut.setObjectName("cbCut")
        self.gridLayout.addWidget(self.cbCut, 6, 0, 1, 1)
        self.cbBCorr = QtWidgets.QCheckBox(Dialog)
        self.cbBCorr.setChecked(False)
        self.cbBCorr.setObjectName("cbBCorr")
        self.gridLayout.addWidget(self.cbBCorr, 5, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 4, 0, 1, 1)
        self.leTimeUnit = QtWidgets.QLineEdit(Dialog)
        self.leTimeUnit.setObjectName("leTimeUnit")
        self.gridLayout.addWidget(self.leTimeUnit, 3, 1, 1, 1)
        self.cbKineticsMeasuredByEach = QtWidgets.QCheckBox(Dialog)
        self.cbKineticsMeasuredByEach.setChecked(True)
        self.cbKineticsMeasuredByEach.setObjectName("cbKineticsMeasuredByEach")
        self.gridLayout.addWidget(self.cbKineticsMeasuredByEach, 3, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.leBCorr0 = QtWidgets.QLineEdit(Dialog)
        self.leBCorr0.setObjectName("leBCorr0")
        self.horizontalLayout.addWidget(self.leBCorr0)
        self.label_6 = QtWidgets.QLabel(Dialog)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout.addWidget(self.label_6)
        self.leBCorr1 = QtWidgets.QLineEdit(Dialog)
        self.leBCorr1.setObjectName("leBCorr1")
        self.horizontalLayout.addWidget(self.leBCorr1)
        self.gridLayout.addLayout(self.horizontalLayout, 5, 1, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.leCut0 = QtWidgets.QLineEdit(Dialog)
        self.leCut0.setObjectName("leCut0")
        self.horizontalLayout_2.addWidget(self.leCut0)
        self.label_7 = QtWidgets.QLabel(Dialog)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_2.addWidget(self.label_7)
        self.leCut1 = QtWidgets.QLineEdit(Dialog)
        self.leCut1.setObjectName("leCut1")
        self.horizontalLayout_2.addWidget(self.leCut1)
        self.gridLayout.addLayout(self.horizontalLayout_2, 6, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.gridLayout_2.addLayout(self.verticalLayout, 0, 0, 1, 1)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Loads UV-Vis kinetics. The spectra can be in various formats (dx, csv, txt, etc.). If blank spectrum is provided, it will be subtracted from each of the spectra. Time corresponding to each spectrum will be set as the name for that spectrum. Times can be provided by time difference (in case kinetics was measured reguraly) or from filename. Postprocessing (baseline correction and/or cut of spectra) can be performed on loaded dataset.</p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Required folder structure for one kinetics:</p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">[kinetics folder]<span style=\" font-family:\'JetBrains Mono,monospace\'; color:#629755;\"><br />    </span><span style=\" font-family:\'JetBrains Mono,monospace\'; color:#000000;\">[spectra folder]<br />         01.dx<br />         02.dx<br />         ...<br />    times.txt (optional)<br />    blank.dx (optional)</span></p></body></html>"))
        self.label_4.setText(_translate("Dialog", "Folders to load:"))
        self.btnChooseDirs.setText(_translate("Dialog", "..."))
        self.label_2.setText(_translate("Dialog", "Blank spectrum name (optional):"))
        self.leSpectra.setText(_translate("Dialog", "spectra"))
        self.label_3.setText(_translate("Dialog", "Spectra folder name:"))
        self.leBlank.setText(_translate("Dialog", "blank.dx"))
        self.leTimes.setText(_translate("Dialog", "times.txt"))
        self.cbCut.setText(_translate("Dialog", "Cut spectra to range:"))
        self.cbBCorr.setText(_translate("Dialog", "Apply baseline correction in range:"))
        self.label_5.setText(_translate("Dialog", "Use times from filename (optional):"))
        self.leTimeUnit.setText(_translate("Dialog", "1"))
        self.cbKineticsMeasuredByEach.setText(_translate("Dialog", "Kinetics measured by each (time unit):"))
        self.leBCorr0.setText(_translate("Dialog", "700"))
        self.label_6.setText(_translate("Dialog", "to"))
        self.leBCorr1.setText(_translate("Dialog", "800"))
        self.leCut0.setText(_translate("Dialog", "230"))
        self.label_7.setText(_translate("Dialog", "to"))
        self.leCut1.setText(_translate("Dialog", "650"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
