# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'spectramanipulator/dialogs\gui_fit_widget.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(448, 662)
        self.gridLayout_7 = QtWidgets.QGridLayout(Form)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.tabWidget = QtWidgets.QTabWidget(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.gridLayout = QtWidgets.QGridLayout(self.tab)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_2 = QtWidgets.QLabel(self.tab)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_4.addWidget(self.label_2)
        self.cbModel = QtWidgets.QComboBox(self.tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cbModel.sizePolicy().hasHeightForWidth())
        self.cbModel.setSizePolicy(sizePolicy)
        self.cbModel.setObjectName("cbModel")
        self.horizontalLayout_4.addWidget(self.cbModel)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_9 = QtWidgets.QLabel(self.tab)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_3.addWidget(self.label_9)
        self.sbSpeciesCount = QtWidgets.QSpinBox(self.tab)
        self.sbSpeciesCount.setMinimum(1)
        self.sbSpeciesCount.setMaximum(20)
        self.sbSpeciesCount.setObjectName("sbSpeciesCount")
        self.horizontalLayout_3.addWidget(self.sbSpeciesCount)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.gridLayout.addLayout(self.verticalLayout, 1, 0, 1, 1)
        self.scrollAreaPredefinedModel = QtWidgets.QScrollArea(self.tab)
        self.scrollAreaPredefinedModel.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scrollAreaPredefinedModel.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scrollAreaPredefinedModel.setWidgetResizable(True)
        self.scrollAreaPredefinedModel.setObjectName("scrollAreaPredefinedModel")
        self.scrollAreaPredefinedModelContent = QtWidgets.QWidget()
        self.scrollAreaPredefinedModelContent.setGeometry(QtCore.QRect(0, 0, 394, 291))
        self.scrollAreaPredefinedModelContent.setObjectName("scrollAreaPredefinedModelContent")
        self.scrollAreaPredefinedModel.setWidget(self.scrollAreaPredefinedModelContent)
        self.gridLayout.addWidget(self.scrollAreaPredefinedModel, 2, 0, 1, 1)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.tab_2)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_21 = QtWidgets.QLabel(self.tab_2)
        self.label_21.setObjectName("label_21")
        self.horizontalLayout_5.addWidget(self.label_21)
        self.cbGenModels = QtWidgets.QComboBox(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cbGenModels.sizePolicy().hasHeightForWidth())
        self.cbGenModels.setSizePolicy(sizePolicy)
        self.cbGenModels.setObjectName("cbGenModels")
        self.horizontalLayout_5.addWidget(self.cbGenModels)
        self.btnSaveCustomModel = QtWidgets.QPushButton(self.tab_2)
        self.btnSaveCustomModel.setObjectName("btnSaveCustomModel")
        self.horizontalLayout_5.addWidget(self.btnSaveCustomModel)
        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        self.label_22 = QtWidgets.QLabel(self.tab_2)
        self.label_22.setObjectName("label_22")
        self.verticalLayout_2.addWidget(self.label_22)
        self.pteScheme = QtWidgets.QPlainTextEdit(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.pteScheme.sizePolicy().hasHeightForWidth())
        self.pteScheme.setSizePolicy(sizePolicy)
        self.pteScheme.setMinimumSize(QtCore.QSize(0, 30))
        self.pteScheme.setMaximumSize(QtCore.QSize(16777215, 100))
        self.pteScheme.setPlainText("")
        self.pteScheme.setObjectName("pteScheme")
        self.verticalLayout_2.addWidget(self.pteScheme)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.btnBuildModel = QtWidgets.QPushButton(self.tab_2)
        self.btnBuildModel.setObjectName("btnBuildModel")
        self.horizontalLayout_7.addWidget(self.btnBuildModel)
        self.cbShowBackwardRates = QtWidgets.QCheckBox(self.tab_2)
        self.cbShowBackwardRates.setChecked(False)
        self.cbShowBackwardRates.setObjectName("cbShowBackwardRates")
        self.horizontalLayout_7.addWidget(self.cbShowBackwardRates)
        self.verticalLayout_2.addLayout(self.horizontalLayout_7)
        self.cbVarProAmps = QtWidgets.QCheckBox(self.tab_2)
        self.cbVarProAmps.setChecked(True)
        self.cbVarProAmps.setObjectName("cbVarProAmps")
        self.verticalLayout_2.addWidget(self.cbVarProAmps)
        self.cbVarProIntercept = QtWidgets.QCheckBox(self.tab_2)
        self.cbVarProIntercept.setChecked(True)
        self.cbVarProIntercept.setObjectName("cbVarProIntercept")
        self.verticalLayout_2.addWidget(self.cbVarProIntercept)
        self.scrollAreaCustomModel = QtWidgets.QScrollArea(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(4)
        sizePolicy.setHeightForWidth(self.scrollAreaCustomModel.sizePolicy().hasHeightForWidth())
        self.scrollAreaCustomModel.setSizePolicy(sizePolicy)
        self.scrollAreaCustomModel.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scrollAreaCustomModel.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scrollAreaCustomModel.setWidgetResizable(True)
        self.scrollAreaCustomModel.setObjectName("scrollAreaCustomModel")
        self.scrollAreaCustomModelContent = QtWidgets.QWidget()
        self.scrollAreaCustomModelContent.setGeometry(QtCore.QRect(0, 0, 392, 157))
        self.scrollAreaCustomModelContent.setObjectName("scrollAreaCustomModelContent")
        self.scrollAreaCustomModel.setWidget(self.scrollAreaCustomModelContent)
        self.verticalLayout_2.addWidget(self.scrollAreaCustomModel)
        self.gridLayout_3.addLayout(self.verticalLayout_2, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab_2, "")
        self.verticalLayout_3.addWidget(self.tabWidget)
        self.label_10 = QtWidgets.QLabel(Form)
        self.label_10.setObjectName("label_10")
        self.verticalLayout_3.addWidget(self.label_10)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_11 = QtWidgets.QLabel(Form)
        self.label_11.setObjectName("label_11")
        self.horizontalLayout.addWidget(self.label_11)
        self.leX0 = MyLineEdit(Form)
        self.leX0.setObjectName("leX0")
        self.horizontalLayout.addWidget(self.leX0)
        self.label_12 = QtWidgets.QLabel(Form)
        self.label_12.setObjectName("label_12")
        self.horizontalLayout.addWidget(self.label_12)
        self.leX1 = MyLineEdit(Form)
        self.leX1.setObjectName("leX1")
        self.horizontalLayout.addWidget(self.leX1)
        self.btnShowRegion = QtWidgets.QPushButton(Form)
        self.btnShowRegion.setCheckable(True)
        self.btnShowRegion.setChecked(False)
        self.btnShowRegion.setObjectName("btnShowRegion")
        self.horizontalLayout.addWidget(self.btnShowRegion)
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_23 = QtWidgets.QLabel(Form)
        self.label_23.setObjectName("label_23")
        self.gridLayout_4.addWidget(self.label_23, 1, 0, 1, 1)
        self.cbMethod = QtWidgets.QComboBox(Form)
        self.cbMethod.setObjectName("cbMethod")
        self.gridLayout_4.addWidget(self.cbMethod, 0, 1, 1, 1)
        self.label_13 = QtWidgets.QLabel(Form)
        self.label_13.setObjectName("label_13")
        self.gridLayout_4.addWidget(self.label_13, 0, 0, 1, 1)
        self.tbAlgorithmSettings = QtWidgets.QToolButton(Form)
        self.tbAlgorithmSettings.setCheckable(False)
        self.tbAlgorithmSettings.setChecked(False)
        self.tbAlgorithmSettings.setObjectName("tbAlgorithmSettings")
        self.gridLayout_4.addWidget(self.tbAlgorithmSettings, 0, 2, 1, 1)
        self.cbResWeighting = QtWidgets.QComboBox(Form)
        self.cbResWeighting.setObjectName("cbResWeighting")
        self.gridLayout_4.addWidget(self.cbResWeighting, 1, 1, 1, 2)
        self.verticalLayout_3.addLayout(self.gridLayout_4)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.cbShowResiduals = QtWidgets.QCheckBox(Form)
        self.cbShowResiduals.setChecked(False)
        self.cbShowResiduals.setObjectName("cbShowResiduals")
        self.horizontalLayout_8.addWidget(self.cbShowResiduals)
        self.cbFitBlackColor = QtWidgets.QCheckBox(Form)
        self.cbFitBlackColor.setChecked(False)
        self.cbFitBlackColor.setObjectName("cbFitBlackColor")
        self.horizontalLayout_8.addWidget(self.cbFitBlackColor)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem)
        self.verticalLayout_3.addLayout(self.horizontalLayout_8)
        spacerItem1 = QtWidgets.QSpacerItem(20, 28, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Ignored)
        self.verticalLayout_3.addItem(spacerItem1)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.btnPrintReport = QtWidgets.QPushButton(Form)
        self.btnPrintReport.setCheckable(False)
        self.btnPrintReport.setChecked(False)
        self.btnPrintReport.setAutoRepeat(False)
        self.btnPrintReport.setAutoDefault(False)
        self.btnPrintReport.setObjectName("btnPrintReport")
        self.gridLayout_2.addWidget(self.btnPrintReport, 1, 1, 1, 1)
        self.btnCancel = QtWidgets.QPushButton(Form)
        self.btnCancel.setAutoDefault(False)
        self.btnCancel.setObjectName("btnCancel")
        self.gridLayout_2.addWidget(self.btnCancel, 1, 4, 1, 1)
        self.btnSimulateModel = QtWidgets.QPushButton(Form)
        self.btnSimulateModel.setCheckable(False)
        self.btnSimulateModel.setChecked(False)
        self.btnSimulateModel.setAutoRepeat(False)
        self.btnSimulateModel.setAutoDefault(False)
        self.btnSimulateModel.setObjectName("btnSimulateModel")
        self.gridLayout_2.addWidget(self.btnSimulateModel, 0, 0, 1, 1)
        self.btnFit = QtWidgets.QPushButton(Form)
        self.btnFit.setCheckable(False)
        self.btnFit.setChecked(False)
        self.btnFit.setAutoRepeat(False)
        self.btnFit.setAutoDefault(False)
        self.btnFit.setObjectName("btnFit")
        self.gridLayout_2.addWidget(self.btnFit, 1, 0, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem2, 1, 2, 1, 1)
        self.btnOK = QtWidgets.QPushButton(Form)
        self.btnOK.setAutoDefault(False)
        self.btnOK.setObjectName("btnOK")
        self.gridLayout_2.addWidget(self.btnOK, 0, 4, 1, 1)
        self.btnClearPlot = QtWidgets.QPushButton(Form)
        self.btnClearPlot.setCheckable(False)
        self.btnClearPlot.setChecked(False)
        self.btnClearPlot.setAutoRepeat(False)
        self.btnClearPlot.setAutoDefault(False)
        self.btnClearPlot.setObjectName("btnClearPlot")
        self.gridLayout_2.addWidget(self.btnClearPlot, 0, 1, 1, 1)
        self.verticalLayout_3.addLayout(self.gridLayout_2)
        self.gridLayout_7.addLayout(self.verticalLayout_3, 0, 0, 1, 1)

        self.retranslateUi(Form)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label_2.setText(_translate("Form", "Model:"))
        self.label_9.setText(_translate("Form", "Species"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Form", "Predefined Model"))
        self.label_21.setText(_translate("Form", "Saved models:"))
        self.btnSaveCustomModel.setText(_translate("Form", "Save Model As"))
        self.label_22.setText(_translate("Form", "<html><head/><body><p><span style=\" font-weight:600;\">Reaction model:</span></p></body></html>"))
        self.btnBuildModel.setText(_translate("Form", "Build Model"))
        self.cbShowBackwardRates.setText(_translate("Form", "Non-zero backward rates"))
        self.cbVarProAmps.setText(_translate("Form", "Calculate amplitudes from data by OLS"))
        self.cbVarProIntercept.setText(_translate("Form", "Calculate intercept from data by OLS"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Form", "Custom Kinetic Model"))
        self.label_10.setText(_translate("Form", "Global fit range:"))
        self.label_11.setText(_translate("Form", "from"))
        self.label_12.setText(_translate("Form", "to"))
        self.btnShowRegion.setText(_translate("Form", "Show region"))
        self.label_23.setText(_translate("Form", "<html><head/><body><p>Residual weighting</p></body></html>"))
        self.label_13.setText(_translate("Form", "<html><head/><body><p>Minimizing algorithm:</p></body></html>"))
        self.tbAlgorithmSettings.setText(_translate("Form", "..."))
        self.cbShowResiduals.setText(_translate("Form", "Show residuals"))
        self.cbFitBlackColor.setText(_translate("Form", "Fits in black color"))
        self.btnPrintReport.setText(_translate("Form", "Print report"))
        self.btnCancel.setText(_translate("Form", "Cancel"))
        self.btnSimulateModel.setText(_translate("Form", "Simulate model"))
        self.btnFit.setText(_translate("Form", "Fit"))
        self.btnOK.setText(_translate("Form", "Ok"))
        self.btnClearPlot.setText(_translate("Form", "Clear Plot"))
from spectramanipulator.dialogs.mylineedit import MyLineEdit


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
