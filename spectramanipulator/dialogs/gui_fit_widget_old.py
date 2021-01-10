# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'spectramanipulator/dialogs\gui_fit_widget_old.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(444, 591)
        self.gridLayout_4 = QtWidgets.QGridLayout(Form)
        self.gridLayout_4.setObjectName("gridLayout_4")
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
        self.cbCustom = QtWidgets.QCheckBox(self.tab)
        self.cbCustom.setObjectName("cbCustom")
        self.verticalLayout.addWidget(self.cbCustom)
        self.label = QtWidgets.QLabel(self.tab)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.pteEquation = QtWidgets.QPlainTextEdit(self.tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(4)
        sizePolicy.setHeightForWidth(self.pteEquation.sizePolicy().hasHeightForWidth())
        self.pteEquation.setSizePolicy(sizePolicy)
        self.pteEquation.setMinimumSize(QtCore.QSize(0, 0))
        self.pteEquation.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.pteEquation.setPlainText("")
        self.pteEquation.setObjectName("pteEquation")
        self.verticalLayout.addWidget(self.pteEquation)
        self.cbUpdateInitValues = QtWidgets.QCheckBox(self.tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cbUpdateInitValues.sizePolicy().hasHeightForWidth())
        self.cbUpdateInitValues.setSizePolicy(sizePolicy)
        self.cbUpdateInitValues.setObjectName("cbUpdateInitValues")
        self.verticalLayout.addWidget(self.cbUpdateInitValues)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_9 = QtWidgets.QLabel(self.tab)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_3.addWidget(self.label_9)
        self.sbParamsCount = QtWidgets.QSpinBox(self.tab)
        self.sbParamsCount.setMinimum(1)
        self.sbParamsCount.setMaximum(10)
        self.sbParamsCount.setObjectName("sbParamsCount")
        self.horizontalLayout_3.addWidget(self.sbParamsCount)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.scrollArea_2 = QtWidgets.QScrollArea(self.tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(10)
        sizePolicy.setHeightForWidth(self.scrollArea_2.sizePolicy().hasHeightForWidth())
        self.scrollArea_2.setSizePolicy(sizePolicy)
        self.scrollArea_2.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollArea_2.setObjectName("scrollArea_2")
        self.scrollAreaWidgetContents_3 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_3.setGeometry(QtCore.QRect(0, 0, 388, 86))
        self.scrollAreaWidgetContents_3.setObjectName("scrollAreaWidgetContents_3")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents_3)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.predefGridLayout = QtWidgets.QGridLayout()
        self.predefGridLayout.setObjectName("predefGridLayout")
        self.label_7 = QtWidgets.QLabel(self.scrollAreaWidgetContents_3)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.predefGridLayout.addWidget(self.label_7, 0, 4, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.scrollAreaWidgetContents_3)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.predefGridLayout.addWidget(self.label_3, 0, 0, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.scrollAreaWidgetContents_3)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.predefGridLayout.addWidget(self.label_8, 0, 5, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.scrollAreaWidgetContents_3)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.predefGridLayout.addWidget(self.label_6, 0, 3, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.scrollAreaWidgetContents_3)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.predefGridLayout.addWidget(self.label_4, 0, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.scrollAreaWidgetContents_3)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.predefGridLayout.addWidget(self.label_5, 0, 2, 1, 1)
        self.gridLayout_6.addLayout(self.predefGridLayout, 0, 0, 1, 1)
        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_3)
        self.verticalLayout.addWidget(self.scrollArea_2)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)
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
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(4)
        sizePolicy.setHeightForWidth(self.pteScheme.sizePolicy().hasHeightForWidth())
        self.pteScheme.setSizePolicy(sizePolicy)
        self.pteScheme.setMinimumSize(QtCore.QSize(0, 0))
        self.pteScheme.setMaximumSize(QtCore.QSize(16777215, 300))
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
        self.cbPlotAllComps = QtWidgets.QCheckBox(self.tab_2)
        self.cbPlotAllComps.setObjectName("cbPlotAllComps")
        self.verticalLayout_2.addWidget(self.cbPlotAllComps)
        self.scrollArea = QtWidgets.QScrollArea(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(10)
        sizePolicy.setHeightForWidth(self.scrollArea.sizePolicy().hasHeightForWidth())
        self.scrollArea.setSizePolicy(sizePolicy)
        self.scrollArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scrollArea.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents_2 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, 0, 388, 134))
        self.scrollAreaWidgetContents_2.setObjectName("scrollAreaWidgetContents_2")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents_2)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.cusModelGridLayout = QtWidgets.QGridLayout()
        self.cusModelGridLayout.setObjectName("cusModelGridLayout")
        self.label_17 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_17.setAlignment(QtCore.Qt.AlignCenter)
        self.label_17.setObjectName("label_17")
        self.cusModelGridLayout.addWidget(self.label_17, 0, 5, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_15.setAlignment(QtCore.Qt.AlignCenter)
        self.label_15.setObjectName("label_15")
        self.cusModelGridLayout.addWidget(self.label_15, 0, 4, 1, 1)
        self.label_20 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_20.setAlignment(QtCore.Qt.AlignCenter)
        self.label_20.setObjectName("label_20")
        self.cusModelGridLayout.addWidget(self.label_20, 0, 2, 1, 1)
        self.label_18 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_18.setAlignment(QtCore.Qt.AlignCenter)
        self.label_18.setObjectName("label_18")
        self.cusModelGridLayout.addWidget(self.label_18, 0, 3, 1, 1)
        self.label_16 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_16.setAlignment(QtCore.Qt.AlignCenter)
        self.label_16.setObjectName("label_16")
        self.cusModelGridLayout.addWidget(self.label_16, 0, 0, 1, 1)
        self.label_19 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_19.setAlignment(QtCore.Qt.AlignCenter)
        self.label_19.setObjectName("label_19")
        self.cusModelGridLayout.addWidget(self.label_19, 0, 1, 1, 1)
        self.gridLayout_5.addLayout(self.cusModelGridLayout, 1, 0, 1, 1)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents_2)
        self.verticalLayout_2.addWidget(self.scrollArea)
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
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_13 = QtWidgets.QLabel(Form)
        self.label_13.setObjectName("label_13")
        self.horizontalLayout_6.addWidget(self.label_13)
        self.cbMethod = QtWidgets.QComboBox(Form)
        self.cbMethod.setObjectName("cbMethod")
        self.horizontalLayout_6.addWidget(self.cbMethod)
        self.verticalLayout_3.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.cbPropWeighting = QtWidgets.QCheckBox(Form)
        self.cbPropWeighting.setChecked(False)
        self.cbPropWeighting.setObjectName("cbPropWeighting")
        self.horizontalLayout_2.addWidget(self.cbPropWeighting)
        self.label_14 = QtWidgets.QLabel(Form)
        self.label_14.setObjectName("label_14")
        self.horizontalLayout_2.addWidget(self.label_14)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Ignored)
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
        self.btnPlotFunction = QtWidgets.QPushButton(Form)
        self.btnPlotFunction.setCheckable(False)
        self.btnPlotFunction.setChecked(False)
        self.btnPlotFunction.setAutoRepeat(False)
        self.btnPlotFunction.setAutoDefault(False)
        self.btnPlotFunction.setObjectName("btnPlotFunction")
        self.gridLayout_2.addWidget(self.btnPlotFunction, 0, 0, 1, 1)
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
        self.gridLayout_4.addLayout(self.verticalLayout_3, 0, 0, 1, 1)

        self.retranslateUi(Form)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label_2.setText(_translate("Form", "Model:"))
        self.cbCustom.setText(_translate("Form", "Custom model (equation window will be editable)"))
        self.label.setText(_translate("Form", "<html><head/><body><p><span style=\" font-weight:600;\">def func(x, *params):</span></p></body></html>"))
        self.cbUpdateInitValues.setText(_translate("Form", "Update initial values when linear region move is finished"))
        self.label_9.setText(_translate("Form", "Params:"))
        self.label_7.setText(_translate("Form", "Fixed"))
        self.label_3.setText(_translate("Form", "Param"))
        self.label_8.setText(_translate("Form", "Error"))
        self.label_6.setText(_translate("Form", "Upper\n"
"bound"))
        self.label_4.setText(_translate("Form", "Lower\n"
"bound"))
        self.label_5.setText(_translate("Form", "<= Value <="))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Form", "Predefined Model"))
        self.label_21.setText(_translate("Form", "Saved models:"))
        self.btnSaveCustomModel.setText(_translate("Form", "Save Model As"))
        self.label_22.setText(_translate("Form", "<html><head/><body><p><span style=\" font-weight:600;\">Reaction model:</span></p></body></html>"))
        self.btnBuildModel.setText(_translate("Form", "Build Model"))
        self.cbShowBackwardRates.setText(_translate("Form", "Non-zero backward rates"))
        self.cbPlotAllComps.setText(_translate("Form", "Plot all compartments"))
        self.label_17.setText(_translate("Form", "Error"))
        self.label_15.setText(_translate("Form", "Fixed"))
        self.label_20.setText(_translate("Form", "<= Value <="))
        self.label_18.setText(_translate("Form", "Upper\n"
"bound"))
        self.label_16.setText(_translate("Form", "Param"))
        self.label_19.setText(_translate("Form", "Lower\n"
"bound"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Form", "Custom Kinetic Model"))
        self.label_10.setText(_translate("Form", "Selected x range:"))
        self.label_11.setText(_translate("Form", "from"))
        self.label_12.setText(_translate("Form", "to"))
        self.label_13.setText(_translate("Form", "<html><head/><body><p>Minimizing algorithm:</p></body></html>"))
        self.cbPropWeighting.setText(_translate("Form", "Proportional weighting"))
        self.label_14.setText(_translate("Form", "e<sub>i</sub>*=1/y<sub>i</sub><sup>2</sup>"))
        self.btnPrintReport.setText(_translate("Form", "Print report"))
        self.btnCancel.setText(_translate("Form", "Cancel"))
        self.btnPlotFunction.setText(_translate("Form", "Plot Function"))
        self.btnFit.setText(_translate("Form", "Fit"))
        self.btnOK.setText(_translate("Form", "Ok"))
        self.btnClearPlot.setText(_translate("Form", "Clear Plot"))
from dialogs.mylineedit import MyLineEdit


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
