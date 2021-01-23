# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'spectramanipulator/dialogs\gui_settings.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.setEnabled(True)
        Dialog.resize(697, 800)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Dialog.sizePolicy().hasHeightForWidth())
        Dialog.setSizePolicy(sizePolicy)
        Dialog.setMinimumSize(QtCore.QSize(0, 0))
        Dialog.setMaximumSize(QtCore.QSize(999999, 999999))
        self.gridLayout_5 = QtWidgets.QGridLayout(Dialog)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.tabWidget = QtWidgets.QTabWidget(Dialog)
        self.tabWidget.setTabPosition(QtWidgets.QTabWidget.North)
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabWidget.setElideMode(QtCore.Qt.ElideNone)
        self.tabWidget.setUsesScrollButtons(True)
        self.tabWidget.setDocumentMode(False)
        self.tabWidget.setTabsClosable(False)
        self.tabWidget.setMovable(False)
        self.tabWidget.setTabBarAutoHide(True)
        self.tabWidget.setObjectName("tabWidget")
        self.tImport = QtWidgets.QWidget()
        self.tImport.setObjectName("tImport")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.tImport)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_11 = QtWidgets.QLabel(self.tImport)
        self.label_11.setWordWrap(True)
        self.label_11.setObjectName("label_11")
        self.verticalLayout_2.addWidget(self.label_11)
        self.groupBox_7 = QtWidgets.QGroupBox(self.tImport)
        self.groupBox_7.setObjectName("groupBox_7")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.groupBox_7)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.cbRemoveEmptyEntries = QtWidgets.QCheckBox(self.groupBox_7)
        self.cbRemoveEmptyEntries.setObjectName("cbRemoveEmptyEntries")
        self.verticalLayout_10.addWidget(self.cbRemoveEmptyEntries)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_12 = QtWidgets.QLabel(self.groupBox_7)
        self.label_12.setObjectName("label_12")
        self.horizontalLayout_3.addWidget(self.label_12)
        self.sbColumns = QtWidgets.QSpinBox(self.groupBox_7)
        self.sbColumns.setMaximum(9999999)
        self.sbColumns.setObjectName("sbColumns")
        self.horizontalLayout_3.addWidget(self.sbColumns)
        self.verticalLayout_10.addLayout(self.horizontalLayout_3)
        self.cbSkipNaNColumns = QtWidgets.QCheckBox(self.groupBox_7)
        self.cbSkipNaNColumns.setObjectName("cbSkipNaNColumns")
        self.verticalLayout_10.addWidget(self.cbSkipNaNColumns)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_13 = QtWidgets.QLabel(self.groupBox_7)
        self.label_13.setObjectName("label_13")
        self.horizontalLayout_5.addWidget(self.label_13)
        self.sbNaNReplacement = QtWidgets.QSpinBox(self.groupBox_7)
        self.sbNaNReplacement.setMinimum(-999999)
        self.sbNaNReplacement.setMaximum(9999997)
        self.sbNaNReplacement.setObjectName("sbNaNReplacement")
        self.horizontalLayout_5.addWidget(self.sbNaNReplacement)
        self.verticalLayout_10.addLayout(self.horizontalLayout_5)
        self.gridLayout_7.addLayout(self.verticalLayout_10, 0, 0, 1, 1)
        self.verticalLayout_2.addWidget(self.groupBox_7)
        self.groupBox = QtWidgets.QGroupBox(self.tImport)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_16 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_16.setObjectName("gridLayout_16")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.general_imp_delimiter = QtWidgets.QLineEdit(self.groupBox)
        self.general_imp_delimiter.setObjectName("general_imp_delimiter")
        self.gridLayout_3.addWidget(self.general_imp_delimiter, 5, 3, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setObjectName("label_3")
        self.gridLayout_3.addWidget(self.label_3, 5, 0, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.groupBox)
        self.label_7.setObjectName("label_7")
        self.gridLayout_3.addWidget(self.label_7, 0, 3, 1, 1)
        self.general_imp_decimal_sep = QtWidgets.QLineEdit(self.groupBox)
        self.general_imp_decimal_sep.setObjectName("general_imp_decimal_sep")
        self.gridLayout_3.addWidget(self.general_imp_decimal_sep, 6, 3, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.groupBox)
        self.label_4.setObjectName("label_4")
        self.gridLayout_3.addWidget(self.label_4, 6, 0, 1, 1)
        self.dx_imp_delimiter = QtWidgets.QLineEdit(self.groupBox)
        self.dx_imp_delimiter.setObjectName("dx_imp_delimiter")
        self.gridLayout_3.addWidget(self.dx_imp_delimiter, 5, 1, 1, 1)
        self.csv_imp_delimiter = QtWidgets.QLineEdit(self.groupBox)
        self.csv_imp_delimiter.setObjectName("csv_imp_delimiter")
        self.gridLayout_3.addWidget(self.csv_imp_delimiter, 5, 2, 1, 1)
        self.dx_imp_decimal_sep = QtWidgets.QLineEdit(self.groupBox)
        self.dx_imp_decimal_sep.setObjectName("dx_imp_decimal_sep")
        self.gridLayout_3.addWidget(self.dx_imp_decimal_sep, 6, 1, 1, 1)
        self.csv_imp_decimal_sep = QtWidgets.QLineEdit(self.groupBox)
        self.csv_imp_decimal_sep.setObjectName("csv_imp_decimal_sep")
        self.gridLayout_3.addWidget(self.csv_imp_decimal_sep, 6, 2, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.groupBox)
        self.label_6.setObjectName("label_6")
        self.gridLayout_3.addWidget(self.label_6, 0, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.groupBox)
        self.label_5.setObjectName("label_5")
        self.gridLayout_3.addWidget(self.label_5, 0, 2, 1, 1)
        self.verticalLayout_9.addLayout(self.gridLayout_3)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.groupBox_5 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_5.setObjectName("groupBox_5")
        self.gridLayout_9 = QtWidgets.QGridLayout(self.groupBox_5)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.dx_import_spectra_name_from_filename = QtWidgets.QRadioButton(self.groupBox_5)
        self.dx_import_spectra_name_from_filename.setObjectName("dx_import_spectra_name_from_filename")
        self.verticalLayout_4.addWidget(self.dx_import_spectra_name_from_filename)
        self.rb_DX_import_from_title = QtWidgets.QRadioButton(self.groupBox_5)
        self.rb_DX_import_from_title.setChecked(True)
        self.rb_DX_import_from_title.setObjectName("rb_DX_import_from_title")
        self.verticalLayout_4.addWidget(self.rb_DX_import_from_title)
        self.dx_if_title_is_empty_use_filename = QtWidgets.QCheckBox(self.groupBox_5)
        self.dx_if_title_is_empty_use_filename.setObjectName("dx_if_title_is_empty_use_filename")
        self.verticalLayout_4.addWidget(self.dx_if_title_is_empty_use_filename)
        self.gridLayout_9.addLayout(self.verticalLayout_4, 0, 0, 1, 1)
        self.horizontalLayout_2.addWidget(self.groupBox_5)
        self.groupBox_6 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_6.setObjectName("groupBox_6")
        self.gridLayout_15 = QtWidgets.QGridLayout(self.groupBox_6)
        self.gridLayout_15.setObjectName("gridLayout_15")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.label = QtWidgets.QLabel(self.groupBox_6)
        self.label.setObjectName("label")
        self.verticalLayout_8.addWidget(self.label)
        self.general_import_spectra_name_from_filename = QtWidgets.QRadioButton(self.groupBox_6)
        self.general_import_spectra_name_from_filename.setObjectName("general_import_spectra_name_from_filename")
        self.verticalLayout_8.addWidget(self.general_import_spectra_name_from_filename)
        self.rb_general_import_from_header = QtWidgets.QRadioButton(self.groupBox_6)
        self.rb_general_import_from_header.setChecked(True)
        self.rb_general_import_from_header.setObjectName("rb_general_import_from_header")
        self.verticalLayout_8.addWidget(self.rb_general_import_from_header)
        self.general_if_header_is_empty_use_filename = QtWidgets.QCheckBox(self.groupBox_6)
        self.general_if_header_is_empty_use_filename.setObjectName("general_if_header_is_empty_use_filename")
        self.verticalLayout_8.addWidget(self.general_if_header_is_empty_use_filename)
        self.gridLayout_15.addLayout(self.verticalLayout_8, 0, 0, 1, 1)
        self.horizontalLayout_2.addWidget(self.groupBox_6)
        self.verticalLayout_9.addLayout(self.horizontalLayout_2)
        self.gridLayout_16.addLayout(self.verticalLayout_9, 1, 0, 1, 1)
        self.verticalLayout_2.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(self.tImport)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.excel_imp_as_text = QtWidgets.QCheckBox(self.groupBox_2)
        self.excel_imp_as_text.setObjectName("excel_imp_as_text")
        self.verticalLayout_3.addWidget(self.excel_imp_as_text)
        self.label_10 = QtWidgets.QLabel(self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_10.sizePolicy().hasHeightForWidth())
        self.label_10.setSizePolicy(sizePolicy)
        self.label_10.setWordWrap(True)
        self.label_10.setObjectName("label_10")
        self.verticalLayout_3.addWidget(self.label_10)
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_8 = QtWidgets.QLabel(self.groupBox_2)
        self.label_8.setObjectName("label_8")
        self.gridLayout_4.addWidget(self.label_8, 0, 0, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.groupBox_2)
        self.label_9.setObjectName("label_9")
        self.gridLayout_4.addWidget(self.label_9, 1, 0, 1, 1)
        self.clip_imp_decimal_sep = QtWidgets.QLineEdit(self.groupBox_2)
        self.clip_imp_decimal_sep.setObjectName("clip_imp_decimal_sep")
        self.gridLayout_4.addWidget(self.clip_imp_decimal_sep, 1, 1, 1, 1)
        self.clip_imp_delimiter = QtWidgets.QLineEdit(self.groupBox_2)
        self.clip_imp_delimiter.setObjectName("clip_imp_delimiter")
        self.gridLayout_4.addWidget(self.clip_imp_delimiter, 0, 1, 1, 1)
        self.verticalLayout_3.addLayout(self.gridLayout_4)
        self.gridLayout_10.addLayout(self.verticalLayout_3, 0, 0, 1, 1)
        self.verticalLayout_2.addWidget(self.groupBox_2)
        self.gridLayout_2.addLayout(self.verticalLayout_2, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tImport, "")
        self.tExport = QtWidgets.QWidget()
        self.tExport.setObjectName("tExport")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.tExport)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.groupBox_4 = QtWidgets.QGroupBox(self.tExport)
        self.groupBox_4.setObjectName("groupBox_4")
        self.gridLayout_12 = QtWidgets.QGridLayout(self.groupBox_4)
        self.gridLayout_12.setObjectName("gridLayout_12")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.label_18 = QtWidgets.QLabel(self.groupBox_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_18.sizePolicy().hasHeightForWidth())
        self.label_18.setSizePolicy(sizePolicy)
        self.label_18.setWordWrap(True)
        self.label_18.setObjectName("label_18")
        self.verticalLayout_5.addWidget(self.label_18)
        self.clip_exp_include_group_name = QtWidgets.QCheckBox(self.groupBox_4)
        self.clip_exp_include_group_name.setObjectName("clip_exp_include_group_name")
        self.verticalLayout_5.addWidget(self.clip_exp_include_group_name)
        self.clip_exp_include_header = QtWidgets.QCheckBox(self.groupBox_4)
        self.clip_exp_include_header.setObjectName("clip_exp_include_header")
        self.verticalLayout_5.addWidget(self.clip_exp_include_header)
        self.gridLayout_6 = QtWidgets.QGridLayout()
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.label_19 = QtWidgets.QLabel(self.groupBox_4)
        self.label_19.setObjectName("label_19")
        self.gridLayout_6.addWidget(self.label_19, 0, 0, 1, 1)
        self.label_20 = QtWidgets.QLabel(self.groupBox_4)
        self.label_20.setObjectName("label_20")
        self.gridLayout_6.addWidget(self.label_20, 1, 0, 1, 1)
        self.clip_exp_decimal_sep = QtWidgets.QLineEdit(self.groupBox_4)
        self.clip_exp_decimal_sep.setObjectName("clip_exp_decimal_sep")
        self.gridLayout_6.addWidget(self.clip_exp_decimal_sep, 1, 1, 1, 1)
        self.clip_exp_delimiter = QtWidgets.QLineEdit(self.groupBox_4)
        self.clip_exp_delimiter.setObjectName("clip_exp_delimiter")
        self.gridLayout_6.addWidget(self.clip_exp_delimiter, 0, 1, 1, 1)
        self.verticalLayout_5.addLayout(self.gridLayout_6)
        self.gridLayout_12.addLayout(self.verticalLayout_5, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.groupBox_4, 1, 0, 1, 1)
        self.groupBox_3 = QtWidgets.QGroupBox(self.tExport)
        self.groupBox_3.setObjectName("groupBox_3")
        self.gridLayout_11 = QtWidgets.QGridLayout(self.groupBox_3)
        self.gridLayout_11.setObjectName("gridLayout_11")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.files_exp_include_group_name = QtWidgets.QCheckBox(self.groupBox_3)
        self.files_exp_include_group_name.setObjectName("files_exp_include_group_name")
        self.verticalLayout_6.addWidget(self.files_exp_include_group_name)
        self.files_exp_include_header = QtWidgets.QCheckBox(self.groupBox_3)
        self.files_exp_include_header.setObjectName("files_exp_include_header")
        self.verticalLayout_6.addWidget(self.files_exp_include_header)
        self.gridLayout_11.addLayout(self.verticalLayout_6, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.groupBox_3, 0, 0, 1, 1)
        self.gridLayout_8.addLayout(self.gridLayout, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tExport, "")
        self.tPlotting = QtWidgets.QWidget()
        self.tPlotting.setObjectName("tPlotting")
        self.verticalLayout_12 = QtWidgets.QVBoxLayout(self.tPlotting)
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.groupBox_8 = QtWidgets.QGroupBox(self.tPlotting)
        self.groupBox_8.setObjectName("groupBox_8")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout(self.groupBox_8)
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.same_color_in_group = QtWidgets.QCheckBox(self.groupBox_8)
        self.same_color_in_group.setObjectName("same_color_in_group")
        self.verticalLayout_11.addWidget(self.same_color_in_group)
        self.different_line_style_among_groups = QtWidgets.QCheckBox(self.groupBox_8)
        self.different_line_style_among_groups.setObjectName("different_line_style_among_groups")
        self.verticalLayout_11.addWidget(self.different_line_style_among_groups)
        self.groupBox_9 = QtWidgets.QGroupBox(self.groupBox_8)
        self.groupBox_9.setObjectName("groupBox_9")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.groupBox_9)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.rbDefaultColorScheme = QtWidgets.QRadioButton(self.groupBox_9)
        self.rbDefaultColorScheme.setObjectName("rbDefaultColorScheme")
        self.verticalLayout_7.addWidget(self.rbDefaultColorScheme)
        self.rbHSVColorScheme = QtWidgets.QRadioButton(self.groupBox_9)
        self.rbHSVColorScheme.setObjectName("rbHSVColorScheme")
        self.verticalLayout_7.addWidget(self.rbHSVColorScheme)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_32 = QtWidgets.QLabel(self.groupBox_9)
        self.label_32.setObjectName("label_32")
        self.horizontalLayout_4.addWidget(self.label_32)
        self.sbHues = QtWidgets.QSpinBox(self.groupBox_9)
        self.sbHues.setMinimum(1)
        self.sbHues.setMaximum(999999)
        self.sbHues.setObjectName("sbHues")
        self.horizontalLayout_4.addWidget(self.sbHues)
        self.verticalLayout_7.addLayout(self.horizontalLayout_4)
        self.gridLayout_14 = QtWidgets.QGridLayout()
        self.gridLayout_14.setObjectName("gridLayout_14")
        self.label_27 = QtWidgets.QLabel(self.groupBox_9)
        self.label_27.setObjectName("label_27")
        self.gridLayout_14.addWidget(self.label_27, 0, 2, 1, 1)
        self.sbMinHue = QtWidgets.QSpinBox(self.groupBox_9)
        self.sbMinHue.setMinimum(0)
        self.sbMinHue.setMaximum(360)
        self.sbMinHue.setObjectName("sbMinHue")
        self.gridLayout_14.addWidget(self.sbMinHue, 0, 3, 1, 1)
        self.label_28 = QtWidgets.QLabel(self.groupBox_9)
        self.label_28.setObjectName("label_28")
        self.gridLayout_14.addWidget(self.label_28, 0, 4, 1, 1)
        self.sbMaxHue = QtWidgets.QSpinBox(self.groupBox_9)
        self.sbMaxHue.setMinimum(0)
        self.sbMaxHue.setMaximum(360)
        self.sbMaxHue.setObjectName("sbMaxHue")
        self.gridLayout_14.addWidget(self.sbMaxHue, 0, 5, 1, 1)
        self.label_26 = QtWidgets.QLabel(self.groupBox_9)
        self.label_26.setObjectName("label_26")
        self.gridLayout_14.addWidget(self.label_26, 1, 0, 1, 1)
        self.sbValues = QtWidgets.QSpinBox(self.groupBox_9)
        self.sbValues.setMinimum(1)
        self.sbValues.setMaximum(999999)
        self.sbValues.setObjectName("sbValues")
        self.gridLayout_14.addWidget(self.sbValues, 1, 1, 1, 1)
        self.label_29 = QtWidgets.QLabel(self.groupBox_9)
        self.label_29.setObjectName("label_29")
        self.gridLayout_14.addWidget(self.label_29, 1, 2, 1, 1)
        self.sbMinValue = QtWidgets.QSpinBox(self.groupBox_9)
        self.sbMinValue.setMinimum(0)
        self.sbMinValue.setMaximum(255)
        self.sbMinValue.setObjectName("sbMinValue")
        self.gridLayout_14.addWidget(self.sbMinValue, 1, 3, 1, 1)
        self.label_30 = QtWidgets.QLabel(self.groupBox_9)
        self.label_30.setObjectName("label_30")
        self.gridLayout_14.addWidget(self.label_30, 1, 4, 1, 1)
        self.sbMaxValue = QtWidgets.QSpinBox(self.groupBox_9)
        self.sbMaxValue.setMinimum(0)
        self.sbMaxValue.setMaximum(255)
        self.sbMaxValue.setObjectName("sbMaxValue")
        self.gridLayout_14.addWidget(self.sbMaxValue, 1, 5, 1, 1)
        self.verticalLayout_7.addLayout(self.gridLayout_14)
        self.rbUserColorScheme = QtWidgets.QRadioButton(self.groupBox_9)
        self.rbUserColorScheme.setObjectName("rbUserColorScheme")
        self.verticalLayout_7.addWidget(self.rbUserColorScheme)
        self.txbUserColorScheme = QtWidgets.QPlainTextEdit(self.groupBox_9)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.txbUserColorScheme.sizePolicy().hasHeightForWidth())
        self.txbUserColorScheme.setSizePolicy(sizePolicy)
        self.txbUserColorScheme.setMinimumSize(QtCore.QSize(0, 0))
        self.txbUserColorScheme.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.txbUserColorScheme.setObjectName("txbUserColorScheme")
        self.verticalLayout_7.addWidget(self.txbUserColorScheme)
        self.cbHSVReversed = QtWidgets.QCheckBox(self.groupBox_9)
        self.cbHSVReversed.setObjectName("cbHSVReversed")
        self.verticalLayout_7.addWidget(self.cbHSVReversed)
        self.verticalLayout_11.addWidget(self.groupBox_9)
        self.verticalLayout_12.addWidget(self.groupBox_8)
        self.antialiasing = QtWidgets.QCheckBox(self.tPlotting)
        self.antialiasing.setObjectName("antialiasing")
        self.verticalLayout_12.addWidget(self.antialiasing)
        self.show_grid = QtWidgets.QCheckBox(self.tPlotting)
        self.show_grid.setObjectName("show_grid")
        self.verticalLayout_12.addWidget(self.show_grid)
        self.chbReverseZOrder = QtWidgets.QCheckBox(self.tPlotting)
        self.chbReverseZOrder.setObjectName("chbReverseZOrder")
        self.verticalLayout_12.addWidget(self.chbReverseZOrder)
        self.gridLayout_13 = QtWidgets.QGridLayout()
        self.gridLayout_13.setObjectName("gridLayout_13")
        self.label_23 = QtWidgets.QLabel(self.tPlotting)
        self.label_23.setObjectName("label_23")
        self.gridLayout_13.addWidget(self.label_23, 4, 0, 1, 1)
        self.label_31 = QtWidgets.QLabel(self.tPlotting)
        self.label_31.setObjectName("label_31")
        self.gridLayout_13.addWidget(self.label_31, 0, 0, 1, 1)
        self.label_21 = QtWidgets.QLabel(self.tPlotting)
        self.label_21.setObjectName("label_21")
        self.gridLayout_13.addWidget(self.label_21, 2, 0, 1, 1)
        self.left_axis_label = QtWidgets.QLineEdit(self.tPlotting)
        self.left_axis_label.setObjectName("left_axis_label")
        self.gridLayout_13.addWidget(self.left_axis_label, 2, 2, 1, 1)
        self.line_width = QtWidgets.QDoubleSpinBox(self.tPlotting)
        self.line_width.setDecimals(1)
        self.line_width.setMinimum(0.1)
        self.line_width.setMaximum(50.0)
        self.line_width.setProperty("value", 1.0)
        self.line_width.setObjectName("line_width")
        self.gridLayout_13.addWidget(self.line_width, 4, 2, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.tPlotting)
        self.label_14.setObjectName("label_14")
        self.gridLayout_13.addWidget(self.label_14, 3, 0, 1, 1)
        self.legend_spacing = QtWidgets.QSpinBox(self.tPlotting)
        self.legend_spacing.setMaximum(200)
        self.legend_spacing.setSingleStep(1)
        self.legend_spacing.setObjectName("legend_spacing")
        self.gridLayout_13.addWidget(self.legend_spacing, 5, 2, 1, 1)
        self.label_22 = QtWidgets.QLabel(self.tPlotting)
        self.label_22.setObjectName("label_22")
        self.gridLayout_13.addWidget(self.label_22, 1, 0, 1, 1)
        self.label_24 = QtWidgets.QLabel(self.tPlotting)
        self.label_24.setObjectName("label_24")
        self.gridLayout_13.addWidget(self.label_24, 5, 0, 1, 1)
        self.bottom_axis_label = QtWidgets.QLineEdit(self.tPlotting)
        self.bottom_axis_label.setObjectName("bottom_axis_label")
        self.gridLayout_13.addWidget(self.bottom_axis_label, 3, 2, 1, 1)
        self.graph_title = QtWidgets.QLineEdit(self.tPlotting)
        self.graph_title.setObjectName("graph_title")
        self.gridLayout_13.addWidget(self.graph_title, 1, 2, 1, 1)
        self.grid_alpha = QtWidgets.QDoubleSpinBox(self.tPlotting)
        self.grid_alpha.setDecimals(1)
        self.grid_alpha.setMinimum(0.0)
        self.grid_alpha.setMaximum(1.0)
        self.grid_alpha.setSingleStep(0.1)
        self.grid_alpha.setProperty("value", 0.3)
        self.grid_alpha.setObjectName("grid_alpha")
        self.gridLayout_13.addWidget(self.grid_alpha, 0, 2, 1, 1)
        self.verticalLayout_12.addLayout(self.gridLayout_13)
        self.tabWidget.addTab(self.tPlotting, "")
        self.verticalLayout.addWidget(self.tabWidget)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.btnRestoreDefaultSettings = QtWidgets.QPushButton(Dialog)
        self.btnRestoreDefaultSettings.setMinimumSize(QtCore.QSize(170, 0))
        self.btnRestoreDefaultSettings.setObjectName("btnRestoreDefaultSettings")
        self.horizontalLayout.addWidget(self.btnRestoreDefaultSettings)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.horizontalLayout.addWidget(self.buttonBox)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.gridLayout_5.addLayout(self.verticalLayout, 0, 0, 1, 1)

        self.retranslateUi(Dialog)
        self.tabWidget.setCurrentIndex(0)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        Dialog.setTabOrder(self.tabWidget, self.general_imp_delimiter)
        Dialog.setTabOrder(self.general_imp_delimiter, self.general_imp_decimal_sep)
        Dialog.setTabOrder(self.general_imp_decimal_sep, self.excel_imp_as_text)
        Dialog.setTabOrder(self.excel_imp_as_text, self.clip_imp_delimiter)
        Dialog.setTabOrder(self.clip_imp_delimiter, self.clip_imp_decimal_sep)
        Dialog.setTabOrder(self.clip_imp_decimal_sep, self.clip_exp_include_group_name)
        Dialog.setTabOrder(self.clip_exp_include_group_name, self.clip_exp_include_header)
        Dialog.setTabOrder(self.clip_exp_include_header, self.clip_exp_delimiter)
        Dialog.setTabOrder(self.clip_exp_delimiter, self.clip_exp_decimal_sep)
        Dialog.setTabOrder(self.clip_exp_decimal_sep, self.left_axis_label)
        Dialog.setTabOrder(self.left_axis_label, self.bottom_axis_label)
        Dialog.setTabOrder(self.bottom_axis_label, self.btnRestoreDefaultSettings)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label_11.setText(_translate("Dialog", "\\n - new line, \\r - cartrige return, \\t - tabulator\n"
"If delimiter field is empty, lines will be splitted by any whitespace (\' \', \'\\n\', \'\\r\', \'\\t\', \'\\f\', \'\\v\') and empty strings will be discarded (empty entries will be always removed). Note, that this does not apply for CSV file, if delimiter for CSV file is empty, default comma \',\' will be used."))
        self.groupBox_7.setTitle(_translate("Dialog", "CSV and Other files + Clipboard settings"))
        self.cbRemoveEmptyEntries.setText(_translate("Dialog", "Remove empty entries for each parsed line"))
        self.label_12.setText(_translate("Dialog", "Skip first n columns:"))
        self.cbSkipNaNColumns.setText(_translate("Dialog", "Skip columns that contain NaN (Not a Number) values"))
        self.label_13.setText(_translate("Dialog", "Replacement for NaN values:"))
        self.groupBox.setTitle(_translate("Dialog", "Files"))
        self.label_3.setText(_translate("Dialog", "Delimiter"))
        self.label_7.setText(_translate("Dialog", "Other (*.txt, etc.)"))
        self.label_4.setText(_translate("Dialog", "Decimal separator"))
        self.label_6.setText(_translate("Dialog", "DX file"))
        self.label_5.setText(_translate("Dialog", "CSV file"))
        self.groupBox_5.setTitle(_translate("Dialog", "DX file"))
        self.dx_import_spectra_name_from_filename.setText(_translate("Dialog", "Import spectra name\n"
"from filename"))
        self.rb_DX_import_from_title.setText(_translate("Dialog", "Import spectra name\n"
"from ##TITLE entry"))
        self.dx_if_title_is_empty_use_filename.setText(_translate("Dialog", "If ##TITLE is empty, use\n"
"spectra name from filename"))
        self.groupBox_6.setTitle(_translate("Dialog", "CSV and Other files"))
        self.label.setText(_translate("Dialog", "If possible (for non-concatenated data),"))
        self.general_import_spectra_name_from_filename.setText(_translate("Dialog", "Import spectra name\n"
"from filename"))
        self.rb_general_import_from_header.setText(_translate("Dialog", "Import spectra name\n"
"from header"))
        self.general_if_header_is_empty_use_filename.setText(_translate("Dialog", "If header is empty, use\n"
"spectra name from filename"))
        self.groupBox_2.setTitle(_translate("Dialog", "Clipboard"))
        self.excel_imp_as_text.setText(_translate("Dialog", "Import data from MS Excel as text (if checked, decimal precision will be lost)"))
        self.label_10.setText(_translate("Dialog", "Set how to import text data from clipboard (in order to retain compatibility with MS Excel/Origin, keep delimiter set as tabulator \\t or empty (any whitespace characters), decimal separator is application specific)"))
        self.label_8.setText(_translate("Dialog", "Delimiter"))
        self.label_9.setText(_translate("Dialog", "Decimal separator"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tImport), _translate("Dialog", "Import"))
        self.groupBox_4.setTitle(_translate("Dialog", "Clipboard"))
        self.label_18.setText(_translate("Dialog", "Set how to export text data to clipboard (in order to retain compatibility with MS Excel/Origin, keep delimiter set as tabulator \\t, decimal separator is application specific)"))
        self.clip_exp_include_group_name.setText(_translate("Dialog", "Include group name"))
        self.clip_exp_include_header.setText(_translate("Dialog", "Include header"))
        self.label_19.setText(_translate("Dialog", "Delimiter"))
        self.label_20.setText(_translate("Dialog", "Decimal separator"))
        self.groupBox_3.setTitle(_translate("Dialog", "Files"))
        self.files_exp_include_group_name.setText(_translate("Dialog", "Include group name"))
        self.files_exp_include_header.setText(_translate("Dialog", "Include header"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tExport), _translate("Dialog", "Export"))
        self.groupBox_8.setTitle(_translate("Dialog", "Color and line style"))
        self.same_color_in_group.setText(_translate("Dialog", "Plot spectra with same color in groups"))
        self.different_line_style_among_groups.setText(_translate("Dialog", "Plot spectra with different line style among groups"))
        self.groupBox_9.setTitle(_translate("Dialog", "Color schemes"))
        self.rbDefaultColorScheme.setText(_translate("Dialog", "Default (red, green, blue, black, yellow, magenta, cyan, gray, .... and repeat)"))
        self.rbHSVColorScheme.setText(_translate("Dialog", "HSV/HSB gradient"))
        self.label_32.setText(_translate("Dialog", "Number of spectra (for gradients)"))
        self.label_27.setText(_translate("Dialog", "Min hue"))
        self.label_28.setText(_translate("Dialog", "Max hue"))
        self.label_26.setText(_translate("Dialog", "Values / \n"
"Brightnesses"))
        self.label_29.setText(_translate("Dialog", "Min value"))
        self.label_30.setText(_translate("Dialog", "Max value"))
        self.rbUserColorScheme.setText(_translate("Dialog", "User defined gradient (1st column position - 0->1, next RGBA values - 0->1 each)"))
        self.cbHSVReversed.setText(_translate("Dialog", "Reversed color scheme"))
        self.antialiasing.setText(_translate("Dialog", "Antialiasing"))
        self.show_grid.setText(_translate("Dialog", "Show Grid"))
        self.chbReverseZOrder.setText(_translate("Dialog", "Reverse Z order (first spectra will be ploted behind next ones)"))
        self.label_23.setText(_translate("Dialog", "Line width"))
        self.label_31.setText(_translate("Dialog", "Grid alpha"))
        self.label_21.setText(_translate("Dialog", "Left axis label"))
        self.label_14.setText(_translate("Dialog", "Bottom axis label"))
        self.label_22.setText(_translate("Dialog", "Graph title"))
        self.label_24.setText(_translate("Dialog", "Legend spacng"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tPlotting), _translate("Dialog", "Plotting"))
        self.btnRestoreDefaultSettings.setText(_translate("Dialog", "Restore Default Settings"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
