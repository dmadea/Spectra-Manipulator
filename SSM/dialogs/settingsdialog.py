from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox

# from PyQt5 import *
from PyQt5.QtCore import Qt
# from PyQt5.QtWidgets import *
from .gui_settings import Ui_Dialog

from ..settings import Settings
# from logger import Logger


class SettingsDialog(QtWidgets.QDialog, Ui_Dialog):
    # static variables
    is_opened = False
    _instance = None

    def __init__(self, parent=None):
        super(SettingsDialog, self).__init__(parent, QtCore.Qt.WindowStaysOnTopHint)
        self.setupUi(self)
        self.setWindowTitle("Settings")

        self.tabWidget.setCurrentIndex(Settings.gui_settings_last_tab_index)
        self.dx_if_title_is_empty_use_filename.stateChanged.connect(self.dx_if_title_is_empty_use_filename_checked_changed)
        self.general_if_header_is_empty_use_filename.stateChanged.connect(self.general_if_header_is_empty_use_filename_checked_changed)

        self.rbDefaultColorScheme.toggled.connect(self.schemes_changes)
        self.rbHSVColorScheme.toggled.connect(self.schemes_changes)
        self.rbUserColorScheme.toggled.connect(self.schemes_changes)
        self.cbSkipNaNColumns.toggled.connect(self.cbSkipNaNColumns_toggled)

        self.btnRestoreDefaultSettings.clicked.connect(self.restore_default_settings)

        self.load_settings()

        SettingsDialog.is_opened = True
        SettingsDialog._instance = self

        self.show()
        self.exec()

    def cbSkipNaNColumns_toggled(self):
        self.sbNaNReplacement.setEnabled(not self.cbSkipNaNColumns.isChecked())

    def restore_default_settings(self):
        Settings.set_default_settings()
        self.load_settings()

    # def rbSkipColumns_changed(self):
    #     if self.rbSkipColumns.isChecked():
    #         self.sbColumns.setEnabled(True)
    #     else:
    #         self.sbColumns.setEnabled(False)

    def schemes_changes(self):
        if self.rbDefaultColorScheme.isChecked():
            self.sbHues.setEnabled(False)
            self.sbValues.setEnabled(False)
            self.sbMinHue.setEnabled(False)
            self.sbMaxHue.setEnabled(False)
            self.sbMinValue.setEnabled(False)
            self.sbMaxValue.setEnabled(False)
            self.txbUserColorScheme.setEnabled(False)

        if self.rbHSVColorScheme.isChecked():
            self.sbHues.setEnabled(True)
            self.sbValues.setEnabled(True)
            self.sbMinHue.setEnabled(True)
            self.sbMaxHue.setEnabled(True)
            self.sbMinValue.setEnabled(True)
            self.sbMaxValue.setEnabled(True)
            self.txbUserColorScheme.setEnabled(False)

        if self.rbUserColorScheme.isChecked():
            self.sbHues.setEnabled(True)
            self.sbValues.setEnabled(False)
            self.sbMinHue.setEnabled(False)
            self.sbMaxHue.setEnabled(False)
            self.sbMinValue.setEnabled(False)
            self.sbMaxValue.setEnabled(False)
            self.txbUserColorScheme.setEnabled(True)


    # def rbDefaultColorScheme_checked_changed(self):

    def dx_if_title_is_empty_use_filename_checked_changed(self):
        if self.dx_if_title_is_empty_use_filename.checkState() == Qt.Checked:
            self.dx_import_spectra_name_from_filename.setEnabled(False)
            self.rb_DX_import_from_title.setEnabled(False)
        else:
            self.dx_import_spectra_name_from_filename.setEnabled(True)
            self.rb_DX_import_from_title.setEnabled(True)

    def general_if_header_is_empty_use_filename_checked_changed(self):
        if self.general_if_header_is_empty_use_filename.checkState() == Qt.Checked:
            self.general_import_spectra_name_from_filename.setEnabled(False)
            self.rb_general_import_from_header.setEnabled(False)
        else:
            self.general_import_spectra_name_from_filename.setEnabled(True)
            self.rb_general_import_from_header.setEnabled(True)

    @staticmethod
    def check_state(checked):
        return Qt.Checked if checked else Qt.Unchecked

    @staticmethod
    def DEcheck_state(checked):
        return True if checked == Qt.Checked else False

    @staticmethod
    def textualize_special_chars(text):
        return text.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')

    @staticmethod
    def DEtextualize_special_chars(text):
        return text.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')

    def load_settings(self):
        # import options

        self.cbRemoveEmptyEntries.setCheckState(self.check_state(Settings.remove_empty_entries))
        self.sbColumns.setValue(Settings.skip_columns_num)

        self.csv_imp_delimiter.setText(self.textualize_special_chars(Settings.csv_imp_delimiter))
        self.csv_imp_decimal_sep.setText(self.textualize_special_chars(Settings.csv_imp_decimal_sep))
        self.general_imp_delimiter.setText(self.textualize_special_chars(Settings.general_imp_delimiter))
        self.general_imp_decimal_sep.setText(self.textualize_special_chars(Settings.general_imp_decimal_sep))
        self.dx_imp_delimiter.setText(self.textualize_special_chars(Settings.dx_imp_delimiter))
        self.dx_imp_decimal_sep.setText(self.textualize_special_chars(Settings.dx_imp_decimal_sep))

        self.dx_import_spectra_name_from_filename.setChecked(Settings.dx_import_spectra_name_from_filename)
        self.rb_DX_import_from_title.setChecked(not Settings.dx_import_spectra_name_from_filename)
        self.dx_if_title_is_empty_use_filename.setCheckState(self.check_state(Settings.dx_if_title_is_empty_use_filename))

        self.general_import_spectra_name_from_filename.setChecked(Settings.general_import_spectra_name_from_filename)
        self.rb_general_import_from_header.setChecked(not Settings.general_import_spectra_name_from_filename)
        self.general_if_header_is_empty_use_filename.setCheckState(
            self.check_state(Settings.general_if_header_is_empty_use_filename))

        self.cbSkipNaNColumns.setChecked(Settings.skip_nan_columns)
        self.sbNaNReplacement.setValue(Settings.nan_replacement)

        # perform change
        self.dx_if_title_is_empty_use_filename_checked_changed()

        self.clip_imp_delimiter.setText(self.textualize_special_chars(Settings.clip_imp_delimiter))
        self.clip_imp_decimal_sep.setText(self.textualize_special_chars(Settings.clip_imp_decimal_sep))

        self.excel_imp_as_text.setCheckState(self.check_state(Settings.excel_imp_as_text))

        # file export options

        self.files_exp_include_group_name.setCheckState(self.check_state(Settings.files_exp_include_group_name))
        self.files_exp_include_header.setCheckState(self.check_state(Settings.files_exp_include_header))
        # self.csv_exp_delimiter.setText(self.textualize_special_chars(Settings.csv_exp_delimiter))
        # self.csv_exp_decimal_sep.setText(self.textualize_special_chars(Settings.csv_exp_decimal_sep))
        # self.general_exp_delimiter.setText(self.textualize_special_chars(Settings.general_exp_delimiter))
        # self.general_exp_decimal_sep.setText(self.textualize_special_chars(Settings.general_exp_decimal_sep))

        # clipboard export options

        self.clip_exp_include_group_name.setCheckState(self.check_state(Settings.clip_exp_include_group_name))
        self.clip_exp_include_header.setCheckState(self.check_state(Settings.clip_exp_include_header))
        self.clip_exp_delimiter.setText(self.textualize_special_chars(Settings.clip_exp_delimiter))
        self.clip_exp_decimal_sep.setText(self.textualize_special_chars(Settings.clip_exp_decimal_sep))

        # plotting settings

        self.graph_title.setText(Settings.graph_title)
        self.antialiasing.setCheckState(self.check_state(Settings.antialiasing))
        self.left_axis_label.setText(Settings.left_axis_label)
        # self.left_axis_unit = None

        self.bottom_axis_label.setText(Settings.bottom_axis_label)
        # self.bottom_axis_unit = None

        self.show_grid.setCheckState(self.check_state(Settings.show_grid))
        self.grid_alpha.setValue(Settings.grid_alpha)

        self.line_width.setValue(Settings.line_width)

        self.same_color_in_group.setCheckState(self.check_state(Settings.same_color_in_group))
        self.different_line_style_among_groups.setCheckState(self.check_state(Settings.different_line_style_among_groups))

        self.legend_spacing.setValue(Settings.legend_spacing)

        if Settings.color_scheme == 0:
            self.rbDefaultColorScheme.setChecked(True)
        elif Settings.color_scheme == 1:
            self.rbHSVColorScheme.setChecked(True)
        else:
            self.rbUserColorScheme.setChecked(True)

        self.cbHSVReversed.setChecked(Settings.HSV_reversed)

        self.sbHues.setValue(Settings.hues)
        self.sbValues.setValue(Settings.values)
        self.sbMinHue.setValue(Settings.minHue)
        self.sbMaxHue.setValue(Settings.maxHue)
        self.sbMinValue.setValue(Settings.minValue)
        self.sbMaxValue.setValue(Settings.maxValue)

        self.txbUserColorScheme.setPlainText(Settings.user_defined_grad)
        self.chbReverseZOrder.setChecked(self.check_state(Settings.reverse_z_order))

    def save_settings(self):
        if self.csv_imp_decimal_sep.text() == "" or self.general_imp_decimal_sep.text() == "" or \
                self.dx_imp_decimal_sep.text() == "" or self.clip_imp_decimal_sep.text() == "" or \
                self.clip_exp_decimal_sep.text() == "" or self.clip_exp_delimiter == "":
            raise ValueError("Please check decimal separator fields, these cannot be empty.")

        # import options

        Settings.remove_empty_entries = self.DEcheck_state(self.cbRemoveEmptyEntries.checkState())
        Settings.skip_columns_num = int(self.sbColumns.value())

        Settings.csv_imp_delimiter = self.DEtextualize_special_chars(self.csv_imp_delimiter.text())
        Settings.csv_imp_decimal_sep = self.DEtextualize_special_chars(self.csv_imp_decimal_sep.text())
        Settings.general_imp_delimiter = self.DEtextualize_special_chars(self.general_imp_delimiter.text())
        Settings.general_imp_decimal_sep = self.DEtextualize_special_chars(self.general_imp_decimal_sep.text())
        Settings.dx_imp_delimiter = self.DEtextualize_special_chars(self.dx_imp_delimiter.text())
        Settings.dx_imp_decimal_sep = self.DEtextualize_special_chars(self.dx_imp_decimal_sep.text())

        Settings.dx_import_spectra_name_from_filename = self.dx_import_spectra_name_from_filename.isChecked()
        Settings.dx_if_title_is_empty_use_filename = self.DEcheck_state(self.dx_if_title_is_empty_use_filename.checkState())

        Settings.general_import_spectra_name_from_filename = self.general_import_spectra_name_from_filename.isChecked()
        Settings.general_if_header_is_empty_use_filename = self.DEcheck_state(
            self.general_if_header_is_empty_use_filename.checkState())

        Settings.skip_nan_columns = self.cbSkipNaNColumns.isChecked()
        Settings.nan_replacement = float(self.sbNaNReplacement.value())

        Settings.clip_imp_delimiter = self.DEtextualize_special_chars(self.clip_imp_delimiter.text())
        Settings.clip_imp_decimal_sep = self.DEtextualize_special_chars(self.clip_imp_decimal_sep.text())

        Settings.excel_imp_as_text = self.DEcheck_state(self.excel_imp_as_text.checkState())

        # file export options

        Settings.files_exp_include_group_name = self.DEcheck_state(self.files_exp_include_group_name.checkState())
        Settings.files_exp_include_header = self.DEcheck_state(self.files_exp_include_header.checkState())
        # Settings.csv_exp_delimiter = self.DEtextualize_special_chars(self.csv_exp_delimiter.text())
        # Settings.csv_exp_decimal_sep = self.DEtextualize_special_chars(self.csv_exp_decimal_sep.text())
        # Settings.general_exp_delimiter = self.DEtextualize_special_chars(self.general_exp_delimiter.text())
        # Settings.general_exp_decimal_sep = self.DEtextualize_special_chars(self.general_exp_decimal_sep.text())

        # clipboard export options

        Settings.clip_exp_include_group_name = self.DEcheck_state(self.clip_exp_include_group_name.checkState())
        Settings.clip_exp_include_header = self.DEcheck_state(self.clip_exp_include_header.checkState())
        Settings.clip_exp_delimiter = self.DEtextualize_special_chars(self.clip_exp_delimiter.text())
        Settings.clip_exp_decimal_sep = self.DEtextualize_special_chars(self.clip_exp_decimal_sep.text())

        # plotting settings

        Settings.graph_title = self.graph_title.text()
        Settings.antialiasing = self.DEcheck_state(self.antialiasing.checkState())
        Settings.left_axis_label = self.left_axis_label.text()
        # Settings.left_axis_unit = None

        Settings.bottom_axis_label = self.bottom_axis_label.text()
        # bottom_axis_unit = None

        Settings.show_grid = self.DEcheck_state(self.show_grid.checkState())
        Settings.grid_alpha = self.grid_alpha.value()

        Settings.line_width = self.line_width.value()

        Settings.same_color_in_group = self.DEcheck_state(self.same_color_in_group.checkState())
        Settings.different_line_style_among_groups = self.DEcheck_state(self.different_line_style_among_groups.checkState())

        Settings.legend_spacing = self.legend_spacing.value()

        # Settings.HSV_color_scheme = self.rbHSVColorScheme.isChecked()
        if self.rbDefaultColorScheme.isChecked():
            Settings.color_scheme = 0
        elif self.rbHSVColorScheme.isChecked():
            Settings.color_scheme = 1
        else:
            Settings.color_scheme = 2

        Settings.hues = self.sbHues.value()
        Settings.values = self.sbValues.value()
        Settings.minHue = self.sbMinHue.value()
        Settings.maxHue = self.sbMaxHue.value()
        Settings.minValue = self.sbMinValue.value()
        Settings.maxValue = self.sbMaxValue.value()
        Settings.HSV_reversed = self.cbHSVReversed.isChecked()

        Settings.user_defined_grad = self.txbUserColorScheme.toPlainText()
        Settings.reverse_z_order = self.DEcheck_state(self.chbReverseZOrder.checkState())



    @staticmethod
    def get_instance():
        return SettingsDialog._instance

    def accept(self):
        try:
            self.save_settings()
        except ValueError as ex:
            QMessageBox.warning(self, 'Information', ex.__str__(), QMessageBox.Ok)
            return

        Settings.gui_settings_last_tab_index = self.tabWidget.currentIndex()
        Settings.save()

        self.accepted = True
        SettingsDialog.is_opened = False
        SettingsDialog._instance = None
        super(SettingsDialog, self).accept()

    def reject(self):
        SettingsDialog.is_opened = False
        SettingsDialog._instance = None
        super(SettingsDialog, self).reject()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    Dialog = SettingsDialog()
    # Dialog.show()
    sys.exit(app.exec_())
