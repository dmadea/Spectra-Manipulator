import json
from spectramanipulator.logger import Logger
# import sys
import os


class Settings(object):
    """Static class containing user settings with static functions for saving and loading to/from json file."""

    # settings that starts with '_' will not be saved into a file
    # static variables

    _config_filename = "config.json"
    _inst_default_settings = None  # default settings instance

    # not in Settings dialog
    enable_multiprocessing = True  # enables importing files as separate processes
    force_multiprocess = False  # importing of files will only run in multiple processes

    # --------SETTINGS----------

    # not in Settings dialog
    general_models_dir = 'general models'
    REG_PROGRAM_NAME = 'SpectraManipulator.projectfile'
    PROJECT_EXTENSION = '.smpj'

    # import options

    remove_empty_entries = True
    skip_columns_num = 0
    skip_nan_columns = False
    nan_replacement = 0

    csv_imp_delimiter = ','
    csv_imp_decimal_sep = '.'
    general_imp_delimiter = ''
    general_imp_decimal_sep = '.'
    dx_imp_delimiter = ' '
    dx_imp_decimal_sep = '.'
    dx_import_spectra_name_from_filename = False
    dx_if_title_is_empty_use_filename = True
    general_import_spectra_name_from_filename = False
    general_if_header_is_empty_use_filename = True

    clip_imp_delimiter = '\t'
    clip_imp_decimal_sep = '.'

    excel_imp_as_text = False

    # file export options

    files_exp_include_group_name = False
    files_exp_include_header = True

    # clipboard export options

    clip_exp_include_group_name = True
    clip_exp_include_header = True
    clip_exp_delimiter = '\t'
    clip_exp_decimal_sep = '.'

    # plotting settings

    graph_title = ""

    antialiasing = True
    left_axis_label = "Absorbance"
    left_axis_unit = None

    bottom_axis_label = "Wavelength / nm"
    bottom_axis_unit = None

    show_grid = False
    grid_alpha = 0.1

    line_width = 1

    graph_title_font_size = 20
    bottom_axis_font_size = 20
    left_axis_font_size = 20

    same_color_in_group = False
    different_line_style_among_groups = False

    legend_spacing = 8

    color_scheme = 0  # 0 - default, 1 - HSV, 2 - user defined

    # plotwidget settings
    # not in Settings dialog
    coordinates_sig_figures = 4  # number of significant figures displayed in status bar for cooridnates

    # HSV_color_scheme = True
    hues = 9
    values = 1
    maxValue = 255
    minValue = 150
    maxHue = 360
    minHue = 0
    sat = 255  # saturation
    alpha = 255  # alpha

    user_defined_grad = "0.0\t1\t0\t0\t1\n0.5\t0\t1\t0\t1\n1.0\t0\t0\t1\t1"
    HSV_reversed = False  # reverse colors

    reverse_z_order = False

    # function plotter

    FP_num_of_points = 500  # number of points to be plotted on selected range for function plotter

    # rename dialog

    last_rename_expression = ""
    last_rename_take_name_from_list = False

    # files dialog last paths

    import_files_dialog_path = ""
    import_LPF_dialog_path = ""
    import_EEM_dialog_path = ""
    open_project_dialog_path = ""
    save_project_dialog_path = ""
    export_spectra_as_dialog_path = ""
    export_spectra_as_dialog_ext = '.txt'
    export_spectra_as_dialog_delimiter = '\t'
    export_spectra_as_dialog_decimal_sep = '.'

    # recent projects filepaths

    recent_project_filepaths = []

    gui_settings_last_tab_index = 0

    _project_independent_settings = [
        'export_spectra_as_dialog_path',
        'export_spectra_as_dialog_ext',
        'export_spectra_as_dialog_delimiter',
        'export_spectra_as_dialog_decimal_sep',
        'recent_project_filepaths',
        'import_files_dialog_path',
        'import_LPF_dialog_path',
        'import_EEM_dialog_path',
        'open_project_dialog_path',
        'save_project_dialog_path',
        'legend_spacing',
        'line_width'
    ]

    def __init__(self):
        """Saves the static attributes to attr and removes project independent settings."""
        self.attr = Settings.get_attributes()

        # delete settings that are project independent
        for s in Settings._project_independent_settings:
            del self.attr[s]

    def set_settings(self):
        """Sets static settings from this instance object (project settings)."""
        # dont set any settings
        pass

        # for key, value in self.attr.items():
        #     setattr(Settings, key, value)

        del self

    @classmethod
    def set_default_settings(cls):
        if Settings._inst_default_settings is not None:
            Settings._inst_default_settings.set_settings()

    @classmethod
    def get_attributes(cls):
        members = vars(cls)
        filtered = {attr: key for attr, key in members.items() if not attr.startswith('_') and
                    not callable(getattr(Settings, attr))}
        return filtered

    @classmethod
    def get_config_filepath(cls):
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(curr_dir, cls._config_filename)

    @classmethod
    def save(cls):

        sett_dict = cls.get_attributes()

        try:

            with open(cls.get_config_filepath(), "w") as file:
                json.dump(sett_dict, file, sort_keys=False, indent=4, separators=(',', ': '))

        except Exception as ex:
            Logger.message("Error saving settings to file {}. Error message:\n{}".format(
                Settings._config_filename, ex.__str__()))

    @classmethod
    def load(cls):
        # basically static constructor method, set up default settings instance
        if Settings._inst_default_settings is None:
            Settings._inst_default_settings = Settings()

        try:
            with open(cls.get_config_filepath(), "r") as file:
                data = json.load(file)

            # instance = object.__new__(cls)

            for key, value in data.items():
                setattr(cls, key, value)

        except Exception as ex:
            Logger.message(
                "Error loading settings from file {}, setting up default settings. Error message:\n{}".format(
                    Settings._config_filename, ex.__str__()))
            Settings.save()

# Settings.save()
#
# inst = Settings()
#
# # inst.save_project_dialog_path = "asaps pasdasdpa sdpaosapsdo apsdopaso pasokd aosdkaps paskdpaoskdpaoskdpaos"
#
# # inst.attr['normalize_range'] = (0, 0)
#
# inst.set_settings()
#
# print(Settings._get_attributes())
