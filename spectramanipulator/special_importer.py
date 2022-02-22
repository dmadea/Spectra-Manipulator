import os
import numpy as np
from .dataloader import parse_files_specific
from .spectrum import SpectrumList
from .settings.settings import Settings
from PyQt5.QtWidgets import QFileDialog
from .parsers.hplc_dxfileparser import parse_HPLC_DX_file
import re
from .logger import Logger


def open_file_dialog(caption='Open ...', initial_dir='...', _filter='All Files (*.*)',
                     initial_filter='All Files (*.*)', choose_multiple=False):

    f = QFileDialog.getOpenFileNames if choose_multiple else QFileDialog.getOpenFileName

    filepaths = f(caption=caption,
                  directory=initial_dir,
                  filter=_filter,
                  initialFilter=initial_filter)

    if not choose_multiple and filepaths[0] == '':
        return None

    if choose_multiple and len(filepaths[0]) < 1:
        return None

    return filepaths[0]


def import_DX_HPLC_files():
    sett = Settings()

    filepaths = open_file_dialog("Import New Agilent HPLC chromatogram", sett['/Private settings/Import DX file dialog path'],
                                 _filter="Agilent HPLC DX Files (*.dx, *.DX);;All Files (*.*)",
                                 initial_filter="Agilent HPLC DX Files (*.dx, *.DX)",
                                 choose_multiple=True)
    if filepaths is None:
        return

    sett['/Private settings/Import DX file dialog path'] = os.path.dirname(filepaths[0])
    sett.save()

    spectral_data = []

    for filepath in filepaths:
        spectral_data += parse_HPLC_DX_file(filepath)

    return spectral_data


def import_LPF_kinetics():
    sett = Settings()

    filepaths = open_file_dialog("Import LFP Kinetics", sett['/Private settings/Import LPF dialog path'],
                                 _filter="Data Files (*.csv, *.CSV);;All Files (*.*)",
                                 initial_filter="Data Files (*.csv, *.CSV)",
                                 choose_multiple=True)

    if filepaths is None:
        return

    sett['/Private settings/Import LPF dialog path'] = os.path.dirname(filepaths[0])
    sett.save()

    kwargs = dict(delimiter=',',
                  decimal_sep='.',
                  remove_empty_entries=False,
                  skip_col_num=3,
                  general_import_spectra_name_from_filename=True,
                  skip_nan_columns=False,
                  nan_replacement=0)

    spectra, _ = parse_files_specific(filepaths, use_CSV_parser=False, **kwargs)
    if len(spectra) == 0:
        return

    max_abs = 3.5

    # CONVERT THE VOLTAGE (proportional to transmittance) TO ABSORBANCE
    for sp in spectra:
        # convert nan and pos infinities to value of max_abs = very high absorbance for nano
        sp.y = np.nan_to_num(-np.log10(-sp.y), nan=max_abs, posinf=max_abs, neginf=0)
        sp.y[sp.y > max_abs] = max_abs  # floor larger values to max absorbance

    return spectra

    # self.tree_widget.import_spectra(spectra)


def import_EEM_Duetta():
    """Used to import excitation emission map from Duetta Fluorimeter.
    Works only for proper data. Data has to be exported from finished kinetics. If the kinetic
    measurement was stopped during the measurement, the exported data will have different
    format and they cannot be imported this way.

    It sets extracted excitation wavelengths as the names of the spectra.
    """

    sett = Settings()

    filepaths = open_file_dialog("Import Excitation Emission Map from Duetta Fluorimeter",
                                 sett['/Private settings/Import EEM dialog path'],
                                 _filter="Data Files (*.txt, *.TXT);;All Files (*.*)",
                                 initial_filter="Data Files (*.txt, *.TXT)",
                                 choose_multiple=True)

    if filepaths is None:
        return

    sett['/Private settings/Import EEM dialog path'] = os.path.dirname(filepaths[0])
    sett.save()

    kwargs = dict(delimiter='\t',
                  decimal_sep='.',
                  remove_empty_entries=False,
                  skip_col_num=0,
                  general_import_spectra_name_from_filename=True,
                  skip_nan_columns=False,
                  nan_replacement=0)

    spectra, parsers = parse_files_specific(filepaths, use_CSV_parser=False, **kwargs)
    if len(spectra) == 0:
        return

    if not isinstance(spectra[0], SpectrumList):
        Logger.message(f"{type(spectra[0])} is not type SpectrumList. "
                       f"Unable to import data. Check the dataparsers.")
        return

    # eg. MeCN blank 2nm step 448:250-1100, [name] [ex]:[em1]-[em2]
    # we need 448 which is the excitation wavelength
    pattern = re.compile(r'(\d+):\d+-\d+')  # use regex to extract the ex. wavelength

    try:
        # extract the wavelengths from parsers
        for sl, parser in zip(spectra, parsers):
            names_list = parser.names_history[0]  # first line in names history
            assert len(names_list) == len(sl) + 1

            # extract the excitation wavelengths from the name history
            new_names = []
            sl_name = None
            for name in names_list:
                if name == '':
                    continue
                m = pattern.search(name)
                if m is None:
                    continue
                new_names.append(m.group(1))
                sl_name = name.replace(m.group(0), '').strip()

            # set the main name
            if sl_name:
                sl.name = sl_name

            # remove each 2nd spectrum as it contains useless X values (starting from second spectrum)
            del sl.children[1::2]

            # setup extracted names = excitation wavelengths
            sl.set_names(new_names)

            # 'sort' the list, the data are imported in opposite way so we can just reverse the list
            sl.children = sl.children[::-1]
    except Exception as e:
        Logger.message(f"Unable to import data: {e.__str__()}")
        return

    return spectra
    # self.tree_widget.import_spectra(spectra)


def import_kinetics_Duetta():
    """Used to import emission kinetics from Duetta Fluorimeter.
    Works only for proper data. Data has to be exported from finished kinetics. If the kinetic
    measurement was stopped during the measurement, the exported data will have different
    format and they cannot be imported this way.

    It sets extracted times as a names of the spectra. First spectrum will have time=0.
    In other words, time of first spectrum will be subtracted from all spectra.
    """

    sett = Settings()

    filepaths = open_file_dialog("Import Kinetics from Duetta Fluorimeter",
                                 sett['/Private settings/Import EEM dialog path'],
                                 _filter="Data Files (*.txt, *.TXT);;All Files (*.*)",
                                 initial_filter="Data Files (*.txt, *.TXT)",
                                 choose_multiple=True)

    if filepaths is None:
        return

    sett['/Private settings/Import EEM dialog path'] = os.path.dirname(filepaths[0])
    sett.save()

    kwargs = dict(delimiter='\t',
                  decimal_sep='.',
                  remove_empty_entries=False,
                  skip_col_num=0,
                  general_import_spectra_name_from_filename=True,
                  skip_nan_columns=False,
                  nan_replacement=0)

    spectra, parsers = parse_files_specific(filepaths, use_CSV_parser=False, **kwargs)
    if len(spectra) == 0:
        return

    if not isinstance(spectra[0], SpectrumList):
        Logger.message(f"{type(spectra[0])} is not type SpectrumList. "
                       f"Unable to import data. Check the dataparsers.")
        return

    # eg. 2Z+MB 310:250-650,1.92,  [name] [ex]:[em1]-[em2],[time]
    # we need 1.92 which is the time at it was measured
    # matches the [ex]:[em1]-[em2],[time] pattern and use [time] as group
    pattern = re.compile(r'\d+:\d+-\d+,(\d+.\d+)')

    try:
        # extract the times from parsers
        for sl, parser in zip(spectra, parsers):
            names_list = parser.names_history[0]  # first line in names history
            assert len(names_list) == len(sl) + 1

            # extract the excitation wavelengths from the name history
            new_names = []
            sl_name = None  # name of the group
            first_time = None
            for name in names_list:
                if name == '':
                    continue
                m = pattern.search(name)
                if m is None:
                    continue

                time = float(m.group(1))
                if first_time is None:
                    first_time = time

                new_names.append(f'{time - first_time:.2f}')  # use 2 digits precision
                sl_name = name.replace(m.group(0), '').strip()

            # set the main name
            if sl_name:
                sl.name = sl_name

            # remove each 2nd spectrum as it contains useless X values (starting from second spectrum)
            del sl.children[1::2]

            # setup extracted names = excitation wavelengths
            sl.set_names(new_names)
    except Exception as e:
        Logger.message(f"Unable to import data: {e.__str__()}")
        return

    return spectra
    # self.tree_widget.import_spectra(spectra)

