
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from .parsers import GeneralParser, DXFileParser, CSVFileParser
from .settings import Settings
from .logger import Logger

# import time


def _process_filepath(args):
    filepath, settings = args
    try:
        parsed_spectra = _parse_file(filepath, settings)
        return None, parsed_spectra
    except Exception as ex:
        return "Error occurred while parsing file {}, skipping the file.\nException: {}\n".format(filepath, ex.__str__()), None


def parse_files(filepaths, settings: dict = None):
    """


    :param filepaths:
    :param settings:
    :return:
    """
    if filepaths is None:
        raise ValueError("Filepaths cannot be None.")
    if isinstance(filepaths, str):
        return _parse_file(filepaths)
    if not isinstance(filepaths, list):
        raise ValueError("Filepaths must be a list of strings")

    filesizes = np.asarray([os.path.getsize(fname) for fname in filepaths])

    spectra = []

    settings_dict = Settings.get_attributes()
    if settings is not None:
        settings_dict.update(settings)
    # start_time = time.time()

    # if at least two files are >= 0.5 MB, switch to multiprocess
    if (Settings.enable_multiprocessing and filesizes.shape[0] > 1 and (filesizes >= 0.5e6).sum() > 1) or \
            Settings.force_multiprocess:

        # Logger.console_message("Loading in multiprocess.")

        # ex = ProcessPoolExecutor() if Settings.load_method == 'multiprocess' else ThreadPoolExecutor()
        with ProcessPoolExecutor() as executor:
            results = executor.map(_process_filepath, zip(filepaths, [settings_dict] * len(filepaths)))
            for res in results:
                if res[0] is not None:
                    Logger.message(res[0])
                    continue

                if res[1] is not None:
                    spectra += res[1]
    else:
        # Logger.console_message("Loading in the main thread.")
        for filepath in filepaths:
            # parsed_spectra is always a list of spectra, if parsing was unsuccessful, None is returned
            try:
                parsed_spectra = _parse_file(filepath, settings_dict)
            except Exception as ex:
                Logger.message(
                    "Error occurred while parsing file {}, skipping the file.\nException: {}\n".format(filepath, ex.__str__()))
                continue
            if parsed_spectra is not None:
                spectra += parsed_spectra

    # elapsed_time = time.time() - start_time
    # Logger.console_message(elapsed_time)

    if len(spectra) != 0:
        return spectra
    else:
        Logger.message("No lines were parsed, check delimiter, decimal separator, number of skipped columns and"
                       " skip in settings.")


def parse_text(text):
    txt_parser = GeneralParser(str_data=text, delimiter=Settings.clip_imp_delimiter,
                               decimal_sep=Settings.clip_imp_decimal_sep,
                               remove_empty_entries=Settings.remove_empty_entries,
                               skip_col_num=Settings.skip_columns_num,
                               general_import_spectra_name_from_filename=Settings.general_import_spectra_name_from_filename,
                               general_if_header_is_empty_use_filename=Settings.general_if_header_is_empty_use_filename,
                               skip_nan_columns=Settings.skip_nan_columns,
                               nan_replacement=Settings.nan_replacement)

    spectra = txt_parser.parse()
    if spectra is None or len(spectra) == 0:
        Logger.message("No lines were parsed, check delimiter, decimal separator and number of skipped columns in settings.")
    return spectra


def _parse_file(filepath, settings):
    file, ext = os.path.splitext(filepath)
    ext = ext.lower()

    if ext == '.csv':
        txt_parser = CSVFileParser(filepath, delimiter=settings['csv_imp_delimiter'],
                                   decimal_sep=settings['csv_imp_decimal_sep'],
                                   remove_empty_entries=settings['remove_empty_entries'],
                                   skip_col_num=settings['skip_columns_num'],
                                   general_import_spectra_name_from_filename=settings['general_import_spectra_name_from_filename'],
                                   general_if_header_is_empty_use_filename=settings['general_if_header_is_empty_use_filename'],
                                   skip_nan_columns=settings['skip_nan_columns'],
                                   nan_replacement=settings['nan_replacement'])
        return txt_parser.parse()
    # DX file
    elif ext == '.dx':
        dx_parser = DXFileParser(filepath, delimiter=settings['dx_imp_delimiter'],
                                 decimal_sep=settings['dx_imp_decimal_sep'],
                                 dx_import_spectra_name_from_filename=settings['dx_import_spectra_name_from_filename'],
                                 dx_if_title_is_empty_use_filename=settings['dx_if_title_is_empty_use_filename'])
        return dx_parser.parse()
    # general (could be any file)
    else:
        txt_parser = GeneralParser(filepath, delimiter=settings['general_imp_delimiter'],
                                   decimal_sep=settings['general_imp_decimal_sep'],
                                   remove_empty_entries=settings['remove_empty_entries'],
                                   skip_col_num=settings['skip_columns_num'],
                                   general_import_spectra_name_from_filename=settings['general_import_spectra_name_from_filename'],
                                   general_if_header_is_empty_use_filename=settings['general_if_header_is_empty_use_filename'],
                                   skip_nan_columns=settings['skip_nan_columns'],
                                   nan_replacement=settings['nan_replacement'])

        return txt_parser.parse()
