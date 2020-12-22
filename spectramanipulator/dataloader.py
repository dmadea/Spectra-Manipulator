
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
# import time

from .parsers import GeneralParser, DXFileParser, CSVFileParser
from .settings import Settings
from .logger import Logger


def parse_files(filepaths, settings: dict = None):
    settings_dict = Settings.get_attributes()
    if settings is not None:
        settings_dict.update(settings)

    parse_func = partial(_process_filepath, settings=settings_dict)
    return _parse_files(filepaths, parse_func)


def parse_files_specific(filepaths, use_CSV_parser=False, **kwargs):
    """kwargs are passed to the parser"""

    parse_func = partial(_process_filepath_specific, use_CSV_parser=use_CSV_parser, **kwargs)
    return _parse_files(filepaths, parse_func)


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


def _process_filepath(filepath, settings):
    try:
        parsed_spectra, parser = _parse_file(filepath, settings)
        return None, parsed_spectra, parser
    except Exception as ex:
        return "Error occurred while parsing file {}, " \
               "skipping the file.\nException: {}\n".format(filepath, ex.__str__()), None, None


def _process_filepath_specific(filepath, use_CSV_parser=True, **kwargs):
    try:
        parser = CSVFileParser(filepath, **kwargs) if use_CSV_parser else GeneralParser(filepath, **kwargs)
        return None, parser.parse(), parser
    except Exception as ex:
        return "Error occurred while parsing file {}, " \
               "skipping the file.\nException: {}\n".format(filepath, ex.__str__()), None, None


def _parse_files(filepaths, parse_func):
    """
    parse_func takes filepath as argument and returns the tuple with error and result.

    :param filepaths:
    :param settings:
    :return:
    """

    if filepaths is None:
        raise ValueError("Filepaths cannot be None.")
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    if not isinstance(filepaths, list):
        raise ValueError("Filepaths must be a list of strings")

    filesizes = np.asarray([os.path.getsize(fname) for fname in filepaths])
    spectra = []
    parsers = []

    # start_time = time.time()

    # if at least two files are >= 0.5 MB, switch to multiprocess
    if (os.cpu_count() > 1 and Settings.enable_multiprocessing and filesizes.shape[0] > 1 and
       (filesizes >= 0.5e6).sum() > 1) or Settings.force_multiprocess:

        # Logger.console_message("Loading in multiprocess.")

        with ProcessPoolExecutor() as executor:
            results = executor.map(parse_func, filepaths)

            for err, p_spectra, parser in results:
                if err is not None:
                    Logger.message(err)
                    continue

                if p_spectra is not None:
                    spectra += p_spectra
                    parsers.append(parser)
    else:
        # Logger.console_message("Loading in the main thread.")
        for filepath in filepaths:
            err, p_spectra, parser = parse_func(filepath)

            if err is not None:
                Logger.message(err)
                continue

            if p_spectra is not None:
                spectra += p_spectra
                parsers.append(parser)

    # elapsed_time = time.time() - start_time
    # Logger.console_message(elapsed_time)

    if len(spectra) != 0:
        return spectra, parsers
    else:
        Logger.message("No lines were parsed, check delimiter, decimal separator, number of skipped columns and"
                       " skip in settings.")
    return None, None


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
        return txt_parser.parse(), txt_parser
    # DX file
    elif ext == '.dx':
        dx_parser = DXFileParser(filepath, delimiter=settings['dx_imp_delimiter'],
                                 decimal_sep=settings['dx_imp_decimal_sep'],
                                 dx_import_spectra_name_from_filename=settings['dx_import_spectra_name_from_filename'],
                                 dx_if_title_is_empty_use_filename=settings['dx_if_title_is_empty_use_filename'])
        return dx_parser.parse(), dx_parser
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

        return txt_parser.parse(), txt_parser
