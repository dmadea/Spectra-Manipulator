import os

from parsers.dxfileparser import DXFileParser
from parsers.csvfileparser import CSVFileParser
from parsers.generalparser import GeneralParser
from parsers.xmlspreadsheetparser import parse_XML_Spreadsheet as _parse_XML_Spreadsheet

from settings import Settings

from logger import Logger


def parse_XML_Spreadsheet(xml_text):
    return _parse_XML_Spreadsheet(xml_text)


def parse_files(filepaths):
    if filepaths is None:
        raise ValueError("Filepaths cannot be None.")
    if isinstance(filepaths, str):
        return _parse_file(filepaths)
    if not isinstance(filepaths, list):
        raise ValueError("Filepaths must be a list of strings")
    spectra = []
    for filepath in filepaths:
        # parsed_spectra is always a list of spectra, if parsing was unsuccessful, None is returned
        try:
            parsed_spectra = _parse_file(filepath)
        except Exception as ex:
            Logger.message(
                "Error occurred while parsing file {}, skipping the file.\n{}".format(filepath, ex.__str__()))
            continue
        if parsed_spectra is not None:
            spectra += parsed_spectra
    if len(spectra) != 0:
        return spectra
    else:
        Logger.message("No lines were parsed, check delimiter, decimal separator and number of skipped columns in settings.")


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


def _parse_file(filepath):
    file, ext = os.path.splitext(filepath)
    ext = ext.lower()

    if ext == '.csv':
        txt_parser = CSVFileParser(filepath, delimiter=Settings.csv_imp_delimiter,
                                   decimal_sep=Settings.csv_imp_decimal_sep,
                                   remove_empty_entries=Settings.remove_empty_entries,
                                   skip_col_num=Settings.skip_columns_num,
                                   general_import_spectra_name_from_filename=Settings.general_import_spectra_name_from_filename,
                                   general_if_header_is_empty_use_filename=Settings.general_if_header_is_empty_use_filename,
                                   skip_nan_columns=Settings.skip_nan_columns,
                                   nan_replacement=Settings.nan_replacement)
        return txt_parser.parse()
    # DX file
    elif ext == '.dx':
        dx_parser = DXFileParser(filepath, delimiter=Settings.dx_imp_delimiter,
                                 decimal_sep=Settings.dx_imp_decimal_sep,
                                 dx_import_spectra_name_from_filename=Settings.dx_import_spectra_name_from_filename,
                                 dx_if_title_is_empty_use_filename=Settings.dx_if_title_is_empty_use_filename)
        return dx_parser.parse()
    # general (could be any file)
    else:
        txt_parser = GeneralParser(filepath, delimiter=Settings.general_imp_delimiter,
                                   decimal_sep=Settings.general_imp_decimal_sep,
                                   remove_empty_entries=Settings.remove_empty_entries,
                                   skip_col_num=Settings.skip_columns_num,
                                   general_import_spectra_name_from_filename=Settings.general_import_spectra_name_from_filename,
                                   general_if_header_is_empty_use_filename=Settings.general_if_header_is_empty_use_filename,
                                   skip_nan_columns=Settings.skip_nan_columns,
                                   nan_replacement=Settings.nan_replacement)

        return txt_parser.parse()
