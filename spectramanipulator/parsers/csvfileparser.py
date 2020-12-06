import csv
from spectramanipulator.parsers.generalparser import GeneralParser


class CSVFileParser(GeneralParser):

    def __init__(self, filepath=None, str_data=None, delimiter=',', decimal_sep='.', remove_empty_entries=True,
                 skip_col_num=0, general_import_spectra_name_from_filename=False,
                 general_if_header_is_empty_use_filename=True, doublequote=True, skipinitialspace=True,
                 skip_nan_columns=False, nan_replacement=0):
        super(CSVFileParser, self).__init__(filepath, str_data, delimiter, decimal_sep, remove_empty_entries,
                                            skip_col_num,
                                            general_import_spectra_name_from_filename,
                                            general_if_header_is_empty_use_filename,
                                            skip_nan_columns, nan_replacement)

        self.doublequote = doublequote
        self.skipinitialspace = skipinitialspace

    def line2list_iterator(self):
        # get line iterator, return parsed list iterator, like eg. ['190', '1.5', '1.3', 'some value']
        it = self.__iter__()
        return csv.reader(it, doublequote=self.doublequote,
                          skipinitialspace=self.skipinitialspace,
                          delimiter=self.delimiter if self.delimiter != '' else ',')
