from spectrum import Spectrum
from parsers.genericparser import GenericParser
import numpy as np


class GeneralParser(GenericParser):

    def __init__(self, filepath=None, str_data=None, delimiter='\t', decimal_sep='.', remove_empty_entries=True,
                 skip_col_num=0, general_import_spectra_name_from_filename=False,
                 general_if_header_is_empty_use_filename=True):
        super(GeneralParser, self).__init__(filepath, str_data, delimiter, decimal_sep)

        self._spectra_buffer = []
        self.remove_empty_entries = remove_empty_entries
        self.skip_col_num = skip_col_num
        self.general_import_spectra_name_from_filename = general_import_spectra_name_from_filename
        self.general_if_header_is_empty_use_filename = general_if_header_is_empty_use_filename

    def float_try_parse(self, num_list):
        return [GenericParser.float_try_parse(self, num) for num in num_list]

    def _parse_data(self, data, names):
        if len(data) < 2:
            data.clear()
            names.clear()
            return False

        col_count = len(data[0])
        # row_count = len(data)

        if len(names) < len(data[0]):  # resize names of spectra so that len(names) == len(data[0])
            names += [''] * (col_count - len(names))

        spectra = []

        np_data = np.asarray(data, dtype=np.float64)  # convert parsed data to numpy matrix for easier manipulation

        for i in range(1, col_count):
            # skip columns that contains NAN values
            if any(np.isnan(np_data[:, i])):
                continue

            sp_data = np_data[:, [0, i]]

            sp = Spectrum(sp_data, self.filepath, names[i].strip(), self.name_of_file)
            spectra.append(sp)

        # for single spectrum, non-concacenated data
        if len(spectra) == 1:
            name = spectra[0].name
            if self.general_import_spectra_name_from_filename and not self.general_if_header_is_empty_use_filename:
                name = self.name_of_file
            if self.general_if_header_is_empty_use_filename:
                name = self.name_of_file if name == '' else name
            spectra[0].name = name

        if len(spectra) > 0:
            self._spectra_buffer.append(spectra[0] if len(spectra) == 1 else spectra)

        data.clear()
        names.clear()

        return True

    def line2list_iterator(self):
        """Iterator method that can be overridden. Get iterator that will iterate through a parsed line into a LIST"""

        # if delimiter is '', split by default - any whitespace and empty strings will be discarded
        delimiter = self.delimiter if self.delimiter != '' else None

        for line in self.__iter__():
            split_line = line.split(delimiter)

            yield split_line

    def parse(self, name=None):

        it = self.line2list_iterator()

        data = []
        names = []

        for line in it:
            # line is a parsed list - ['entry1', 'entry2', .... ]

            if len(line) < 2:
                self._parse_data(data, names)
                continue

            if len(line) != 0:
                # skip defined number of columns
                line = line[self.skip_col_num:]

                # remove empty entries
                if self.remove_empty_entries:
                    line = list(filter(None, line))

            l_values = self.float_try_parse(line)

            if l_values[0] is None:
                self._parse_data(data, names)
                names = line

                continue

            data.append(l_values)

        self._parse_data(data, names)

        return self._spectra_buffer
