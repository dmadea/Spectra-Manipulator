import numpy as np
from spectramanipulator.spectrum import Spectrum, SpectrumList
from spectramanipulator.parsers.parser import Parser


class GeneralParser(Parser):

    def __init__(self, filepath=None, str_data=None, delimiter='\t', decimal_sep='.', remove_empty_entries=True,
                 skip_col_num=0, general_import_spectra_name_from_filename=False,
                 general_if_header_is_empty_use_filename=True, skip_nan_columns=False, nan_replacement=0):
        super(GeneralParser, self).__init__(filepath, str_data, delimiter, decimal_sep)

        self._spectra_buffer = []
        self.remove_empty_entries = remove_empty_entries
        self.skip_col_num = skip_col_num
        self.general_import_spectra_name_from_filename = general_import_spectra_name_from_filename
        self.general_if_header_is_empty_use_filename = general_if_header_is_empty_use_filename
        self.skip_nan_columns = skip_nan_columns
        self.nan_replacement = nan_replacement

        self.names_history = []

    def line2list_iterator(self):
        """Iterator method that can be overridden. Get iterator that will iterate through a parsed line into a LIST"""

        # if delimiter is '', split by default - any whitespace and empty strings will be discarded
        delimiter = self.delimiter if self.delimiter != '' else None

        for line in self.__iter__():
            split_line = line.split(delimiter)

            yield split_line

    def parse(self, name=None):

        data = []
        last_names = []

        for line in self.line2list_iterator():
            # line is a parsed list - ['entry1', 'entry2', .... ]

            if len(line) < 2:
                self._parse_chunk(data, last_names)
                continue

            if len(line) != 0:
                # skip defined number of columns
                line = line[self.skip_col_num:]

                # remove empty entries
                if self.remove_empty_entries:
                    line = list(filter(None, line))

            l_values = [self.float_try_parse(num) for num in line]

            if l_values[0] is None:
                self._parse_chunk(data, last_names)
                last_names = line
                self.names_history.append(last_names.copy())

                continue

            data.append(l_values)

        self._parse_chunk(data, last_names)

        return self._spectra_buffer

    def _parse_chunk(self, data, names):
        if len(data) < 2:
            data.clear()
            names.clear()
            return

        col_count = len(data[0])

        if len(names) < len(data[0]):  # resize names of spectra so that len(names) == len(data[0])
            names += [''] * (col_count - len(names))

        # convert parsed data to numpy matrix for easier manipulation
        # by setting a dtype=np.float64, invalid entries will become np.nan
        np_data = np.asarray(data, dtype=np.float64)

        if self.skip_nan_columns:  # if option skip columns containing nan values is true
            nan_cols = np.isnan(np_data).sum(axis=0, keepdims=False)
            value_cols_idxs = np.argwhere(nan_cols == 0).squeeze()
            np_data = np_data[:, value_cols_idxs]
            names = [names[i] for i in value_cols_idxs]   # names is a list, not ndarray
        else:
            np_data = np.nan_to_num(np_data, nan=self.nan_replacement)  # convert nan values to used defined

        # sort according to first column (x values)
        np_data = np_data[np_data[:, 0].argsort()]

        spectra = SpectrumList()
        for i in range(1, np_data.shape[1]):
            # separate the matrix to individual spectra, x values are from first column, the data are from i-th column
            sp_data = np_data[:, [0, i]]

            sp = Spectrum(sp_data, filepath=self.filepath, name=names[i].strip(), assume_sorted=True)
            spectra.children.append(sp)
            spectra.name = self.name_of_file

        # for single spectrum, non-concatenated data
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
