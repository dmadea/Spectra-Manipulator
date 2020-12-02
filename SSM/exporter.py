
import os
import io
import numpy as np
import csv
from SSM import Spectrum


def list_to_files(list_of_spectra, dir_path, extension, include_group_name=True, include_header=True,
                  delimiter='\t', decimal_sep='.', x_data_name='Wavelength / nm'):

    _list_to_stream(list_of_spectra, include_group_name, include_header, delimiter,
                             decimal_sep, True, dir_path, extension, x_data_name)


def list_to_string(list_of_spectra, include_group_name=True, include_header=True, delimiter='\t',
                   decimal_sep='.', x_data_name='Wavelength / nm'):

    return _list_to_stream(list_of_spectra, include_group_name, include_header, delimiter,
                                    decimal_sep, False, x_data_name=x_data_name)


def _list_to_stream(list_of_spectra, include_group_name=True, include_header=True, delimiter='\t',
                    decimal_sep='.', save_to_file=False, dir_path=None, extension=None,
                    x_data_name='Wavelength / nm'):

    # this generator iterates the ndarray data and yield returns a list of formatted row for csv writer
    def iterate_data(iterable):
        for row in iterable:
            yield [str(num).replace('.', decimal_sep) for num in row]

    if not isinstance(list_of_spectra, list):
        raise ValueError("Argument 'list_of_spectra' must be type of list.")

    buffer = ""

    dialect = csv.excel
    dialect.delimiter = delimiter
    dialect.lineterminator = '\n'
    dialect.quoting = csv.QUOTE_MINIMAL

    for i, node in enumerate(list_of_spectra):

        if save_to_file:
            name = node.name if isinstance(node, Spectrum) else node[0].group_name
            filepath = os.path.join(dir_path,
                                    (name if name != '' else 'Untitled{}'.format(i)) + extension)
            w = csv.writer(open(filepath, 'w', encoding='utf-8'), dialect=dialect)
        else:
            stream = io.StringIO('')
            w = csv.writer(stream, dialect=dialect)

        # export as group
        if isinstance(node, list):
            if len(node) == 0:
                continue

            if include_group_name:
                w.writerow([node[0].group_name])

            # add header if it is user defined
            if include_header:
                w.writerow([x_data_name] + [sp.name for sp in node])

            # add row of wavelength, then, we will transpose the matrix, otherwise, we would have to reshape
            matrix = node[0].data[:, 0]

            # add absorbance data to matrix from all exported spectra
            for sp in node:
                if sp.length() != node[0].length():
                    raise ValueError(
                        f"Spectra '{node[0].name}' and '{sp.name}' in group '{node[0].group_name}'"
                        f" have not the same length (dimension). Unable to export.")
                if sp.data[0, 0] != node[0].data[0, 0] or sp.data[-1, 0] != node[0].data[-1, 0]:
                    raise ValueError(
                        f"Spectra '{node[0].name}' and '{sp.name}' in group '{node[0].group_name}'"
                        f" have not the same length (dimension). Unable to export.")
                matrix = np.vstack((matrix, sp.data[:, 1]))

            matrix = matrix.T  # transpose

            # write matrix
            w.writerows(iterate_data(matrix))

        # export as single spectrum file
        if isinstance(node, Spectrum):
            if include_header:
                w.writerow([x_data_name, node.name])

            w.writerows(iterate_data(node.data))

        if not save_to_file:
            buffer += stream.getvalue() + '\n'
            stream.close()

    if buffer != "":
        # return ret_buffer and remove the 2 new line characters that are at the end
        return buffer[:-2]

