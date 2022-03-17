# standalone version located on my Github https://github.com/dmadea/File-Converters/blob/master/uv2csv.py

import numpy as np
import struct
from spectramanipulator.spectrum import Spectrum, SpectrumList
from .hplc_dxfileparser import read_utf16
from numba import njit


@njit
def get_array(i2arr: np.ndarray, n_wls: int):
    """

    :param i2arr: np.int16 array of raw data
    :param n_wls: number of wavelengths
    :return:
    """
    ret_array = np.empty(n_wls, dtype=np.float64)
    idx = 0
    value = 0
    for i in range(n_wls):
        mv = i2arr[idx]
        if mv == -32768:
            # reinterpret the next two int16 values as int32
            value = i2arr[idx + 1:idx + 3].view(np.int32)[0]
            idx += 3
        else:
            # cumulative add otherwise
            value += mv
            idx += 1
        ret_array[i] = value
    return ret_array


# adapted from https://github.com/bovee/Aston/blob/master/aston/tracefile/agilent_uv.py
# modified with numba function
def _read_data(file):

    space_len = 22  # length of leading bytes before individual spectra
    scale_wl = 1 / 20  # wavelength

    file.seek(0x35A)
    sample_name = read_utf16(file)

    file.seek(0xC15)
    yunit = read_utf16(file)

    file.seek(0x116)
    nrec, = struct.unpack('>i', file.read(4))  # number of records (time points)

    # read data scale factor
    file.seek(0xc0d)
    scale_fac, = struct.unpack('>d', file.read(8))

    times = np.empty(nrec, dtype=np.float64)
    wavelengths = None
    data_mat = None

    file.seek(0x1000)  # data starts here
    for i in range(nrec):
        leading_bytes = file.read(space_len)
        block_size, = struct.unpack('<H', leading_bytes[2:4])
        times[i], = struct.unpack('<i', leading_bytes[4:8])  # time of measurement

        wl_start, wl_end, wl_step = struct.unpack('<HHH', leading_bytes[8:14])
        if wavelengths is None:
            wavelengths = np.arange(wl_start, wl_end + wl_step, wl_step) * scale_wl
            data_mat = np.empty((nrec, wavelengths.shape[0]), dtype=np.float64)  # create a data matrix for our data
        else:
            assert (wl_end - wl_start) // wl_step + 1 == wavelengths.shape[0], "invalid file or different format"

        i2arr = np.frombuffer(file.read(block_size - space_len), dtype='<i2')

        data_mat[i, :] = get_array(i2arr, wavelengths.shape[0])

        # old working code
        # v = 0
        # for j in range(wavelengths.shape[0]):
        #     ov = struct.unpack('<h', file.read(2))[0]
        #     if ov == -32768:
        #         v = struct.unpack('<i', file.read(4))[0]
        #     else:
        #         v += ov
        #     data_mat[i, j] = v

    data_mat *= scale_fac  # / 2000
    times /= 60000

    return data_mat, times, wavelengths, yunit, sample_name


def parse_HPLC_UV_file(fpath):

    with open(fpath, 'rb') as f:

        data_mat, elution_times, wavelengths, yunit, name = _read_data(f)

        spectral_data = SpectrumList(name=f'Absorption - {name}')
        for i in range(wavelengths.shape[0]):
            sp = Spectrum.from_xy_values(elution_times, data_mat[:, i], str(wavelengths[i]))
            spectral_data.children.append(sp)

        return [spectral_data]

