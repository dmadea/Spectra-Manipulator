# standalone version located on my Github https://github.com/dmadea/File-Converters/blob/master/uv2csv.py

import numpy as np
import struct
from spectramanipulator.spectrum import Spectrum, SpectrumList
from .hplc_dxfileparser import read_utf16


# copied from agilent_uv.py, https://github.com/bovee/Aston
# TODO rewrite, make it faster
def _read_data(file):

    file.seek(0x35A)
    sample_name = read_utf16(file)

    file.seek(0xC15)
    yunit = read_utf16(file)

    file.seek(0x116)
    nscans = struct.unpack('>i', file.read(4))[0]

    # get all wavelengths and times
    wvs = set()
    times = np.empty(nscans)
    npos = 0x1002
    for i in range(nscans):
        file.seek(npos)
        npos += struct.unpack('<H', file.read(2))[0]
        times[i] = struct.unpack('<L', file.read(4))[0]
        nm_srt, nm_end, nm_stp = struct.unpack('<HHH', file.read(6))
        n_wvs = np.arange(nm_srt, nm_end, nm_stp) / 20.
        wvs.update(set(n_wvs).difference(wvs))
    wvs = list(wvs)

    ndata = np.empty((nscans, len(wvs)), dtype="<i4")
    npos = 0x1002

    for i in range(nscans):
        file.seek(npos)
        dlen = struct.unpack('<H', file.read(2))[0]
        npos += dlen
        file.seek(file.tell() + 4)  # skip time
        nm_srt, nm_end, nm_stp = struct.unpack('<HHH', file.read(6))
        file.seek(file.tell() + 8)

        # OLD CODE
        v = 0
        pos = file.tell()
        for wv in np.arange(nm_srt, nm_end, nm_stp) / 20.:
            ov = struct.unpack('<h', file.read(2))[0]
            if ov == -32768:
                v = struct.unpack('<i', file.read(4))[0]
            else:
                v += ov
            ndata[i, wvs.index(wv)] = v
        file.seek(pos)

    return ndata / 2000., times / 60000., np.asarray(wvs), yunit, sample_name


def parse_HPLC_UV_file(fpath):

    with open(fpath, 'rb') as f:

        data_mat, elution_times, wavelengths, yunit, name = _read_data(f)

        spectral_data = SpectrumList(name=f'Absorption - {name}')
        for i in range(wavelengths.shape[0]):
            sp = Spectrum.from_xy_values(elution_times, data_mat[:, i], str(wavelengths[i]))
            spectral_data.children.append(sp)

        return [spectral_data]

