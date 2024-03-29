# standalone version located on my Github https://github.com/dmadea/File-Converters/blob/master/dx2csv-hplc.py

import os
import numpy as np
import struct
import xml.etree.ElementTree as ET
from spectramanipulator.spectrum import Spectrum, SpectrumList


def read_utf16(f):
    """modified string read method which works with Agilent files"""
    # determine length to read
    read_len, = struct.unpack('>B', f.read(1))
    # read values, decode, and strip
    data = f.read(2 * read_len)
    text = data.decode('utf16').strip()

    return text


def _read_data(file, loc, data_scale_factor):
    """Reads the actual data, file is an open binary file, loc is the location of the data in the file,
    data_scale_factor additionally scales the read data matrix."""

    # various shifts from the initial location for different data
    nrec_shift = 375  # number of records, >i
    sample_name_shift = 955  # utf-16 and the first byte is length of the string
    scale_factor_shift = 3182  # scale factor for data, >d, unit is just after this factor
    data_shift = 4193

    space_len = 22  # length of leading bytes before individual spectra, both for FLD and UV data

    # scaling factors
    scale_time = 1 / 60000  # time
    scale_wl = 1 / 20  # wavelength

    # load number of records (it is also specified in XML tree)
    file.seek(loc + nrec_shift)
    nrec, = struct.unpack('>i', file.read(4))

    # load name of sample, not necessary
    file.seek(loc + sample_name_shift)
    sample_name = read_utf16(file)

    # load scaling factor for data
    file.seek(loc + scale_factor_shift)
    scale_fac, = struct.unpack('>d', file.read(8))
    scale_fac *= data_scale_factor
    # load unit of data
    unit = read_utf16(file)

    # load data itself
    file.seek(loc + data_shift)

    times = np.empty(nrec, dtype=np.float64)
    wavelengths = None
    data_mat = None

    for i in range(nrec):
        # in those 22 bytes before the data, there are 4 same bytes for all spectra:
        # e.g. 46 00 EE 07, last two bytes denotes the size of the block, then there
        # are 4 bytes of time the data corresponds to as <i (le int32), then there are
        # 3 x 2 bytes of little endian uint16 that corresponds to start, end and step of
        # the wavelength range, the remaining bytes are the same for all records and IDK
        # what they are
        leading_bytes = file.read(space_len)
        block_size, = struct.unpack('<H', leading_bytes[2:4])
        times[i],  = struct.unpack('<i', leading_bytes[4:8])  # time of measurement
        if wavelengths is None:
            wl_start, wl_end, wl_step = struct.unpack('<HHH', leading_bytes[8:14])
            wavelengths = np.arange(wl_start, wl_end + wl_step, wl_step) * scale_wl
            data_mat = np.empty((nrec, wavelengths.shape[0]), dtype=np.float64)  # create a data matrix for our data

        # if this is not valid for some type of files, the algorithm needs to be rewritten
        # it assumes the block_size is the same for all records, so far it worked...
        assert (block_size - space_len) / 8 == wavelengths.shape[0]

        # read the block of <d (le float64) values and put them into matrix
        data_mat[i, :] = np.frombuffer(file.read(8 * wavelengths.shape[0]), dtype='<d')

    # apply the scale for data
    data_mat *= scale_fac
    times *= scale_time

    return data_mat, times, wavelengths, sample_name, unit


def parse_HPLC_DX_file(fpath):

    with open(fpath, 'rb') as f:
        # read the last 35000 bytes (this should contain all XML tree at the end of DX file)
        block_size = 35000
        f.seek(0, os.SEEK_END)
        f.seek(f.tell() - block_size)

        data = f.read(block_size)

        # find the XML tree
        init = bytes('<ACMD xmlns="urn:schemas-agilent-com:acmd20">', 'utf8')
        end = bytes('</ACMD>', 'utf8')

        start_idx = data.find(init)
        end_idx = data.find(end)

        # get the root of XML tree
        xml_data = data[start_idx:end_idx + len(end)]
        root = ET.fromstring(xml_data)

        ssNs = '{urn:schemas-agilent-com:acmd20}'

        # IDs for location of fluorescence and UV data
        fld_ID = None
        uv_ID = None

        # find the IDs
        for signals in root.iter(ssNs + 'Signal'):
            for item in signals.iter(ssNs + 'Description'):
                if item.text == 'FLD1P,FLD: Spectrum':
                    traceID_el = list(signals.iter(ssNs + "TraceId"))[0]
                    fld_ID = traceID_el.text
                    continue

                if item.text == 'DAD1I,DAD: Spectrum':
                    traceID_el = list(signals.iter(ssNs + "TraceId"))[0]
                    uv_ID = traceID_el.text
                    continue

        spectral_data = []

        # ----------- READ FLD DATA -------

        if fld_ID is not None:
            # find the locations of fluorescence and UV data (locations are after the XML tree)
            fld_id_idx = data.find(bytes(fld_ID + '.UVPK', 'utf8'))
            # 4 bytes before the ID is location in data file
            FLD_loc, = struct.unpack('<I', data[fld_id_idx - 4: fld_id_idx])

            data_mat, times, wavelengths, sample_name, unit = _read_data(f, FLD_loc, 1e-6)  # an extra scaling factor

            group1 = SpectrumList(name=f'Fluorescence - {sample_name}')
            for i in range(wavelengths.shape[0]):
                sp = Spectrum.from_xy_values(times, data_mat[:, i], str(wavelengths[i]))
                group1.children.append(sp)

            spectral_data.append(group1)

        # ----------- READ UV DATA -------

        if uv_ID is not None:
            uv_id_idx = data.find(bytes(uv_ID + '.UVPK', 'utf8'))
            UV_loc, = struct.unpack('<I', data[uv_id_idx - 4: uv_id_idx])

            data_mat, times, wavelengths, sample_name, unit = _read_data(f, UV_loc, 1 / 2000)  # an extra scaling factor

            group2 = SpectrumList(name=f'Absorption - {sample_name}')
            for i in range(wavelengths.shape[0]):
                sp = Spectrum.from_xy_values(times, data_mat[:, i], str(wavelengths[i]))
                group2.children.append(sp)

            spectral_data.append(group2)

        return spectral_data

