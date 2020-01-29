import numpy as np
import matplotlib.pyplot as plt
from spectrum import Spectrum


def calculate_phis(item, c_0_act, c_0_com, act_idxs, com_idxs, phi_act, wl_range=(240, 330), em_source=None):
    sp = item[act_idxs[0]][0]

    idx_0 = Spectrum.find_nearest_idx(sp.data[:, 0], wl_range[0])
    idx_1 = Spectrum.find_nearest_idx(sp.data[:, 0], wl_range[1]) + 1

    # hardcoded number of solutions, this is not general, for other use, please change this number or rewrite the code
    solutions = 81
    wls = idx_1 - idx_0  # number of wavelengths
    wls_mat = np.tile(sp.data[idx_0:idx_1, 0], (solutions, 1))  # define matrix of wavelengths (x values for plotting)
    matrix = np.zeros((solutions, wls))  # define matrix of results, dimensions: 27 x [number of wavelengths]

    # for each wavelength in defined range
    for i in range(wls):
        # get the current wavelength
        wl = sp.data[idx_0 + i, 0]

        # calculate the 27 values at this wavelength
        phis = calculate_phi(item, c_0_act, c_0_com, act_idxs, com_idxs, phi_act, wl=wl, em_source=em_source)

        # fill the i-th column of the matrix with the solution
        matrix[:, i] = np.asarray(phis)

    # plot the matrix, parameter s is size of the circle, alpha sets transparency
    plt.rcParams['figure.figsize'] = [8, 5]
    plt.scatter(wls_mat, matrix, s=30, alpha=0.1)
    plt.ylabel('$\Phi$')
    plt.xlabel('Wavelength / nm')
    plt.show()

    return wls_mat, matrix


def calculate_phi(item, c_0_act, c_0_com, act_idxs, com_idxs, phi_act, wl=285, em_source=None):
    # define list of results
    phis = []

    # for every actinometer set
    for act in act_idxs:
        # for every compound set
        for com in com_idxs:
            # perform calculation between current sets and add the results into list
            phis += calculate_set(item[act], item[com], c_0_act, c_0_com, phi_act, wl, em_source)

    return phis


def calculate_set(act_set, com_set, c_0_act, c_0_com, phi_act, wl=285, em_source=None):
    # assuming that the number spectra in actinometer and compound set is the same
    # thus, len(act_set) == len(com_set), ** both act_set and com_set are instances of SpectrumList!

    # number of items in set
    length = len(act_set)

    # find index in data array of actinometer and compound that corresponds to the choosed wavelength
    idx_act = Spectrum.find_nearest_idx(act_set[0].data[:, 0], wl)
    idx_com = Spectrum.find_nearest_idx(com_set[0].data[:, 0], wl)

    # set absorbances at time zero for both actinometer and compound at wavelength wl
    A0_act = act_set[0].data[idx_act, 1]
    A0_com = com_set[0].data[idx_com, 1]

    # calculate the ratio of c0 and A0 for actinometer and compound
    c0A0_act = c_0_act / A0_act
    c0A0_com = c_0_com / A0_com

    # define list of computed quantum yields
    phis = []

    # for each actinometer difference - index i
    for i in range(1, length):
        # for each compound difference - index j
        for j in range(1, length):
            # calculate change of absorbance between two measurements for act and com
            dA_act = act_set[i - 1].data[idx_act, 1] - act_set[i].data[idx_act, 1]
            dA_com = com_set[j - 1].data[idx_com, 1] - com_set[j].data[idx_com, 1]

            # calculate corresponding change in concentrations
            dc_act = c0A0_act * dA_act
            dc_com = c0A0_com * dA_com

            # calculate the changes in times, use the value in name of the spectrum
            dt_act = float(act_set[i - 1].name) - float(act_set[i].name)
            dt_com = float(com_set[j - 1].name) - float(com_set[j].name)

            # calculate the quantum yield
            phi_com = phi_act * dc_com * dt_act / (dc_act * dt_com)

            # if user provided emission source, calculate the correction
            if em_source is not None:
                # calculate the absorbed light intensity (we use absorption spectrum
                # at the begining of the interval - i-1
                abs_int_act = em_source * (1 - 10 ** (-act_set[i - 1]))
                abs_int_com = em_source * (1 - 10 ** (-com_set[j - 1]))

                # integrate those spectra
                x_abs_act = abs_int_act.integral()
                x_abs_com = abs_int_com.integral()

                # apply the correction
                phi_com *= x_abs_act / x_abs_com

            # add the result to the list
            phis.append(phi_com)

    return phis
