"""
The following functions to compute linear correlation on discrete signals
or on point-processes (e.g. timestamps).
"""
import os
import time
import numpy as np
import numba
import pandas as pd
import h5py

@numba.jit(nopython=True)
def pcorrelate(t, u, bins):
    """Compute correlation of two arrays of descrete events (point-process).

    The input arrays need to be values of a point process, such as
    photon arrival times or positions. The correlation is efficiently
    computed on an arbitrary array of lag-bins. For example bins can be
    uniformly spaced in log-space and span several orders of magnitudes.
    This function implements the algorithm described in
    `(Laurence 2006) <https://doi.org/10.1364/OL.31.000829>`__.

    Arguments:
        t (array): first array of "points" to correlate. The array needs
            to be monothonically increasing.
        u (array): second array of "points" to correlate. The array needs
            to be monothonically increasing.
        bins (array): bin edges for lags where correlation is computed.

    Returns:
        Array containing the correlation of `t` and `u`.    
        The size is `len(bins) - 1`.

    See also:
        :func:`make_loglags` to genetate log-spaced lag bins.
    """
    nbins = len(bins) - 1

    # Array of counts (histogram)
    Y = np.zeros(nbins, dtype=np.int64)

    # For each bins, imin is the index of first `u` >= of each left bin edge
    imin = np.zeros(nbins, dtype=np.int64)
    # For each bins, imax is the index of first `u` >= of each right bin edge
    imax = np.zeros(nbins, dtype=np.int64)

    # For each ti, perform binning of (u - ti) and accumulate counts in Y
    for ti in t:
        for k, (tau_min, tau_max) in enumerate(zip(bins[:-1], bins[1:])):
            #print ('\nbin %d' % k)

            if k == 0:
                j = imin[k]
                # We start by finding the index of the first `u` element
                # which is >= of the first bin edge `tau_min`
                while j < len(u):
                    if u[j] - ti >= tau_min:
                        break
                    j += 1
            imin[k] = j
            if imax[k] > j:
                j = imax[k]
            while j < len(u):
                if u[j] - ti >= tau_max:
                    break
                j += 1
            imax[k] = j
            # Now j is the index of the first `u` element >= of
            # the next bin left edge
        Y += imax - imin
    return Y / np.diff(bins)


@numba.jit
def ucorrelate(t, u, maxlags=None):
    """Compute correlation of two signals defined at uniformly-spaced points.

    The correlation is defined only for positive lags (including zero).
    The input arrays represent signals defined at uniformily-spaced
    points. This function is equivalent to :func:`numpy.correlate`, but can
    efficiently compute correlations on a limited number of lags.

    Note that binning point-processes with uniform bins, provides
    signals that can be passed as argument to this function.

    Arguments:
        tx (array): first signal to be correlated
        ux (array): second signal to be correlated
        maxlags (int): number of lags wher correlation is computed.
            If None, computes all the lags where signals overlap
            `min(tx.size, tu.size) - 1`.

    Returns:
        Array contained the correlation at different lags.
        The size of this array is `maxlags` (if defined) or
        `min(tx.size, tu.size) - 1`.

    Example:

        Correlation of two signals `t` and `u`::

            >>> t = np.array([1, 2, 0, 0])
            >>> u = np.array([0, 1, 1])
            >>> pycorrelate.ucorrelate(t, u)
            array([2, 3, 0])

        The same result can be obtained with numpy swapping `t` and `u` and
        restricting the results only to positive lags::

            >>> np.correlate(u, t, mode='full')[t.size - 1:]
            array([2, 3, 0])
    """
    if maxlags is None:
        maxlags = u.size
    maxlags = int(min(u.size, maxlags))
    C = np.zeros(maxlags, dtype=np.int64)
    for lag in range(maxlags):
        tmax = min(u.size - lag, t.size)
        umax = min(u.size, t.size + lag)
        C[lag] = (t[:tmax] * u[lag:umax]).sum()
    return C


def make_loglags(exp_min, exp_max, points_per_base, base=10):
    """Make a log-spaced array useful as lag bins for cross-correlation.

    This function conveniently creates an arrays on lag-bins to be used
    with :func:`pcorrelate`.

    Arguments:
        exp_min (int): exponent of the minimum value
        exp_max (int): exponent of the maximum value
        points_per_base (int): number of points per base
            (i.e. in a decade when `base = 10`)
        base (int): base of the exponent. Default 10.

    Returns:
        Array of log-spaced values with specified range and spacing.

    Example:

        Compute log10-spaced bins with 2 bins per decade, starting
        from 10^-1 and stopping at 10^3::

            >>> make_loglags(-1, 3, 2)
            array([  1.00000000e-01,   3.16227766e-01,   1.00000000e+00,
                     3.16227766e+00,   1.00000000e+01,   3.16227766e+01,
                     1.00000000e+02,   3.16227766e+02,   1.00000000e+03])

    See also:
        :func:`pcorrelate`
    """
    num_points = points_per_base * (exp_max - exp_min) + 1
    bins = np.logspace(exp_min, exp_max, num_points, base=base)
    return bins
def normalize_G(t, u, bins):
    """Normalize ACF and CCF.
    """
    G = pcorrelate(t, u, bins)    
    duration = max((t.max(), u.max())) - min((t.min(), u.min()))
    Gn = G.copy()
    for i, tau in enumerate(bins[1:]):
        Gn[i] *= ((duration - tau) 
                  / (float((t >= tau).sum()) * 
                     float((u <= (u.max() - tau)).sum())))
    return Gn

def fcs_photonhdf5(file_path_hdf5, tmin=None, tmax=None,
                   t_fcsrange=[1e-6, 1], nbins=10,
                   overwrite=False):
    '''
    '''
    file_path_hdf5 = os.path.abspath(file_path_hdf5)
    file_path_hdf5analysis = file_path_hdf5[:-5] + '_analysis.hdf5'
    # check if output hdf5 file exist, else create one
    if not os.path.isfile(file_path_hdf5analysis):
        h5_analysis = h5py.File(file_path_hdf5analysis, 'w')
    else:
        h5_analysis = h5py.File(file_path_hdf5analysis, 'r+')
    # check if changepoint group exist, else create one
    grp_fcs = 'fcs'
    if not '/' + grp_fcs in h5_analysis.keys():
        h5_analysis.create_group(grp_fcs)
    # Read and extract time stamps from photonhdf4 file
    h5 = h5py.File(file_path_hdf5, 'r')
    unit = h5['photon_data']['timestamps_specs']['timestamps_unit'].value
    timestamps = unit * h5['photon_data']['timestamps'][...]
    if not tmin:
        tmin = min(timestamps)
    if not tmax:
        tmax = max(timestamps)
    mask = np.logical_and(timestamps >= tmin, timestamps <= tmax)
    timestamps = timestamps[mask]
    h5.close()
    # check if  exist
    data_fpars = '/' + grp_fcs + '/fcs_nbins' + str(nbins)
    if data_fpars in h5_analysis.keys() and overwrite:
        print('already exists and deleting for new analysis')
        del h5_analysis[data_fpars]
    if not data_fpars in h5_analysis.keys() or overwrite:
        bins = make_loglags(np.log10(t_fcsrange[0]),
                            np.log10(t_fcsrange[1]), nbins)
        Gn = normalize_G(timestamps, timestamps, bins)
        Gn = np.hstack((Gn[:1], Gn)) - 1
        df_fcs = pd.DataFrame({'lag time': bins,
                               'nGn': Gn})
        h5_analysis[data_fpars] = df_fcs
        h5_analysis[data_fpars].attrs['tmin'] = tmin
        h5_analysis[data_fpars].attrs['tmax'] = tmax
        cols = 'lag time, G(t)-1'
        h5_analysis[data_fpars].attrs['cols'] = cols
        h5_analysis[data_fpars].attrs['bins per one order of time'] = nbins
        h5_analysis.flush()
    h5_analysis.close()
    return
# =========== FOLDERWISE ==============


def fcs_folderwise(folderpath, t_fcsrange=[1e-6, 1], nbins=100, overwrite=False):
    start_time = time.time()
    pt3_extension = [".pt3"]
    for dirpath, dirname, filenames in os.walk(folderpath):
        for filename in [f for f in filenames if f.endswith(tuple(pt3_extension))]:
            file_path_pt3 = os.path.join(dirpath, filename)
            file_path_hdf5 = file_path_pt3[:-3] + 'hdf5'
            file_path_datn = file_path_hdf5[:-4] + 'pt3.datn'
            if os.path.isfile(file_path_datn):
                start_time_i = time.time()
                print("---%.1f : fcs calculation started for %s\n" %
                      (start_time_i, file_path_hdf5))
                try:
                    df_datn = pd.read_csv(file_path_datn, header=None)
                    tmin = min(df_datn[0])
                    tmax = max(df_datn[0])
                    fcs_photonhdf5(file_path_hdf5, tmin=tmin, tmax=tmax,
                                   t_fcsrange=t_fcsrange, nbins=nbins,
                                   overwrite=overwrite)
                except:
                    fcs_photonhdf5(file_path_hdf5, tmin=None, tmax=None,
                                   t_fcsrange=t_fcsrange, nbins=nbins,
                                   overwrite=overwrite)
                processtime = time.time() - start_time_i
                print("---TOTAL time took for the file: %s IS: %s seconds ---\n" %
                      (file_path_hdf5, processtime))
            else:
                print(file_path_datn + ' : doesnot exist\n')
    print("---TOTAL time took for the folder: %s seconds ---\n" %
          (time.time() - start_time))
    return

# # run for a folder , remember it can take very long time; Create a temp file and run them in section
temp_data = '/home/biswajit/Research/Reports_ppt/reports/AzurinSM-MS4/data/S106d18May17_635_CuAzu655_longtime/S106d18May17_60.5_635_A9_CuAzu655_100mV(18)/data'
data_path = '/home/biswajit/Research/Reports_ppt/reports/AzurinSM-MS4/data'
fcs_folderwise(data_path, t_fcsrange=[1e-6, 10], nbins=100, overwrite=True)