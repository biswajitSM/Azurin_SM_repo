"""
The following functions to compute linear correlation on discrete signals
or on point-processes (e.g. timestamps).
"""
import os
import time
import numpy as np
import scipy as sp
import numba
import pandas as pd
import h5py
import lmfit
from scipy.optimize import curve_fit

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

# ========== fcs of photonhdf5 format  ==========
def fcs_photonhdf5(file_path_hdf5, tmin=None, tmax=None,
                   t_fcsrange=[1e-6, 1], nbins=100,
                   overwrite=False):
    '''
    '''
    file_path_hdf5 = os.path.abspath(file_path_hdf5)
    file_path_hdf5analysis = file_path_hdf5[:-5] + '.analysis.hdf5'
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
    h5_saved = h5py.File(file_path_hdf5analysis, 'r')
    fcs_out = pd.DataFrame(h5_saved['fcs/fcs_nbins100'][:],
                            columns=['lag_time', 'G(t)-1'])
    h5_saved.close()
    return file_path_hdf5analysis, fcs_out

# ============ fitting ========
'''Different equation for fitting
http://www.fcsxpert.com/classroom/theory/autocorrelation-function-list.html
'''

def t_on_off_fromFCS(lag_time, Gn, tmin=1e-5, tmax=1.0e0,
                     signal=3.0e3, bg=2.0e2, bg_corr=False,
                     fitype='mono_exp', plotting=False, ax=None):
    '''
    Argument:
    fitype: 'mono_exp' or 'bi_exp'
    '''
    xdata = lag_time
    mask = np.logical_and(xdata >= tmin, xdata <= tmax)
    xdata = xdata[mask]
    ydata = Gn[mask]
    correction_BG = ((signal + bg) / signal)**2
    if bg_corr:
        ydata = ((ydata) * correction_BG)
    def mono_exp(x, A1, t_ac1):
        return A1*np.exp(-x/t_ac1)
    def bi_exp(x, A1, t1, A2, t2):
        return A1*np.exp(-x/t1) + A2*np.exp(-x/t2)
    if fitype == 'mono_exp':
        monofit, pcov = curve_fit(mono_exp, xdata, ydata,
                                  p0=[1, 1], bounds=(0, np.inf))
        perr = np.sqrt(np.diag(pcov))
        A1 = monofit[0]; A1_err = perr[0]
        t_ac1 = monofit[1]; t_ac1_err = perr[1]
        toff1 = t_ac1*(1+A1); ton1 = t_ac1*(1+(1/A1))# check the formula: A1*t_ac1*(1+A1); ton1 = t_ac1*(1+(1/A1))
        toff1_err = t_ac1_err*(1+A1); ton1_err = t_ac1_err*(1+(1/A1))
        #rounding figures
        Mylist = [ton1, ton1_err, toff1, toff1_err]
        roundMylist = ['%.4f' % elem for elem in Mylist]
        # roundMylist = [ np.round(elem, 3) for elem in Mylist ]
        [ton1, ton1_err, toff1, toff1_err] = roundMylist
        fcs_fit_result = {'A1': A1,
                          'A1_err': A1_err,
                          't_ac1': t_ac1,
                          't_ac1_err': t_ac1_err,
                          'ton1': ton1,
                          'ton1_err': ton1_err,
                          'toff1': toff1,
                          'toff1_err': toff1_err
                         }
    if fitype == 'bi_exp':
        bifit, pcov = curve_fit(bi_exp, xdata, ydata,
                                p0=[1, 1, 1, 1], bounds=(0, np.inf))
        perr = np.sqrt(np.diag(pcov))
        if bifit[1] > bifit[3]:
            A1 = bifit[0]; t_ac1 = bifit[1]; t_ac1_err = perr[1]
            A2 = bifit[2]; t_ac2 = bifit[3]; t_ac2_err = perr[3]
        else:
            A1 = bifit[2]; t_ac1 = bifit[3]; t_ac1_err = perr[3]
            A2 = bifit[0]; t_ac2 = bifit[1]; t_ac2_err = perr[1]
        toff1 = t_ac1*(1+A1); ton1 = t_ac1*(1+(1/A1))
        toff1_err = t_ac1_err*(1+A1); ton1_err = t_ac1_err*(1+(1/A1))
        toff2 = t_ac2*(1+A2); ton2 = t_ac2*(1+(1/A2))
        toff2_err = t_ac2_err*(1+A2); ton2_err = t_ac2_err*(1+(1/A2))
        #rounding figures
        Mylist = [ton1, ton1_err, toff1, toff1_err,
                  ton2, ton2_err, toff2, toff2_err]
        roundMylist = ['%.4f' % elem for elem in Mylist]
        # roundMylist = [np.round(elem, 3) for elem in Mylist]
        [ton1, ton1_err, toff1, toff1_err,
         ton2, ton2_err, toff2, toff2_err] = roundMylist
        fcs_fit_result = {'ton1': ton1,
                          'ton1_err': ton1_err,
                          'toff1': toff1,
                          'toff1_err': toff1_err,
                          'ton2': ton2,
                          'ton2_err': ton2_err,
                          'toff2': toff2,
                          'toff2_err': toff2_err
                         }
    if plotting and ax:
        ax.plot(xdata, ydata, label='data')
        if fitype == 'mono_exp':
            ax.plot(xdata, mono_exp(xdata, *monofit),
                    color='r', linewidth=2.0)
        if fitype == 'bi_exp':
            ax.plot(xdata, bi_exp(xdata, *bifit),
                    color='r', linewidth=2.0)
        ax.set_xscale('log')
        ax.grid(True); ax.grid(True, which='minor', lw=0.3)
        ax.set_xlim(tmin, tmax)
        ax.set_ylim(0, None)
        ax.legend()
        # ax.set_title(fcs_fit_result)
    return fcs_fit_result

def t_on_off_fromFCS_2(lag_time, Gn, tmin=1e-5, tmax=1.0e0,
                       signal=3.0e3, bg=2.0e2, bg_corr=True,
                       fitype='mono_exp', plotting=False, ax=None):
    '''
    Argument:
    fitype: 'mono_exp' or 'bi_exp'
    '''
    xdata = lag_time
    mask = np.logical_and(xdata >= tmin, xdata <= tmax)
    xdata = xdata[mask]
    ydata = Gn[mask]
    correction_BG = ((signal + bg) / signal)**2
    if bg_corr:
        ydata = ((ydata) * correction_BG)

    def mono_exp(x, ton1, toff1):
        #A*np.exp(-x/t_ac)
        return (toff1 / ton1) * np.exp(-x * (ton1 + toff1) / (ton1 * toff1))
    def bi_exp(x, ton1, toff1, ton2, toff2):
        g1 = (toff1 / ton1) * np.exp(-x * (ton1 + toff1) / (ton1 * toff1))
        g2 = (toff2 / ton2) * np.exp(-x * (ton2 + toff2) / (ton2 * toff2))
        return g1 * g2
    from lmfit import Model
    if fitype == 'mono_exp':
        gmodel = Model(mono_exp)
        gmodel.set_param_hint('ton1', value=0.01)  # , value=1, min=0.05, max=100
        gmodel.set_param_hint('toff1', value=0.05)  # , value=0.005, min=1, max=100
        pars = gmodel.make_params()
        result = gmodel.fit(ydata, pars, x=xdata)  # , A=1, B=1, t_ac=1
        params = result.params
        ton1 = params['ton1'].value
        ton1_err = float(str(params['ton1']).split('+/-')[1].split(',')[0])
        toff1 = params['toff1'].value
        toff1_err = float(str(params['toff1']).split('+/-')[1].split(',')[0])
        #rounding figures
        Mylist = [ton1, ton1_err, toff1, toff1_err]
        roundMylist = [np.round(elem, 3) for elem in Mylist]
        [ton1, ton1_err, toff1, toff1_err] = roundMylist
        fcs_fit_result = {'ton1': ton1,
                          'ton1_err': ton1_err,
                          'toff1': toff1,
                          'toff1_err': toff1_err}
    if fitype == 'bi_exp':
        gmodel = Model(bi_exp)
        gmodel.set_param_hint('ton1', value=0.1)  # , value=1, min=0.05, max=100
        gmodel.set_param_hint('toff1', value=0.2)  # , value=0.005, min=1, max=100
        gmodel.set_param_hint('ton2', value=0.2)  # , value=1, min=0.05, max=100
        gmodel.set_param_hint('toff2', value=0.3)  # , value=0.005, min=1, max=100
        pars = gmodel.make_params()
        result = gmodel.fit(ydata, pars, x=xdata)  # , A=1, B=1, t_ac=1
        params = result.params
        ton1 = params['ton1'].value
        ton1_err = float(str(params['ton1']).split('+/-')[1].split(',')[0])
        toff1 = params['toff1'].value
        toff1_err = float(str(params['toff1']).split('+/-')[1].split(',')[0])
        #rounding figures
        Mylist = [ton1, ton1_err, toff1, toff1_err]
        roundMylist = [np.round(elem, 3) for elem in Mylist]
        [ton1, ton1_err, toff1, toff1_err] = roundMylist
        fcs_fit_result = {'ton1': ton1,
                          'ton1_err': ton1_err,
                          'toff1': toff1,
                          'toff1_err': toff1_err}
    if plotting and ax:
        print('plot')
        ax.plot(xdata, ydata, 'b.', label='data')
        ax.plot(xdata, result.best_fit, 'r--', label='fit')
        ax.set_xscale('log')
        ax.grid(True); ax.grid(True, which='minor', lw=0.3)
        ax.set_xlim(tmin, tmax)
        ax.set_ylim(0, None)
        ax.legend()
        ax.set_title(fcs_fit_result)
    return fcs_fit_result, result  # , t_on_err, t_off_err


class fcsFitClass(object):
    """docstring for fcsFit"""
    parameters = {"TimeLimit": [None, None],
                  "BackgroundCorrection":False,
                  "Signal": 1e4,
                  "Background":2e2,
                  "widthXY": 0.237*1e-6,
                  "widthXZ": 0.65*1e-6,
                 }
    def __init__(self, lagTime, G, parameters=parameters):
        for key in parameters:
            setattr(self, key, parameters[key])
        self.lagTime = lagTime
        self.G = G
        # update min and max time limit
        if self.TimeLimit[0] is None:
            self.TimeLimit[0] = min(self.lagTime)
        if self.TimeLimit[1] is None:
            self.TimeLimit[1] = max(self.lagTime)
        mask = np.logical_and(self.lagTime > self.TimeLimit[0],
                              self.lagTime < self.TimeLimit[1])
        self.lagTime = self.lagTime[mask]
        self.G = self.G[mask]

    def CorrectForBackground(self):
        correctionFactor = ((self.Signal + self.Background) / self.Signal)**2
        self.G = ((self.G) * correctionFactor)
    def fitDiffusion3D(self, C_init=10e-9, tD_init=1e-4):
        self.ModelForFit = lmfit.Model(diffusion3D)
        self.params = lmfit.Parameters()
        self.params.add('widthXY', self.widthXY, vary=True, min=1e-7)
        self.params.add('widthXZ', self.widthXZ, vary=True, min=1e-7)
        self.params.add('C', C_init, min=0)
        self.params.add('tD', tD_init, min=0)
        self.fitResult = self.ModelForFit.fit(self.G, self.params,
                                              x=self.lagTime)
    def fitMonoExp(self, A1_init=1, t_ac1_init=0.1):
        self.ModelForFit = lmfit.Model(monoExp)
        self.params = lmfit.Parameters()
        self.params.add('A1', A1_init, min=0)
        self.params.add('t_ac1', t_ac1_init, min=0)
        self.fitResult = self.ModelForFit.fit(self.G, self.params,
                                              x=self.lagTime)
    def fitBiExp(self, A1_init=1, t1_init=0.1,
                 A2_init=0.5, t2_init=0.01):
        self.ModelForFit = lmfit.Model(biExp)
        self.params = lmfit.Parameters()
        self.params.add('A1', A1_init, min=0)
        self.params.add('t1', t1_init, min=0)
        self.params.add('A2', A2_init, min=0)
        self.params.add('t2', t2_init, min=0)
        self.fitResult = self.ModelForFit.fit(self.G, self.params,
                                              x=self.lagTime)        

def diffusion3D(x, widthXY, widthXZ, C, tD):
    K = widthXZ / widthXY
    Veff = (np.pi**(3/2)) * (widthXY**2) * widthXZ * 1e3 # convert to liter
    return (1/(Veff*C*6.022140857e+23)) * (1+x/tD)**-1 * (1+x/(tD*(K**2)))**-1
def monoExp(x, A1, t_ac1):
    return (A1*np.exp(-x/t_ac1))
def biExp(x, A1, t1, A2, t2):
    return A1*np.exp(-x/t1) + A2*np.exp(-x/t2)
