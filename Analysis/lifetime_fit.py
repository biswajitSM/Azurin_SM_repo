import numpy as np
import lmfit
from lmfit import Parameters, report_fit, Model

def lifetime_fitail(time_ns, decay_hist, offset=None, model='biexp'):
    if not offset:
        offset = time_ns[decay_hist == max(decay_hist)][-1]
    #mask for 0 and correction
    mask = np.logical_and(time_ns > offset + 0.2, decay_hist > 0)
    decay_hist_tail = decay_hist[mask]
    time_ns_tail = time_ns[mask]
    #mask for nan

    weights = 1. / np.sqrt(decay_hist_tail)
    if model == 'monoexp':
        params = Parameters()
        params.add('ampl1', value=5e4, min=0)
        params.add('tau1', value=0.8, min=0.1)
        params.add('baseline', value=30)
        fit_res = lmfit.minimize(residuals_monoexp, params,
                                 args=(time_ns_tail, decay_hist_tail, weights),
                                 method='leastsq')
    elif model == 'biexp':
        params = Parameters()
        params.add('ampl1', value=5e4, min=0)
        params.add('tau1', value=0.8, min=0.1)
        params.add('ampl2', value=2e3, min=0)
        params.add('tau2', value=1.8, min=0.1)
        params.add('baseline', value=30)
        fit_res = lmfit.minimize(residuals_biexp, params,
                                 args=(time_ns_tail, decay_hist_tail, weights),
                                 method='leastsq')
    elif model == 'triexp':
        params = Parameters()
        params.add('ampl1', value=5e3, min=0)
        params.add('tau1', value=0.2, min=0)
        params.add('ampl2', value=2e3, min=0)
        params.add('tau2', value=0.8, min=0)
        params.add('ampl3', value=3e3, min=0)
        params.add('tau3', value=1.8, min=0)
        params.add('baseline', value=30)
        fit_res = lmfit.minimize(residuals_triexp, params,
                                 args=(time_ns_tail, decay_hist_tail, weights),
                                 method='leastsq')
    return fit_res, time_ns_tail, decay_hist_tail

def monoexp_lifetime(x, tau1, ampl1, baseline):
    y = ampl1 * np.exp(-(x - min(x)) / tau1)
    y += baseline
    return y

def biexp_lifetime(x, tau1, ampl1, tau2, ampl2, baseline):
    y = ampl1 * np.exp(-(x - min(x)) / tau1) +\
        ampl2 * np.exp(-(x - min(x)) / tau2)
    y += baseline
    return y


def triexp_lifetime(x, tau1, ampl1, tau2, ampl2, tau3, ampl3, baseline):
    y = ampl1 * np.exp(-(x - min(x)) / tau1) +\
        ampl2 * np.exp(-(x - min(x)) / tau2) +\
        ampl3 * np.exp(-(x - min(x)) / tau3)
    y += baseline
    return y

def residuals_monoexp(params, x, y, weights):
    tau1 = params['tau1'].value
    ampl1 = params['ampl1'].value
    baseline = params['baseline'].value
    return (y - monoexp_lifetime(x, tau1, ampl1, baseline)) * weights

def residuals_biexp(params, x, y, weights):
    tau1 = params['tau1'].value
    ampl1 = params['ampl1'].value
    tau2 = params['tau2'].value
    ampl2 = params['ampl2'].value
    baseline = params['baseline'].value
    return (y - biexp_lifetime(x, tau1, ampl1, tau2, ampl2, baseline)) * weights


def residuals_triexp(params, x, y, weights):
    tau1 = params['tau1'].value
    ampl1 = params['ampl1'].value
    tau2 = params['tau2'].value
    ampl2 = params['ampl2'].value
    tau3 = params['tau3'].value
    ampl3 = params['ampl3'].value
    baseline = params['baseline'].value
    return (y - triexp_lifetime(x, tau1, ampl1, tau2, ampl2, tau3, ampl3, baseline)) * weights