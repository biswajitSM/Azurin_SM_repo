import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from Analysis import *
from autocorrelate import autocorrelate

def histogram_on_off_1mol(foldername= foldername, input_potential=[100],
						 pointnumbers=[1], time_lim = [0, 1e5], 
						 bins_on=50, range_on=[0, 0.2], 
						 bins_off=50, range_off=[0, 0.5], 
						 sum_points=10):
    df_datn_emplot, df_FCS, folder = dir_mV_molNo(foldername)
    df_specific = df_datn_emplot[df_datn_emplot['Point number'].isin(pointnumbers)]#keep all the points that exist
    df_specific = df_specific[df_specific['Potential'].isin(input_potential)]; df_specific.reset_index(drop=True, inplace=True)
    f_emplot_path = 'x'; f_datn_path='x'; t_ons=[];t_offs=[]
    if not df_specific.empty:
        f_datn_path = df_specific['filepath[.datn]'].values[0]
        f_emplot_path = df_specific['filepath[.em.plot]'].values[0]
    if os.path.isfile(f_emplot_path):
        out_on_off = t_on_off_fromCP(f_emplot_path, time_lim = time_lim)
        # df_ton, df_toff, average_ton, average_toff, average_ton_err, average_toff_err
        t_ons = np.array(out_on_off[0]);
        t_offs = np.array(out_on_off[1]);
        # defining figure positions
        fig = plt.figure(figsize=(6.3, 8))
        nrows=4; ncols= 2;
        ax00 = plt.subplot2grid((nrows,ncols), (0,0));
        ax01 = plt.subplot2grid((nrows,ncols), (0,1));
        ax1 = plt.subplot2grid((nrows, ncols), (1,0), colspan=2)
        ax2 = plt.subplot2grid((nrows, ncols), (2,0), colspan=2)
        ax30 = plt.subplot2grid((nrows, ncols), (3,0))
        ax31 = plt.subplot2grid((nrows, ncols), (3,1))
        #ax3 = plt.subplot2grid((nrows, ncols), (3,0), colspan=2)

        # waitime_hist_inset(waitimes, axis, bins, binrange, insetrange, fit_func)
        waitime_hist_inset(t_ons, ax00, bins_on, range_on, [0, 0.01], risetime_fit);
        waitime_hist_inset(t_offs, ax01, bins_off, range_off, [0, 0.1], risetime_fit)
        # trace of on and off times as a function of time
        t_av_on, t_av_off, t_abs = trace_on_off_times(ax1, t_ons, t_offs, sum_points,
        										 (-0.01, None), (0, 1))
        # E0 trace from ontraces and off traces
        E0_list = trace_E0(ax2, t_av_on, t_av_off, t_abs, input_potential[0])
        # Autocorrelation of trace of average on and average off times
        G = corr_onoff_av(ax30, t_av_on, t_av_off, [None, 1e2]) 
        # fit of e0 values with a gaussian
        out_E0fit = E0_gaussfit(ax31, E0_list, 40, [-100, 100])

    return(t_av_on, t_av_off, G)

#============on/off time histogram with their fit==============
from mpl_toolkits.axes_grid.inset_locator import inset_axes
def waitime_hist_inset(waitimes, axis, bins, binrange, insetrange, fit_func):
    '''waitimes: list of on-times or off-times
    '''
    n,bins_hist = np.histogram(waitimes, bins=bins, range=binrange)
    t=bins_hist[:-1]; n = n[:];
    t_fit = np.linspace(binrange[0], binrange[1], 1000)
    binwidth = np.mean(np.diff(t))
    if fit_func.__code__.co_code == mono_exp.__code__.co_code:
    	p0 = [10,1.1]
    elif fit_func.__code__.co_code == risetime_fit.__code__.co_code:
    	p0=[10,1.1, 0.1]
    fit, pcov = curve_fit(fit_func, t, n, p0=p0, bounds=(0, np.inf))
    print('k1:'+str(fit[0]))
    #plot as bar
    axis.bar(t, n, width=binwidth, color='k', alpha=0.5)
    axis.plot(t_fit, fit_func(t_fit, *fit), 'k',lw=1,label='k1:'+str(fit[0])+'\n'+str(fit[1]))
    axis.set_xlim(0, None)
    axis.set_ylim(1e0, None)
    axis.set_yticks([])
    axis.set_xlabel('waitimes/s')
    axis.set_ylabel('#')
    axis.set_yscale('log')
    #inset
    axis_in = inset_axes(axis, height="50%", width="50%")
    axis_in.bar(t, n, width=binwidth, color='k', alpha=0.5)
    axis_in.plot(t, n, drawstyle= 'steps-mid', lw=0.5, color='k')
    axis_in.plot(t_fit, fit_func(t_fit, *fit), 'k',lw=1,label='k1:'+str(fit[0])+'\n'+str(fit[1]))
    axis_in.set_xlim(insetrange)
    axis_in.set_ylabel('#')
#==========trace of on off time and potential distribution==================
def trace_on_off_times(axis, t_ons, t_offs, sum_points, on_ylim, off_ylim, plotting=True):
    if len(t_ons)> len(t_offs):
        t_ons = t_ons[:len(t_offs)]
    else:
        t_offs = t_offs[:len(t_ons)]

    t_av_on = []; t_av_off = []; t_abs = [];
    start=0;
    num_outputs = int(len(t_ons)/sum_points);
    for i in range(num_outputs):
        t_av_on_temp = sum(t_ons[start:start+sum_points])/sum_points
        t_av_of_temp = sum(t_offs[start:start+sum_points])/sum_points
        start += sum_points
        t_av_on.append(t_av_on_temp)
        t_av_off.append(t_av_of_temp)
        t_abs_temp = sum(t_ons[:start+sum_points]) + sum(t_offs[:start+sum_points])
        t_abs.append(t_abs_temp)
    t_av_on = pd.Series(t_av_on);
    t_av_off = pd.Series(t_av_off)
    t_on_ratio = t_av_off/t_av_on;
    #plotting
    if plotting == True:
        axis.plot(t_abs, t_av_on, 'b', label='On_av')
        axis.set_ylim(on_ylim)#CAREFUL
        axis.set_xlim(0, None)
        axis.tick_params('y', colors='b')
        axis.set_ylabel('ton_av/s', color='b')
        #axis.set_xticks([])
        axis.legend(loc='center right')
        axis_r = axis.twinx()

        axis_r.plot(t_abs, t_av_off, 'r',label='Off_av')
        axis_r.tick_params('y', colors='r')
        axis_r.set_ylabel('toff_av/s', color='r')
        axis_r.set_ylim(off_ylim)#BE CAREFUL of this limit
        axis_r.set_xlim(0, None)
        #axis_r.set_xlabel('time/s')
        #axis_r.set_xticks([])
        axis_r.legend(loc='center')
    return t_av_on, t_av_off, t_abs# def corr_onoff_av(axis)
#===================trace E0======================
def trace_E0(axis, t_av_on, t_av_off, t_abs, potential):
    t_on_ratio = t_av_off/t_av_on;
    E0_list = potential - 59*log10(t_on_ratio);
    axis.plot(t_abs, E0_list, label='E_0')
    axis.set_xlim(0, None)
    axis.set_ylabel('E_0/mV')
    axis.set_xlabel('time/s')
    return E0_list
#==================gaussian fit to midpooint potentials=============
def E0_gaussfit(axis, E0_list, bins, range_fit):
    bin_centers = linspace(range_fit[0], range_fit[1], bins);
    n, bins_hist, patches = axis.hist(E0_list, bins=bins, range=range_fit)
    y=n; x=bin_centers;
    fit, pcov = curve_fit(gaussian, xdata=x, ydata=y, p0=[100, 5, 25], bounds=(-np.inf, np.inf))
    perr = np.sqrt(np.diag(pcov));
    center=round(fit[1], 1); center_err = round(perr[1], 1);
    fwhm=round(2.3548*fit[2],1); fwhm_err = round(2.3548*perr[2],1);
    #print('Center is %.1f and fwhm is %.1f' %(center, fwhm))
    E0_fitrange = linspace(range_fit[0], range_fit[1], 100);
    axis.plot(E0_fitrange, gaussian(E0_fitrange, *fit), label=str(center)+'+-'+str(center_err)+'\n'+
                                    str(fwhm)+'+-'+str(fwhm_err)+'\n')
    axis.set_xlabel('Midpoint Potential/mV')
    axis.set_ylabel('#')
    axis.set_xlabel(r'$E_0/mV$')
    axis.set_xlim(range_fit[0], range_fit[1])
    axis.legend()
    return center, center_err, fwhm, fwhm_err
#==========Correlation from average on/off times
def corr_onoff_av(axis, t_av_on, t_av_off, xlim):
    G = [];
    try:
        m=(len(t_av_on)/2)-5;
        t_av_tot = average(t_av_on)+average(t_av_off);
        G = autocorrelate(t_av_on[:-1], m=m ,deltat=1,normalize=True);
        axis.plot(G[:,0]*t_av_tot, G[:,1], 'b',label='On time correlation')
        axis.set_xscale('log')
        ymax_lim_on = G[1]
        #off-time correlation-------
        m=(len(t_av_off)/2)-5;
        G = autocorrelate(t_av_off[:-1], m=m ,deltat=1,normalize=True);
        axis.plot(G[:,0]*t_av_tot, G[:,1], 'r',label='Off time correlation')
        axis.set_xscale('log')
        axis.set_xlabel('time/s')
        axis.set_ylabel('G(t)')
        ymax_lim_off = G[1]
        if ymax_lim_on > ymax_lim_off:
            ymax_lim = ymax_lim_on;
        else:
            ymax_lim = ymax_lim_off;
        # axis.set_ylim(-0.01, ymax_lim[1]+0.5*ymax_lim[1])
        # axis.set_xlim(xlim[0], xlim[1])
        axis.set_ylim(-0.01, 0.2)
        axis.set_xlim(None, 100)        
        axis.legend()
    except:
        pass
    return G      
def t_ons_t_offs(df_dig):
    df_ons = df_dig[df_dig['count_rate'] > min(df_dig['count_rate'])];
    df_ons_mins = df_ons.groupby('dig_cp').timestamps.min().values;
    df_ons_maxs = df_ons.groupby('dig_cp').timestamps.max().values;
    t_ons = df_ons_maxs - df_ons_mins;
    t_ons = t_ons[np.nonzero(t_ons)]

    df_offs = df_dig[df_dig['count_rate'] < max(df_dig['count_rate'])];
    df_offs_mins = df_offs.groupby('dig_cp').timestamps.min().values;
    df_offs_maxs = df_offs.groupby('dig_cp').timestamps.max().values;
    t_offs = df_offs_maxs - df_offs_mins;
    t_offs = t_offs[np.nonzero(t_offs)]
    return t_ons, t_offs
#===============fitting functions=============
def risetime_fit(t, k1, k2, A):
    return ((A*k1*k2/(k2-k1)) * (exp(-k1*t) - exp(-k2*t)))
def mono_exp(t, k1, A):
    return A*exp(-k1*t)
def gaussian(x, a, b, c):
    return a*exp((-(x-b)**2)/(2*c**2))    