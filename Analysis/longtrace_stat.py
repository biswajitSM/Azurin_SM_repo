import os
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import datetime

from Analysis import *
from autocorrelate import autocorrelate
from pycorrelate import *

def histogram_on_off_1mol(file_path_hdf5, time_lim = [0, 1e3], input_potential=100,
                          bins_on=50, range_on=[0, 0.2], 
                          bins_off=50, range_off=[0, 0.5], 
                          E0range = [-100, 200], sum_points=10,
                          G_lim=[0, 0.5], tlag_lim=[0, 100]):
    df = onoff_values(file_path_hdf5, time_lim=(0, 2.5e3))
    df_mean = df.groupby(np.arange(len(df))//sum_points).mean().dropna().reset_index(drop=True)
    df_roll_mean = df.rolling(sum_points).mean().dropna().reset_index(drop=True)    
    df_select = df_roll_mean
    # df_ton, df_toff, average_ton, average_toff, average_ton_err, average_toff_err
    # defining ====FIGURE-1======
    plt.close('all')
    fig1 = plt.figure(figsize=(8, 8))
    nrows=4; ncols= 4;
    # ax00 = plt.subplot2grid((nrows,ncols), (0,0));
    ax01 = plt.subplot2grid((nrows,ncols), (0,1), colspan=3);
    ax10 = plt.subplot2grid((nrows,ncols), (1,0));
    ax11 = plt.subplot2grid((nrows, ncols), (1,1), colspan=3);
    ax20 = plt.subplot2grid((nrows,ncols), (2,0));
    ax21 = plt.subplot2grid((nrows, ncols), (2,1), colspan=3)
    ax30 = plt.subplot2grid((nrows,ncols), (3,0));
    ax31 = plt.subplot2grid((nrows, ncols), (3,1), colspan=3)
    # real plotting starts here
    timetrace_real(ax01, file_path_hdf5, time_lim, y_lim_max = 7e3, bintime= 5e-3);
    waitime_hist_inset(df['ontimes'], ax10, bins_on, range_on, [0, 0.02], risetime_fit,xlabel='t_ons');
    ax11.plot(df_select['abstime_on'],df_select['ontimes'], 'b', label='Bright times')
    ax11.set_xlim(time_lim)
    waitime_hist_inset(df['offtimes'], ax20, bins_off, range_off, [0, 0.035], risetime_fit, xlabel='t_off/s')
    ax21.plot(df_select['abstime_off'],df_select['offtimes'], 'r', label='Dark times')
    ax21.set_xlim(time_lim)    
    E0_list = trace_E0(ax31, df_select['ontimes'], df_select['offtimes'],
                       df_select['abstime_on'], input_potential, E0range, time_lim)
    out_E0fit = E0_gaussfit(ax30, E0_list, 50, E0range);
    #defining =====FIGURE-2=====
    fig2 = plt.figure(figsize=(6.3, 6.3))
    nrows=2; ncols= 2;
    ax00 = plt.subplot2grid((nrows,ncols), (0,0));
    ax01 = plt.subplot2grid((nrows, ncols), (0,1))
    ax10 = plt.subplot2grid((nrows, ncols), (1,0))
    ax11 = plt.subplot2grid((nrows, ncols), (1,1))
    # waitime_hist_inset(waitimes, axis, bins, binrange, insetrange, fit_func)
    waitime_hist_inset(df['ontimes'], ax00, bins_on, range_on, [0, 0.02], risetime_fit);
    waitime_hist_inset(df['offtimes'], ax01, bins_off, range_off, [0, 0.035], risetime_fit, xlabel='t_off/s')
    # Autocorrelation of trace of average on and average off times
    G = corr_onoff_av(ax10, df_select['ontimes'], df['offtimes'], tlag_lim, G_lim) #averaged
    # fit of e0 values with a gaussian
    out_E0fit = E0_gaussfit(ax11, E0_list, 40, E0range)
    return fig1, fig2
    
def plot_tt_zoomin(foldername= foldername, input_potential=[100],
                    pointnumbers=[1], time_lim_1 = [0, 1e2], time_lim_2 = [2e2, 3e2], bintime=5e-3):
    df_datn_emplot, df_FCS, folder = dir_mV_molNo(foldername)
    df_specific = df_datn_emplot[df_datn_emplot['Point number'].isin(pointnumbers)]#keep all the points that exist
    df_specific = df_specific[df_specific['Potential'].isin(input_potential)]; df_specific.reset_index(drop=True, inplace=True)
    f_emplot_path = 'x'; f_datn_path='x'; t_ons=[];t_offs=[]
    if not df_specific.empty:
        f_datn_path = df_specific['filepath[.datn]'].values[0]
        f_emplot_path = df_specific['filepath[.em.plot]'].values[0]
        f_hdf5 = df_specific['filepath[.hdf5]'].values[0]
    if os.path.isfile(f_emplot_path):
        plt.close('all')
        fig1 = plt.figure(figsize=(8, 4))
        nrows=2; ncols= 2;
        ax00 = plt.subplot2grid((nrows,ncols), (0,0), colspan=3);
        ax10 = plt.subplot2grid((nrows,ncols), (1,0));
        ax11 = plt.subplot2grid((nrows,ncols), (1,1));
        timetrace_real(ax00, f_hdf5, f_emplot_path, time_lim=[0, 350],
                        y_lim_max = 7e3, bintime= bintime);
        timetrace_real(ax10, f_hdf5, f_emplot_path, time_lim=time_lim_1,
                        y_lim_max = 7e3, bintime= bintime);
        timetrace_real(ax11, f_hdf5, f_emplot_path, time_lim=time_lim_2,
                        y_lim_max = 7e3, bintime= bintime);                                
    return        
#=================plot time trace Original======================
def timetrace_real(axis, file_path_hdf5, time_lim, y_lim_max, bintime):
    #Real trace
    h5 = h5py.File(file_path_hdf5);
    unit = h5['photon_data']['timestamps_specs']['timestamps_unit'][...]
    df = unit * h5['photon_data']['timestamps'][...]
    h5.close()
    tt_length=max(df)-min(df);
    # tt_length = tt_length, 0);
    binpts=int(tt_length/bintime);
    df_hist = np.histogram(df, bins=binpts,range=(min(df), max(df)));
    axis.plot(df_hist[1][:-1], df_hist[0]/bintime, 'b')#original data
    axis.set_xlim(time_lim);
    axis.set_xticks([])
    axis.set_ylim(0, y_lim_max)# 1.5*max(df[1]/1000)
    return
#============on/off time histogram with their fit==============
from mpl_toolkits.axes_grid.inset_locator import inset_axes
def waitime_hist_inset(waitimes, axis, bins, binrange, insetrange, fit_func, xlabel='t_on/s'):
    '''waitimes: list of on-times or off-times
    '''
    n,bins_hist = np.histogram(waitimes, bins=bins, range=(min(waitimes) , binrange[1]))#avoiding zero
    t=bins_hist[:-1]; n = n[:];
    t_fit = np.linspace(min(waitimes), binrange[1], 1000)
    binwidth = np.mean(np.diff(t))
    #fit
    if fit_func.__code__.co_code == mono_exp.__code__.co_code:
    	p0 = [10,1.1]
    elif fit_func.__code__.co_code == risetime_fit.__code__.co_code:
    	p0=[10,1.1, 0.1]
    elif fit_func.__code__.co_code == streched_exp.__code__.co_code:
        p0=[2,0.8, 100]        
    fit, pcov = curve_fit(fit_func, t, n, p0=p0, bounds=(0, np.inf))
    print('k1:'+str(fit[0]))
    print(fit)
    #fit streched
    # fit_str, pcov_str = curve_fit(streched_exp, t[5:], n[5:], p0=[10, 0.3, 100], bounds=(-np.inf, np.inf))
    #plot as bar
    from matplotlib import pyplot, transforms
    rot = transforms.Affine2D().rotate_deg(90)
    axis.bar(t, n, width=binwidth, color='k', alpha=0.5)
    axis.plot(t_fit, fit_func(t_fit, *fit), 'k',lw=1,label='k1:'+str(fit[0])+'\n'+str(fit[1]))
    # axis.plot(t_fit, streched_exp(t_fit, *fit_str), '--k',lw=1,label='k1:'+str(fit_str[0])+'\nb:'+str(fit_str[1]))
    axis.set_xlim(0, None)
    axis.set_ylim(1e0, None)
    axis.set_yscale('log')
    # axis.get_yaxis().set_ticklabels([])
    # axis.get_xaxis().set_ticklabels([])
    # axis.set_xlabel(xlabel)
    # axis.set_ylabel('#')
    #inset
    axis_in = inset_axes(axis, height="50%", width="50%")
    axis_in.bar(t, n, width=binwidth, color='k', alpha=0.5)
    axis_in.plot(t, n, drawstyle= 'steps-mid', lw=0.5, color='k')
    axis_in.plot(t_fit, fit_func(t_fit, *fit), 'k',lw=1,label='k1:'+str(fit[0])+'\n'+str(fit[1]))
    axis_in.set_xlim(insetrange)
    axis_in.get_yaxis().set_ticklabels([])
#==========trace of on off time and potential distribution==================
def durations_plot(axis, abstime, duration, _ylim, time_lim):
    axis.plot(abstime, duration, 'b', label='On_av')
    axis.set_ylim(on_ylim)#CAREFUL
    # axis.set_xlim(time_lim)
    axis.tick_params('y', colors='b')
    axis.set_ylabel('ton_av/s', color='b')
    axis.set_xticks([])
    axis.legend(loc='center right')
    axis.yaxis.set_label_position("right")
    axis.yaxis.tick_right()
    axis.set_xlim(min(t_abs), max(t_abs))
    return t_av_on, t_av_off, t_abs# def corr_onoff_av(axis)
#===================trace E0======================
def trace_E0(axis, t_av_on, t_av_off, t_abs, potential, E0range, time_lim):
    t_on_ratio = t_av_off/t_av_on;
    E0_list = potential - 59*log10(t_on_ratio);
    axis.plot(t_abs, E0_list, label='E_0')
    axis.set_xlim(time_lim)
    axis.set_ylim(E0range)
    axis.set_ylabel('E_0/mV')
    axis.set_xlabel('time/s')
    axis.yaxis.set_label_position("right")
    axis.yaxis.tick_right()
    return E0_list
#==================gaussian fit to midpooint potentials=============
def E0_gaussfit(axis, E0_list, bins, E0range):
    bin_centers = linspace(E0range[0], E0range[1], bins);
    n, bins_hist, patches = axis.hist(E0_list, bins=bins, range=E0range)
    y=n; x=bin_centers;
    fit, pcov = curve_fit(gaussian, xdata=x, ydata=y, p0=[100, 5, 25], bounds=(-np.inf, np.inf))
    perr = np.sqrt(np.diag(pcov));
    center=round(fit[1], 1); center_err = round(perr[1], 1);
    fwhm=round(2.3548*fit[2],1); fwhm_err = round(2.3548*perr[2],1);
    #print('Center is %.1f and fwhm is %.1f' %(center, fwhm))
    E0_fitrange = linspace(E0range[0], E0range[1], 100);
    axis.plot(E0_fitrange, gaussian(E0_fitrange, *fit), label=str(center)+'+-'+str(center_err)+'\n'+
                                    str(fwhm)+'+-'+str(fwhm_err)+'\n')
    axis.set_xlabel('Midpoint Potential/mV')
    axis.set_ylabel('#')
    axis.set_xlabel(r'$E_0/mV$')
    axis.set_xlim(E0range[0], E0range[1]);
    axis.set_xticks([])
    axis.set_yticks([])        
    axis.legend()
    return center, center_err, fwhm, fwhm_err
#==========Correlation from average on/off times (G)===========
def corr_onoff_av(axis, t_av_on, t_av_off, tlag_lim, G_lim):
    G = []; ymax_lim=0.2;
    try:
        m=(len(t_av_on)/5)-5;
        t_av_tot = average(t_av_on)+average(t_av_off);
        G = autocorrelate(t_av_on[:-1], m=m ,deltat=1,normalize=True);
        axis.plot(G[:,0]*t_av_tot, G[:,1], 'b',label='On time correlation')
        ymax_lim_on = G[0,1]
        #off-time correlation-------
        m=(len(t_av_off)/2)-5;
        G = autocorrelate(t_av_off[:-1], m=m ,deltat=1,normalize=True);
        axis.plot(G[:,0]*t_av_tot, G[:,1], 'r',label='Off time correlation')
        axis.set_xscale('log')
        axis.set_xlabel('time/s')
        axis.set_ylabel('G(t)')
        ymax_lim_off = G[0,1]
        if ymax_lim_on > ymax_lim_off:
            ymax_lim = ymax_lim_on;
        else:
            ymax_lim = ymax_lim_off;
        axis.legend()
    except:
        pass
    axis.set_ylim(0, ymax_lim)#+0.01*ymax_lim
    axis.set_ylim(G_lim)#+0.01*ymax_lim
    axis.set_xlim(tlag_lim)       
    return G      
def onoff_values(file_path_hdf5, time_lim=(0, 1e5)):
    df_dig = digitize_photonstamps(file_path_hdf5, pars=(1, 0.1, 0.9, 2),
                                   time_sect=100, time_lim=time_lim, bintime=5e-3,
                                   int_photon=False, nano_time=False, duration_cp=False)
    df_dig = df_dig[df_dig['timestamps'] < time_lim[1]]
    df_dig = df_dig[df_dig['timestamps'] > time_lim[0]]
    df_on = df_dig[df_dig['state']==2].reset_index(drop=True)
    df_off = df_dig[df_dig['state']==1].reset_index(drop=True)
    time_left = df_on.groupby('cp_no').timestamps.min();
    time_right = df_on.groupby('cp_no').timestamps.max();
    abstime_on = df_on.groupby('cp_no').timestamps.mean();
    ontimes = time_right - time_left
    
    time_left = df_off.groupby('cp_no').timestamps.min();
    time_right = df_off.groupby('cp_no').timestamps.max();    
    abstime_off = df_off.groupby('cp_no').timestamps.mean();
    offtimes = time_right - time_left
    l = len(abstime_on) - 2
    df = pd.DataFrame()
    df['abstime_on'] = abstime_on.values[:l]
    df['ontimes'] = ontimes.values[:l]
    df['abstime_off'] = abstime_off.values[:l]
    df['offtimes'] = offtimes.values[:l]
    return df
# ===============================================
# ============= Changepoint free ===============
# ===============================================
def longtrace_byparts(timestamps, nanotimes, save_folder,
                      window=1e4, period=1e3, plotting=False):
    '''
    Arguments:
    timestamps and nanotimes should be of equal length and
    both in the units of seconds
    window and period in number of photons
    '''
    length = len(timestamps)
    index_left = 0
    length_update = length - index_left
    df_fcs = pd.DataFrame()
    df_lt = pd.DataFrame()  # lifetiem
    df_ip = pd.DataFrame()  # interphoton
    df_ts = pd.DataFrame()
    while length_update > window:
        tleft = int(index_left)
        tright = int(index_left + window)
        # change "period" to "window" to avoid rolling
        index_left = int(index_left + period)
        length_update = int(length - index_left)

        t_mac_temp = timestamps[tleft:tright]
        t_mic_temp = 1e9 * nanotimes[tleft:tright]
        df_ts['t'] = t_mac_temp
        df_ts[str(tleft)] = t_mac_temp

        # interphoton histogram
        t_diff = np.diff(t_mac_temp)
        binned_trace = np.histogram(t_diff, bins=500, range=(1e-5, 1e-1))
        t = binned_trace[1][:-1]
        n = binned_trace[0]
        df_ip['t'] = t
        df_ip[str(tleft)] = n / max(n)
        # lifetime histogram
        binned_trace = np.histogram(t_mic_temp, bins=50, range=(0, 8))
        t = binned_trace[1][:-1]
        n = binned_trace[0]
        df_lt['t'] = t
        df_lt[str(tleft)] = n / max(n)
        # FCS
        bin_lags = make_loglags(-6, 0, 10)
        Gn = normalize_G(t_mac_temp, t_mac_temp, bin_lags)
        Gn = np.hstack((Gn[:1], Gn)) - 1
        df_fcs['t'] = bin_lags
        df_fcs[str(tleft)] = Gn
    if plotting:
        for column in df_ts.iloc[:, 1:]:
            plt.close('all')
            nrows = 3
            ncols = 2
            fig = plt.figure(figsize=(10, 8))
            ax00 = plt.subplot2grid((nrows, ncols), (0, 0), colspan=2)
            ax10 = plt.subplot2grid((nrows, ncols), (1, 0))
            ax11 = plt.subplot2grid((nrows, ncols), (1, 1))
            ax20 = plt.subplot2grid((nrows, ncols), (2, 0))
            ax21 = plt.subplot2grid((nrows, ncols), (2, 1))
            #plot all
            plot_timetrace(ax00, timestamps, bintime=5e-3)
            ax11.plot(df_lt.iloc[:, 0], df_lt.iloc[:, 1:], alpha=0.3)
            ax20.plot(df_ip.iloc[:, 0], df_ip.iloc[:, 1:], alpha=0.3)
            ax21.plot(df_fcs.iloc[:, 0], df_fcs.iloc[:, 1:], alpha=0.3)
            #plot individual
            plot_timetrace(ax10, df_ts[column], bintime=5e-3)
            ax00.axvspan(min(df_ts[column]), max(df_ts[column]),
                         color='r', alpha=0.3, lw=0)
            ax11.plot(df_lt.iloc[:, 0], df_lt[column],
                      '*b', ms=3, label='Fluorescence lifetime')
            ax20.plot(df_ip.iloc[:, 0], df_ip[column],
                      '*b', ms=3, label='Interphoton time')
            ax21.plot(df_fcs.iloc[:, 0], df_fcs[column],
                      '*b', ms=3, label='FCS')
            # axis properties
            ax00.set_ylim(0, None)
            ax10.set_ylim(0, None)
            ax10.legend(['Highlighted part of the trace'])
            ax11.set_xlim(4.5, 7.5)
            ax11.set_ylim(1e-1, None)
            ax11.set_yscale('log')
            ax11.set_xlabel('Lifetime/ns')
            ax11.set_ylabel('#')
            ax11.legend()
            ax20.set_xlim(1e-5, 1e-1)
            ax20.set_xscale('log')
            ax20.set_yscale('log')
            ax20.set_xlabel('Interphoton time/s')
            ax20.set_ylabel('#')
            ax20.legend()
            ax21.set_xlim(1e-5, 1)
            ax21.set_ylim(0, 4)
            ax21.set_xscale('log')
            ax21.set_xlabel('lag time/s')
            ax21.set_ylabel('G(t)-1')
            ax21.legend()
            fig.tight_layout()
            # save figure
            date = datetime.datetime.today().strftime('%Y%m%d_%H%M')
            savename = date+'_'+str(column) + '.png'
            savename = os.path.join(save_folder, savename)
            fig.savefig(savename, dpi=300)
    return df_ts, df_lt, df_fcs, df_ip
def longtrace_byparts_color(timestamps, nanotimes, save_folder,
                            window=1e4, period=1e3,
                            tmin = None, tmax=None,
                            by_photon=False, plotting=False):
    '''
    Arguments:
    timestamps and nanotimes should be of equal length and
    both in the units of seconds
    window and period in number of photons
    by_photon: by defaul it is false and it devides the trace by time. If true, it will devide the trace by the number of photons
    '''
    if tmax:
        mask = timestamps<tmax
        timestamps = timestamps[mask]
        nanotimes = nanotimes[mask]
    if tmin:
        mask = timestamps>tmin
        timestamps = timestamps[mask]
        nanotimes = nanotimes[mask]
    df_fcs = pd.DataFrame()
    df_lt = pd.DataFrame()  # lifetiem
    df_ip = pd.DataFrame()  # interphoton
    df_ts = pd.DataFrame()
    if by_photon:
        length = len(timestamps)
        index_left = 0
        length_update = length - index_left

        while length_update > window:
            tleft = int(index_left)
            tright = int(index_left + window)
            # change "period" to "window" to avoid rolling
            index_left = int(index_left + period)
            length_update = int(length - index_left)

            t_mac_temp = timestamps[tleft:tright]
            t_mic_temp = 1e9 * nanotimes[tleft:tright]
            
            df_ts['t'] = t_mac_temp
            df_ts[str(tleft)] = t_mac_temp

            # interphoton histogram
            t_diff = np.diff(t_mac_temp)
            binned_trace = np.histogram(t_diff, bins=500, range=(1e-5, 1e-1))
            t = binned_trace[1][:-1]
            n = binned_trace[0]
            df_ip['t'] = t
            df_ip[str(tleft)] = n / max(n)
            # lifetime histogram
            binned_trace = np.histogram(t_mic_temp, bins=50, range=(0, 8))
            t = binned_trace[1][:-1]
            n = binned_trace[0]
            df_lt['t'] = t
            df_lt[str(tleft)] = n / max(n)
            # FCS
            bin_lags = make_loglags(-6, 0, 10)
            Gn = normalize_G(t_mac_temp, t_mac_temp, bin_lags)
            Gn = np.hstack((Gn[:1], Gn)) - 1
            df_fcs['t'] = bin_lags
            df_fcs[str(tleft)] = Gn
    else:
        length = max(timestamps)
        index_left = min(timestamps)
        length_update = length - index_left
        col_names = []
        tleft = 0
        t_mac_temp = timestamps
        while length_update > window:
            tleft = int(index_left)
            tright = int(index_left + window)
            # change "period" to "window" to avoid rolling
            index_left = int(index_left + period)
            length_update = int(length - index_left)
            
            mask = np.logical_and(timestamps >= tleft, timestamps <= tright)
            t_mac_temp = timestamps[mask]
            t_mic_temp = 1e9 * nanotimes[mask]
#             df_ts['t'] = t_mac_temp
#             df_ts[str(tleft)] = t_mac_temp
            df_ts_temp = pd.DataFrame({str(tleft):t_mac_temp})
#             print(len(df_ts_temp))
            df_ts = pd.concat([df_ts, df_ts_temp], ignore_index=True, axis=1)
            # interphoton histogram
            t_diff = np.diff(t_mac_temp)
            binned_trace = np.histogram(t_diff, bins=500, range=(1e-5, 1e-1))
            t = binned_trace[1][:-1]
            n = binned_trace[0]
            df_ip['t'] = t
            df_ip[str(tleft)] = n / max(n)
            # lifetime histogram
            binned_trace = np.histogram(t_mic_temp, bins=50, range=(0, 8))
            t = binned_trace[1][:-1]
            n = binned_trace[0]
            df_lt['t'] = t
            df_lt[str(tleft)] = n / max(n)
            # FCS
            bin_lags = make_loglags(-6, 1, 10)
            Gn = normalize_G(t_mac_temp, t_mac_temp, bin_lags)
            Gn = np.hstack((Gn[:1], Gn)) - 1
            df_fcs['t'] = bin_lags
            df_fcs[str(tleft)] = Gn
            col_names.append(str(tleft))

        df_ts_temp = pd.DataFrame({str(tleft):t_mac_temp})
        df_ts = pd.concat([df_ts_temp, df_ts], ignore_index=True, axis=1)
        col_names.append(str(tright))
        print(col_names)
        df_ts.columns = col_names
    if plotting:
        plt.close('all')
        nrows = 3
        ncols = 2
        fig = plt.figure(figsize=(10, 8))
        ax00 = plt.subplot2grid((nrows, ncols), (0, 0), colspan=2)
        ax10 = plt.subplot2grid((nrows, ncols), (1, 0))
        ax11 = plt.subplot2grid((nrows, ncols), (1, 1))
        ax20 = plt.subplot2grid((nrows, ncols), (2, 0))
        ax21 = plt.subplot2grid((nrows, ncols), (2, 1))        
        cmap = plt.get_cmap('jet')#jet_r
        N=len(df_ts.columns)
        i=1
        for column in df_ts.iloc[:, 1:-1]:
            color = cmap(float(i)/N)
            i=i+1
            #plot all
            plot_timetrace(ax00, df_ts[column], bintime=5e-3, color=color)#timestamps
            #plot individual
            ax11.plot(df_lt.iloc[:, 0], df_lt[column],color=color)
            ax20.plot(df_ip.iloc[:, 0], df_ip[column],color=color)
            ax21.plot(df_fcs.iloc[:, 0], df_fcs[column],color=color)
        ax00.axvspan(min(df_ts[column]), max(df_ts[column]),color=color
                     , alpha=0.3, lw=0)          
        # axis properties
        plot_timetrace(ax10, df_ts[column], bintime=5e-3, color=color)
        ax00.set_ylim(0, None)
        ax00.set_xlim(min(timestamps), max(df_ts[column]))
        ax10.set_ylim(0, None)
        ax10.set_xlim(min(df_ts[column]), max(df_ts[column]))
        ax10.legend(['Highlighted part of the trace'])
        ax11.set_xlim(2, 8)
        ax11.set_ylim(1e-1, None)
        ax11.set_yscale('log')
        ax11.set_xlabel('Lifetime/ns')
        ax11.set_ylabel('#')
        ax20.set_xlim(1e-5, 1e-1)
        ax20.set_xscale('log')
        ax20.set_yscale('log')
        ax20.set_xlabel('Interphoton time/s')
        ax20.set_ylabel('#')
        ax21.set_xlim(1e-5, 1)
        ax21.set_ylim(0, 4)
        ax21.set_xscale('log')
        ax21.set_xlabel('lag time/s')
        ax21.set_ylabel('G(t)-1')
        ax11.text(3, 0.15, 'Life time histogram', style='italic')
        ax20.text(2e-5, 1e-5, 'Inter photon times')
        ax21.text(2e-5, 1,'FCS')
        # fig.tight_layout()
        # save figure
        date = datetime.datetime.today().strftime('%Y%m%d_%H%M')
        savename = date+'_'+str(column) + '.png'
        savename = os.path.join(save_folder, savename)
        fig.savefig(savename, dpi=300)
    return df_ts, df_lt, df_fcs, df_ip
def plot_timetrace(ax, timestamps, bintime, color='b'):
    tmin = min(timestamps)
    tmax = max(timestamps)
    tt_length = tmax - tmin
    binpts = int(tt_length / bintime)
    hist, trace = np.histogram(timestamps, bins=binpts,
                               range=(tmin, tmax))
    ax.plot(trace[:-1], hist * 1e-3 / bintime, color=color)
    ax.set_ylabel('counts/kcps')
    ax.set_xlabel('time/s')
    # ax.set_xlim(tmin, tmax)
    #ax.set_title('bintime: ' + str(bintime))
    return


def photonhdf5_longtrace_byparts(file_path_hdf5, delete_oldpng=True):
    file_path_datn = file_path_hdf5[:-4] + 'pt3.datn'
    df_datn = pd.read_csv(file_path_datn, header=None)
    tmin = min(df_datn[0])
    tmax = max(df_datn[0])
    h5 = h5py.File(file_path_hdf5)
    unit = h5['photon_data']['timestamps_specs']['timestamps_unit'][...]
    tcspc_unit = h5['photon_data']['nanotimes_specs']['tcspc_unit'][...]
    timestamps = unit * h5['photon_data']['timestamps'][...]
    nanotimes = tcspc_unit * h5['photon_data']['nanotimes'][...]
    mask = np.logical_and(timestamps >= tmin, timestamps <= tmax)
    timestamps = timestamps[mask]
    nanotimes = nanotimes[mask]
    h5.close()
    hdf5file_name = os.path.basename(file_path_hdf5)
    longtraces_folder = '/home/biswajit/Downloads/temp/longtraces'
    save_folder = os.path.join(longtraces_folder, hdf5file_name[:-5])
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if delete_oldpng:
        filelist = [f for f in os.listdir(save_folder) if f.endswith(".png")]
        for f in filelist:
            os.remove(os.path.join(save_folder, f))
    out = longtrace_byparts(timestamps, nanotimes, save_folder,
                            window=1e4, period=1e3, plotting=True)
    return out


def simulatedhdf5_longtrace_byparts(simulated_hdf5, delete_oldpng=True):
    h5 = h5py.File(hdf5_simulated, 'r')
    timestamps = h5['onexp_offexp']['timestamps'][...]
    nanotimes = h5['onexp_offexp']['nanotimes'][...]
    h5.close()
    hdf5file_name = os.path.basename(simulated_hdf5)
    longtraces_folder = '/home/biswajit/Downloads/temp/longtraces'
    save_folder = os.path.join(longtraces_folder, hdf5file_name[:-5])
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if delete_oldpng:
        filelist = [f for f in os.listdir(save_folder) if f.endswith(".png")]
        for f in filelist:
            os.remove(os.path.join(save_folder, f))
    out = longtrace_byparts(timestamps, nanotimes, save_folder,
                            window=1e4, period=1e3, plotting=True)
    return out
#===============fitting functions=============
def risetime_fit(t, k1, k2, A):
    return ((A*k1*k2/(k2-k1)) * (exp(-k1*t) - exp(-k2*t)))
def mono_exp(t, k1, A):
    return A*exp(-k1*t)
def gaussian(x, a, b, c):
    return a*exp((-(x-b)**2)/(2*c**2))    
def streched_exp(t, k, b, A):
    return A*np.exp(-(k*t)**b)
