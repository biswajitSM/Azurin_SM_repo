import os
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from scipy.optimize import curve_fit

from ChangePointProcess import digitize_photonstamps, digitize_simulatedphoton
from autocorrelate import autocorrelate
from pycorrelate import make_loglags, normalize_G

class LongTraceClass(object):
    """docstring for LongTraceClass"""
    parameters = {
                "TimeLimit": [None, None],
                "BintimeForIntTrace" : 5e-3,
                "BinsForBrightHist": 50,
                "RangeForBrightHist": (0, 1),
                "BinsForDarkHist": 50,
                "RangeForDarkHist": (0, 10),
                "PlotInsetForDurationHist": True,
                "InsetRangeBrightHist": (0, 0.05),
                "InsetRangeDarkHist": (0, 0.1),
                "NumPointsForAveraging": 10,
                "RangeForMidPotentialHist": (0, 125),
                "BinsForMidPotentialHist": 20,
                "AppliedPotential": 100,
                "LagTimeLimitCorrelation": (0, 100),
                "ContrastLimitCorrelation":(None, None),
                "FigureSize": (8, 8),
                "ChangePointParams" : (1, 0.01, 0.99, 2)
                }

    def __init__(self, file_path_hdf5, Simulation = False,
        parameters=parameters):

        self.Simulation = Simulation
        self.file_path_hdf5 = file_path_hdf5
        for key in parameters:
            setattr(self, key, parameters[key])
        if self.Simulation:
            h5 = h5py.File(self.file_path_hdf5, 'r')
            self.timestamps = h5['onexp_offexp']['timestamps'][...]
            self.nanotimes = h5['onexp_offexp']['nanotimes'][...]
            h5.close()
            # update min and max time limit
            if self.TimeLimit[0] is None:
                self.TimeLimit[0] = min(self.timestamps)
            if self.TimeLimit[1] is None:
                self.TimeLimit[1] = max(self.timestamps)
        else:
            h5 = h5py.File(self.file_path_hdf5);
            self.unit = h5['photon_data']['timestamps_specs']['timestamps_unit'][...]
            self.tcspc_unit = h5['photon_data']['nanotimes_specs']['tcspc_unit'][...]
            self.timestamps = self.unit * h5['photon_data']['timestamps'][...]
            self.nanotimes = self.tcspc_unit * h5['photon_data']['nanotimes'][...]
            h5.close()
            # update min and max time limit
            if self.TimeLimit[0] is None:
                self.TimeLimit[0] = min(self.timestamps)
            if self.TimeLimit[1] is None:
                self.TimeLimit[1] = max(self.timestamps)

    def DigitizePhotons(self, int_photon=False,
        nanotimes_bool=False,
        duration_cp=False):

        if self.Simulation:
            df_dig = digitize_simulatedphoton(
                            simulatedhdf5 = self.file_path_hdf5,
                            pars = self.ChangePointParams,
                            time_sect = 100,
                            time_lim = self.TimeLimit,
                            bintime = self.BintimeForIntTrace,
                            int_photon = int_photon,
                            nanotimes_bool = nanotimes_bool,
                            duration_cp = duration_cp)
        else:
            df_dig = digitize_photonstamps(
                            file_path_hdf5 = self.file_path_hdf5,
                            pars=self.ChangePointParams,
                            time_sect=100,
                            time_lim=self.TimeLimit,
                            bintime=self.BintimeForIntTrace,
                            int_photon = int_photon,
                            nanotimes_bool = nanotimes_bool,
                            duration_cp = duration_cp)
        return df_dig

    def onoff_values(self):
        df_dig = self.DigitizePhotons()
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
        self.df_durations = pd.DataFrame()
        self.df_durations['abstime_on'] = abstime_on.values[:l]
        self.df_durations['ontimes'] = ontimes.values[:l]
        self.df_durations['abstime_off'] = abstime_off.values[:l]
        self.df_durations['offtimes'] = offtimes.values[:l]
        return self.df_durations

    def PlotDurationsVsTime(self):
        # Define axis positions and numbers
        self.FigureDuration = plt.figure(figsize=self.FigureSize)
        nrows=4; ncols= 4;
        # self.ax00 = plt.subplot2grid((nrows,ncols), (0,0));
        self.axis01 = plt.subplot2grid((nrows,ncols), (0,1), colspan=3);
        self.axis10 = plt.subplot2grid((nrows,ncols), (1,0));
        self.axis11 = plt.subplot2grid((nrows, ncols), (1,1), colspan=3);
        self.axis20 = plt.subplot2grid((nrows,ncols), (2,0));
        self.axis21 = plt.subplot2grid((nrows, ncols), (2,1), colspan=3)
        self.axis30 = plt.subplot2grid((nrows,ncols), (3,0));
        self.axis31 = plt.subplot2grid((nrows, ncols), (3,1), colspan=3)
        # get Bright and Dark times
        df_durations = self.onoff_values()
        df_mean = df_durations.groupby(np.arange(len(df_durations))//
            self.NumPointsForAveraging).mean().dropna().reset_index(drop=True)
        df_roll_mean = df_durations.rolling(
            self.NumPointsForAveraging).mean().dropna().reset_index(drop=True)        
        df_select = df_roll_mean
        # Plot in each axis
        plot_timetrace(self.axis01, self.timestamps,
                        self.BintimeForIntTrace, color='b')
        self.axis01.set_xlim(self.TimeLimit)
        waitime_hist_inset(waitimes = df_durations['ontimes'],
                            axis = self.axis10,
                            bins = self.BinsForBrightHist,
                            binrange = self.RangeForBrightHist,
                            insetrange = self.InsetRangeBrightHist,
                            fit_func = streched_exp,
                            PlotInset = self.PlotInsetForDurationHist,
                            )
        waitime_hist_inset(waitimes = df_durations['offtimes'],
                            axis = self.axis20,
                            bins = self.BinsForDarkHist,
                            binrange = self.RangeForDarkHist,
                            insetrange = self.InsetRangeDarkHist,
                            fit_func = streched_exp,
                            PlotInset = self.PlotInsetForDurationHist,
                            )
        self.axis11.plot(df_select['abstime_on'],df_select['ontimes'],
                            'b', label='Bright times')
        self.axis11.set_xlim(self.TimeLimit)
        self.axis21.plot(df_select['abstime_off'],df_select['offtimes'],
                        'r', label='Dark times')
        self.axis21.set_xlim(self.TimeLimit)

        E0_list = MidPointPotentialTimeTrace(self.axis31,
                            df_select['ontimes'], df_select['offtimes'],
                            df_select['abstime_on'],
                            AppliedPotential = self.AppliedPotential,
                            E0range = self.RangeForMidPotentialHist,
                            TimeLimit = self.TimeLimit)

        out_E0fit = MidPointPotentialGaussFit(self.axis30,
                            E0_list = E0_list,
                            bins = self.BinsForMidPotentialHist,
                            E0range = self.RangeForMidPotentialHist)

    def PlotTimeTrace(self, figsize=(10, 4)):
        # Define axis positions and numbers
        self.FigureTimetrace = plt.figure(figsize=figsize)
        nrows=1; ncols= 1;
        self.axis00 = plt.subplot2grid((nrows,ncols), (0,0), colspan=1)
        plot_timetrace(self.axis00, self.timestamps,
                        self.BintimeForIntTrace, color='b')
        self.axis00.set_xlim(self.TimeLimit)

    def PlotCorrelation(self):
        # Define axis positions and numbers
        self.FigureCorrelation = plt.figure(figsize=(6, 3))
        nrows=1; ncols= 1;
        self.axis00 = plt.subplot2grid((nrows,ncols), (0,0), colspan=1)
        
        df_durations = self.onoff_values()
        df_mean = df_durations.groupby(np.arange(len(df_durations))//
            self.NumPointsForAveraging).mean().dropna().reset_index(drop=True)
        df_roll_mean = df_durations.rolling(
            self.NumPointsForAveraging).mean().dropna().reset_index(drop=True)        
        df_select = df_durations                
        self.CorrBright, self.CorrDark = CorrelationBrightDark(self.axis00,
                            df_select["ontimes"], df_select["offtimes"],
                            tlag_lim = self.LagTimeLimitCorrelation,
                            G_lim = self.ContrastLimitCorrelation)

    def PlotFCS(self):
        self.FigureCorrelation = plt.figure(figsize=(6, 3))
        nrows=1; ncols= 1;
        self.axis00 = plt.subplot2grid((nrows,ncols), (0,0), colspan=1)
        bin_lags = make_loglags(-6, 2, 10)
        G = normalize_G(self.timestamps, self.timestamps, bin_lags)
        self.G = np.hstack((G[:1], G)) - 1
        self.axis00.plot(bin_lags, self.G)
        self.axis00.set_xlabel('lag time/s')
        self.axis00.set_ylabel('G(0)-1')
        self.axis00.set_xscale('log')
        self.axis00.set_xlim(min(bin_lags))

    def LongTraceByParts(self,
        TimeWindow = 1e2, TimePeriod = 1e2,
        PhotonWindow = 1e4, PhotonPeriod = 1e4,
        by_photon=False, plotting=False):
        '''
        Arguments:
        timestamps and nanotimes should be of equal length and
        both in the units of seconds
        window and period in number of photons
        by_photon: by defaul it is false and it devides the trace by time. If true, it will devide the trace by the number of photons
        '''
        self.TimeWindow = TimeWindow
        self.TimePeriod = TimePeriod
        self.PhotonWindow = PhotonWindow
        self.PhotonPeriod = PhotonPeriod

        # update timestamps and nanotimes
        mask = np.logical_and(self.timestamps >= self.TimeLimit[0],
                                self.timestamps <= self.TimeLimit[1])
        timestamps = self.timestamps[mask]
        nanotimes = self.nanotimes[mask]
        df_fcs = pd.DataFrame()
        df_lt = pd.DataFrame()  # lifetiem
        df_ip = pd.DataFrame()  # interphoton
        df_ts = pd.DataFrame()
        if by_photon:
            length = len(timestamps)
            index_left = 0
            length_update = length - index_left

            while length_update > self.PhotonWindow:
                tleft = int(index_left)
                tright = int(index_left + self.PhotonWindow)
                # change "period" to "window" to avoid rolling
                index_left = int(index_left + self.PhotonPeriod)
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
            '''Analyze time trace splitted/parted with with time window'''
            length = max(timestamps)
            index_left = min(timestamps)
            length_update = length - index_left
            col_names = []
            tleft = 0
            t_mac_temp = timestamps
            while length_update > self.TimeWindow:
                tleft = int(index_left)
                tright = int(index_left + self.TimeWindow)
                # change "period" to "window" to avoid rolling
                index_left = int(index_left + self.TimePeriod)
                length_update = int(length - index_left)
                
                mask = np.logical_and(timestamps >= tleft, timestamps <= tright)
                t_mac_temp = timestamps[mask]
                t_mic_temp = 1e9 * nanotimes[mask]
                # df_ts['t'] = t_mac_temp
                # df_ts[str(tleft)] = t_mac_temp
                df_ts_temp = pd.DataFrame({str(tleft):t_mac_temp})
                # print(len(df_ts_temp))
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
            nrows = 3
            ncols = 2
            self.FigureByParts = plt.figure(figsize=(10, 8))
            self.axis00 = plt.subplot2grid((nrows, ncols), (0, 0), colspan=2)
            self.axis10 = plt.subplot2grid((nrows, ncols), (1, 0))
            self.axis11 = plt.subplot2grid((nrows, ncols), (1, 1))
            self.axis20 = plt.subplot2grid((nrows, ncols), (2, 0))
            self.axis21 = plt.subplot2grid((nrows, ncols), (2, 1))        
            cmap = plt.get_cmap('jet')#jet_r
            N=len(df_ts.columns)
            i=1
            for column in df_ts.iloc[:, 1:-1]:
                color = cmap(float(i)/N)
                i=i+1
                #plot all
                plot_timetrace(self.axis00, df_ts[column], bintime=5e-3, color=color)#timestamps
                #plot individual
                self.axis11.plot(df_lt.iloc[:, 0], df_lt[column],color=color)
                self.axis20.plot(df_ip.iloc[:, 0], df_ip[column],color=color)
                self.axis21.plot(df_fcs.iloc[:, 0], df_fcs[column],color=color)
            self.axis00.axvspan(min(df_ts[column]), max(df_ts[column]),color=color
                         ,alpha=0.3, lw=0)          
            # axis properties
            plot_timetrace(self.axis10, df_ts[column], bintime=5e-3, color=color)
            self.axis00.set_ylim(0, None)
            self.axis00.set_xlim(min(timestamps), max(df_ts[column]))
            self.axis10.set_ylim(0, None)
            self.axis10.set_xlim(min(df_ts[column]), max(df_ts[column]))
            self.axis10.legend(['Highlighted part of the trace'])
            self.axis11.set_xlim(2, 8)
            self.axis11.set_ylim(1e-1, None)
            self.axis11.set_yscale('log')
            self.axis11.set_xlabel('Lifetime/ns')
            self.axis11.set_ylabel('#')
            self.axis20.set_xlim(1e-5, 1e-1)
            self.axis20.set_xscale('log')
            self.axis20.set_yscale('log')
            self.axis20.set_xlabel('Interphoton time/s')
            self.axis20.set_ylabel('#')
            self.axis21.set_xlim(1e-5, 1)
            self.axis21.set_ylim(0, 4)
            self.axis21.set_xscale('log')
            self.axis21.set_xlabel('lag time/s')
            self.axis21.set_ylabel('G(t)-1')
            self.axis11.text(3, 0.15, 'Life time histogram', style='italic')
            self.axis20.text(2e-5, 1e-5, 'Inter photon times')
            self.axis21.text(2e-5, 1,'FCS')
        return df_ts, df_lt, df_fcs, df_ip

def plot_timetrace(axis, timestamps, bintime, color='b'):
    tmin = min(timestamps)
    tmax = max(timestamps)
    tt_length = tmax - tmin
    binpts = int(tt_length / bintime)
    hist, trace = np.histogram(timestamps, bins=binpts,
                               range=(tmin, tmax))
    axis.plot(trace[:-1], hist * 1e-3 / bintime, color=color)
    axis.set_ylabel('counts/kcps')
    axis.set_xlabel('time/s')
    # ax.set_xlim(tmin, tmax)
    #ax.set_title('bintime: ' + str(bintime))
    return

def waitime_hist_inset(waitimes, axis, bins, binrange,
    insetrange, fit_func, PlotInset):
    '''waitimes: list of on-times or off-times
    '''
    n,bins_hist = np.histogram(waitimes, bins=bins,
                            range=(min(waitimes) , binrange[1]))#avoiding zero
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
    #plot as bar
    from matplotlib import pyplot, transforms
    rot = transforms.Affine2D().rotate_deg(90)
    axis.bar(t, n, width=binwidth, color='k', alpha=0.5)
    axis.plot(t_fit, fit_func(t_fit, *fit), 'k',
        lw=1,label='k1:'+str(fit[0])+'\n'+str(fit[1]))
    axis.set_xlim(0, None)
    axis.set_ylim(1e0, None)
    axis.set_yscale('log')
    #inset
    if PlotInset:
        axis_in = inset_axes(axis, height="50%", width="50%")
        axis_in.bar(t, n, width=binwidth, color='k', alpha=0.5)
        axis_in.plot(t, n, drawstyle= 'steps-mid', lw=0.5, color='k')
        axis_in.plot(t_fit, fit_func(t_fit, *fit), 'k',
            lw=1,label='k1:'+str(fit[0])+'\n'+str(fit[1]))
        axis_in.set_xlim(insetrange)
        axis_in.get_yaxis().set_ticklabels([])

def MidPointPotentialTimeTrace(axis, t_av_on, t_av_off,
    t_abs, AppliedPotential, E0range, TimeLimit):
    t_on_ratio = t_av_off/t_av_on;
    E0_list = AppliedPotential - 59*np.log10(t_on_ratio);
    axis.plot(t_abs, E0_list, label='E_0')
    axis.set_xlim(TimeLimit)
    axis.set_ylim(E0range)
    axis.set_ylabel('E_0/mV')
    axis.set_xlabel('time/s')
    axis.yaxis.set_label_position("right")
    axis.yaxis.tick_right()
    return E0_list

def MidPointPotentialGaussFit(axis, E0_list, bins, E0range):
    bin_centers = np.linspace(E0range[0], E0range[1], bins);
    n, bins_hist, patches = axis.hist(E0_list, bins=bins, range=E0range)
    y=n; x=bin_centers;
    fit, pcov = curve_fit(gaussian, xdata=x, ydata=y,
        p0=[100, 5, 25], bounds=(-np.inf, np.inf))
    perr = np.sqrt(np.diag(pcov));
    center=round(fit[1], 1); center_err = round(perr[1], 1);
    fwhm=round(2.3548*fit[2],1); fwhm_err = round(2.3548*perr[2],1);
    #print('Center is %.1f and fwhm is %.1f' %(center, fwhm))
    E0_fitrange = np.linspace(E0range[0], E0range[1], 100);
    axis.plot(E0_fitrange, gaussian(E0_fitrange, *fit),
        label=str(center)+'+-'+str(center_err)+'\n'+
        str(fwhm)+'+-'+str(fwhm_err)+'\n')
    axis.set_xlabel('Midpoint Potential/mV')
    axis.set_ylabel('#')
    axis.set_xlabel(r'$E_0/mV$')
    axis.set_xlim(E0range[0], E0range[1]);
    axis.set_xticks([])
    axis.set_yticks([])        
    axis.legend()
    return center, center_err, fwhm, fwhm_err

def CorrelationBrightDark(axis, t_av_on, t_av_off, tlag_lim, G_lim):
    G_on = []; G_off = [];
    ymax_lim=0.2;
    # try:
    m=(len(t_av_on)/5)-5;
    t_av_tot = np.average(t_av_on)+np.average(t_av_off);
    G_on = autocorrelate(t_av_on[:-1], m=m ,deltat=1,normalize=True);
    axis.plot(G_on[:,0]*t_av_tot, G_on[:,1], 'b',label='On time correlation')
    ymax_lim_on = G_on[0,1]
    #off-time correlation-------
    m=(len(t_av_off)/2)-5;
    G_off = autocorrelate(t_av_off[:-1], m=m ,deltat=1,normalize=True);
    axis.plot(G_off[:,0]*t_av_tot, G_off[:,1], 'r',label='Off time correlation')
    axis.set_xscale('log')
    axis.set_xlabel('time/s')
    axis.set_ylabel('G(t)')
    ymax_lim_off = G_off[0,1]
    if ymax_lim_on > ymax_lim_off:
        ymax_lim = ymax_lim_on;
    else:
        ymax_lim = ymax_lim_off;
    axis.legend()
    # except:
    #     pass
    axis.set_ylim(0, ymax_lim)#+0.01*ymax_lim
    axis.set_ylim(G_lim)#+0.01*ymax_lim
    axis.set_xlim(tlag_lim)       
    return G_on, G_off
#===============fitting functions=============
def risetime_fit(t, k1, k2, A):
    return ((A*k1*k2/(k2-k1)) * (np.exp(-k1*t) - np.exp(-k2*t)))
def mono_exp(t, k1, A):
    return A*np.exp(-k1*t)
def gaussian(x, a, b, c):
    return a*np.exp((-(x-b)**2)/(2*c**2))    
def streched_exp(t, k, b, A):
    return A*np.exp(-(k*t)**b)
    