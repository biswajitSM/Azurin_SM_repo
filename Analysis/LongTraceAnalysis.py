import os
import numpy as np
import pandas as pd
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from scipy.optimize import curve_fit
import lmfit
import yaml

from ChangePointProcess import digitize_photonstamps
from autocorrelate import autocorrelate
from pycorrelate import make_loglags, normalize_G, t_on_off_fromFCS
from LifeTimeFluorescence import LifeTimeFitTail, BiExpLifetime

class LongTraceClass(object):
    """docstring for LongTraceClass"""
    parameters = {
                "TimeLimit": [None, None],
                "BintimeForIntTrace" : 5e-3,
                "BinsForBrightHist": 50,
                "RangeForBrightHist": [0.01, 1],
                "BinsForDarkHist": 50,
                "RangeForDarkHist": [0.01, 10],
                "PlotInsetForDurationHist": False,
                "InsetRangeBrightHist": [0, 0.05],
                "InsetRangeDarkHist": [0, 0.1],
                "AveragingType": 'rolling', # 'rolling' or 'mean' or 'noAveraging'
                "NumPointsForAveraging": 10,
                "BinsForAvgBrightHist": 50,
                "BinsForAvgDarkHist": 50,
                "Range2DHistBright": [0, 1],
                "Range2DHistDark": [0, 1],
                "RangeForMidPotentialHist": [0, 125],
                "BinsForMidPotentialHist": 50,
                "LagTimeLimitCorrelation": [0, 100],
                "ContrastLimitCorrelation":(None, None),
                "FigureSize": (12, 12),
                "ChangePointParams" : (1, 0.01, 0.99, 2),
                "BinsFCS" : 10,
                "RangeFCS" : [-5, 2]
                }

    def __init__(self, file_path_hdf5, SimulatedHDF5,
                 Simulation = False, parameters=parameters):

        print('Input file is {}'.format(os.path.basename(file_path_hdf5)))
        self.Simulation = Simulation
        self.file_path_hdf5 = file_path_hdf5
        self.SimulatedHDF5 = SimulatedHDF5
        for key in parameters:
            setattr(self, key, parameters[key])
        self.FilePathYaml = self.file_path_hdf5[:-5] + '.yaml'
        with open(self.FilePathYaml) as f:
            dfyaml = yaml.load(f)
        tmin = dfyaml['TimeLimit']['MinTime']
        tmax = dfyaml['TimeLimit']['MaxTime']
        self.TimeLimit = [tmin, tmax]
        self.AppliedPotential = dfyaml['Potential']['Value'] # Unit: mV
        if self.Simulation:
            h5 = h5py.File(self.file_path_hdf5, 'r')
            self.timestamps = h5['onexp_offexp']['timestamps'][...]
            self.nanotimes = h5['onexp_offexp']['nanotimes'][...]
            h5.close()
            # update min and max time limit
            self.TimeLimit = [0, 2500]
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
            df_dig = digitize_photonstamps(
                            file_path_hdf5=self.file_path_hdf5,
                            pars = self.ChangePointParams,
                            time_sect = 100,
                            Simulated = True,
                            time_lim = self.TimeLimit,
                            bintime = self.BintimeForIntTrace,
                            cp_no = True,
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
                            cp_no = True,
                            int_photon = int_photon,
                            nanotimes_bool = nanotimes_bool,
                            duration_cp = duration_cp)
        return df_dig

    def onoff_values(self):
        df_dig = self.DigitizePhotons()
        df_on = df_dig[df_dig['state']==2].reset_index(drop=True)
        df_off = df_dig[df_dig['state']==1].reset_index(drop=True)
        time_left = df_on.groupby('cp_no').timestamps.min()
        time_right = df_on.groupby('cp_no').timestamps.max()
        abstime_on = df_on.groupby('cp_no').timestamps.mean()
        ontimes = time_right - time_left
        
        time_left = df_off.groupby('cp_no').timestamps.min()
        time_right = df_off.groupby('cp_no').timestamps.max()    
        abstime_off = df_off.groupby('cp_no').timestamps.mean()
        offtimes = time_right - time_left
        l = len(abstime_on) - 2
        self.df_durations = pd.DataFrame()
        self.df_durations['abstime_on'] = abstime_on.values[:l]
        self.df_durations['ontimes'] = ontimes.values[:l]
        self.df_durations['abstime_off'] = abstime_off.values[:l]
        self.df_durations['offtimes'] = offtimes.values[:l]
        # for simulated file
        h5 = h5py.File(self.SimulatedHDF5, 'r')
        ontimes = h5['onexp_offexp']['ontimes_exp'][...]
        offtimes = h5['onexp_offexp']['offtimes_exp'][...]
        h5.close()
        abstime = np.cumsum(ontimes) + np.cumsum(offtimes)
        self.dfDurationSimulated = pd.DataFrame({'abstime_on': abstime,
                                                 'ontimes': ontimes,
                                                 'abstime_off': abstime,
                                                 'offtimes': offtimes,
                                                 })
        return

    def PlotDurationsVsTime(self):
        # Define axis positions and numbers
        self.FigureDuration = plt.figure(figsize=self.FigureSize)
        nrows=4
        ncols= 4
        self.axis00 = plt.subplot2grid((nrows,ncols), (0,0), colspan=3)
        self.axis03 = plt.subplot2grid((nrows,ncols), (0,3))
        self.axis10 = plt.subplot2grid((nrows,ncols), (1,0), colspan=3)
        self.axis13 = plt.subplot2grid((nrows, ncols), (1,3))
        self.axis20 = plt.subplot2grid((nrows, ncols), (2, 0), colspan=3)
        self.axis23 = plt.subplot2grid((nrows, ncols), (2,3))
        self.axis30 = plt.subplot2grid((nrows,ncols), (3,0))
        self.axis31 = plt.subplot2grid((nrows, ncols), (3,1))
        self.axis32 = plt.subplot2grid((nrows, ncols), (3, 2))
        self.axis33 = plt.subplot2grid((nrows, ncols), (3,3))
        # get Bright and Dark times
        df_durations = self.df_durations
        dfDurationSimulated = self.dfDurationSimulated
        if self.AveragingType == 'noAveraging':
            dfAverage = self.df_durations
            dfAverageSimulated = dfDurationSimulated
        elif self.AveragingType == 'mean':
            dfAverage = df_durations.groupby(np.arange(len(df_durations))//
                self.NumPointsForAveraging).mean().dropna().reset_index(drop=True)
            dfAverageSimulated = dfDurationSimulated.groupby(np.arange(len(dfDurationSimulated))//
                self.NumPointsForAveraging).mean().dropna().reset_index(drop=True)
        elif self.AveragingType == 'rolling':
            dfAverage = df_durations.rolling(
                self.NumPointsForAveraging).mean().dropna().reset_index(drop=True)
            dfAverageSimulated = dfDurationSimulated.rolling(
                self.NumPointsForAveraging).mean().dropna().reset_index(drop=True)
        # MidPotential calculation 
        TimeRatio = dfAverage['offtimes']/dfAverage['ontimes']
        dfE0Average = pd.DataFrame()
        dfE0Average['E0'] = self.AppliedPotential - 59 * np.log10(TimeRatio)
        dfE0Average['abstime'] = dfAverage['abstime_on']
        TimeRatio = dfAverageSimulated['offtimes']/dfAverageSimulated['ontimes']
        dfE0AverageSimulated = pd.DataFrame()
        dfE0AverageSimulated['E0'] = self.AppliedPotential - 59 * np.log10(TimeRatio)
        dfE0AverageSimulated['abstime'] = dfAverageSimulated['abstime_on']
        # # Plot in each axis
        # # plot_timetrace(self.axis01, self.timestamps,
        # #                 self.BintimeForIntTrace, color='b')
        # # self.axis01.set_xlim(self.TimeLimit)
 
        self.axis00.plot(dfAverage['abstime_on'], dfAverage['ontimes'],
                         'b', label='Average Bright times')
        self.axis00.set_xlim(self.TimeLimit)
        self.axis00.set_ylabel(r'$Avg \tau_b/s$')
        self.axis00.set_xticklabels([])
        self.axis00.legend()
        ylimAxis00 = self.axis00.get_ylim()
        brightHist = np.histogram(dfAverage['ontimes'].values,
                                  bins = 50,
                                  range=self.axis00.get_ylim(),
                                  density = True)
        self.axis03.plot(brightHist[0], brightHist[1][:-1], 'b')
        # plot simulated for comparision
        brightHist = np.histogram(dfAverageSimulated['ontimes'].values,
                                  bins = 50,
                                  range=self.axis00.get_ylim(),
                                  density = True)
        self.axis03.plot(brightHist[0], brightHist[1][:-1], '--b')
        # result = AverageDurationsLogNormalFit(axis = self.axis03,
        #                         AvgDurations = dfAverage['ontimes'].values,
        #                         Bins=50, BinRange=ylimAxis00,
        #                         color ='b', Plotting=True)
        # # print(self.axis00.get_ylim())
        # BinsSimulated = np.linspace(ylimAxis00[0], ylimAxis00[1], 100)
        # self.axis03.plot(logNormal(BinsSimulated, 0.8*result.values['amp'],
        #                             result.values['cen'], 0.36),
        #                             BinsSimulated, '--k')
        self.axis03.set_yticklabels([])
        self.axis03.set_xticklabels([])
        self.axis03.set_xlabel('PDF')
        
        self.axis10.plot(dfAverage['abstime_off'],dfAverage['offtimes'],
                        'r', label='Average Dark times')
        self.axis10.set_xlim(self.TimeLimit)
        self.axis10.set_ylabel(r'$Avg \tau_d/s$')
        self.axis10.set_xticklabels([])
        self.axis10.legend()
        ylimAxis = self.axis10.get_ylim()
        darkHist = np.histogram(dfAverage['offtimes'].values,
                                  bins = 50,
                                  range = self.axis10.get_ylim(),
                                  density = True)
        self.axis13.plot(darkHist[0], darkHist[1][:-1], 'r')
        darkHist = np.histogram(dfAverageSimulated['offtimes'].values,
                                  bins = 50,
                                  range=self.axis10.get_ylim(),
                                  density = True)
        self.axis13.plot(darkHist[0], darkHist[1][:-1], '--r')

        # result = AverageDurationsLogNormalFit(axis = self.axis13,
        #                         AvgDurations = dfAverage['offtimes'].values,
        #                         Bins=50, BinRange=ylimAxis,
        #                         color ='r', Plotting=True)        
        # BinsSimulated = np.linspace(ylimAxis[0], ylimAxis[1], 100)
        # self.axis13.plot(logNormal(BinsSimulated, 0.8 * result.values['amp'],
        #                            result.values['cen'], 0.3),
        #                  BinsSimulated, '--k')
        self.axis13.set_yticklabels([])
        self.axis13.set_xticklabels([])
        self.axis13.set_xlabel('PDF')

        self.axis20.plot(dfE0Average['abstime'], dfE0Average['E0'], 'm', label=r'$E_0/mV$')
        self.axis20.set_ylabel(r'$E_0/mV$')
        E0lim = self.axis20.get_ylim()
        E0Hist = np.histogram(dfE0Average['E0'].values,
                                  bins = 50,
                                  range = self.axis20.get_ylim(),
                                  density = True)
        self.axis23.plot(E0Hist[0], E0Hist[1][:-1], 'm')
        E0Hist = np.histogram(dfE0AverageSimulated['E0'].values,
                                  bins = 50,
                                  range = self.axis20.get_ylim(),
                                  density = True)
        self.axis23.plot(E0Hist[0], E0Hist[1][:-1], '--m')

        self.axis23.set_yticklabels([])
        self.axis23.set_xticklabels([])
        self.axis23.set_xlabel('#')
        self.axis23.set_ylabel('')

        HistBrightRaw = waitime_hist_inset(waitimes = df_durations['ontimes'],
                            axis = self.axis31,
                            bins = self.BinsForBrightHist,
                            binrange = [0, 2],
                            insetrange = self.InsetRangeBrightHist,
                            fit_func = streched_exp,
                            PlotInset = self.PlotInsetForDurationHist,
                            color = 'b'
                            )
        self.axis31.legend(['Bright time','Dark times',
                            'Dark times','fit'])
        self.axis31.set_xlabel('time/s', color='b')
        self.axis31.tick_params('x', direction='in', colors='b')

        axis31_up = plt.twiny(ax = self.axis31)
        HistDarkRaw = waitime_hist_inset(waitimes = df_durations['offtimes'],
                            axis=axis31_up,
                            bins = self.BinsForDarkHist,
                            binrange = [0, 10],
                            insetrange = self.InsetRangeDarkHist,
                            fit_func = streched_exp,
                            PlotInset = self.PlotInsetForDurationHist,
                            color = 'r'
                            )
        # self.axis31.set_xlim(0, 10)
        axis31_up.set_xlabel('')
        axis31_up.tick_params('x', direction='in', colors='r')
        self.axis31.set_yticklabels([])
        self.axis31.set_ylabel('log(PDF)')
                
        Plot2Ddurations(self.axis30, df_durations, RollMean=False,
                        NumPointsForAveraging=self.NumPointsForAveraging,
                        shift_range=range(1, 10, 1),
                        ontimes=True,
                        bins=40, rangehist=self.Range2DHistBright)
        out = CorrelationBrightDark(self.axis32,
                                    df_durations["ontimes"],
                                    df_durations["offtimes"],
                                    tlag_lim=self.LagTimeLimitCorrelation,
                                    G_lim=self.ContrastLimitCorrelation)
        self.axis32.legend(['', ''])
        [self.CorrBright, self.CorrDark] = out
        # plot FCS
        # update timestamps and nanotimes
        mask = np.logical_and(self.timestamps >= self.TimeLimit[0],
                              self.timestamps <= self.TimeLimit[1])
        timestamps = self.timestamps[mask]
        result = PlotFCS(self.axis33, timestamps,
                         self.RangeFCS, self.BinsFCS)
        [self.bin_lags, self.G] = result
        return
    
    def Plot2Ddurations(self, RollMean = False,
                        range_on = [0, 1],
                        range_off = [0,1],
                        NumPointsForAveraging = 10,
                        shift_range = range(1, 10, 1)):
        range_on = self.Range2DHistBright
        range_off = self.Range2DHistDark
        NumPointsForAveraging = self.NumPointsForAveraging
        df_durations = self.df_durations
        if RollMean:
            df_roll_mean = df_durations.rolling(
                NumPointsForAveraging).mean().dropna().reset_index(drop=True)
            dfAverage = df_roll_mean
        else:
            df_mean = df_durations.groupby(np.arange(len(df_durations)) //
                                        NumPointsForAveraging
                                        ).mean().dropna().reset_index(drop=True)
            dfAverage = df_mean
        # set figure paramters
        self.Figure2Ddurations = plt.figure(figsize=(12, 5))
        nrows = 1
        ncols = 2
        self.axis00 = plt.subplot2grid((nrows, ncols), (0, 0), colspan=1)
        self.axis01 = plt.subplot2grid((nrows, ncols), (0, 1), colspan=1)
        T = np.array([])
        T_shift = np.array([])
        for i in shift_range:
            shift = i
            T = np.append(T, dfAverage['ontimes'][shift:].values)
            T_shift = np.append(T_shift, dfAverage['ontimes'][:-shift].values)

        self.axis00.hist2d(T, T_shift,
                           bins=40, range=([range_on, range_on]),
                           norm=mpl.colors.LogNorm())
        self.axis00.set_xlabel('t(n)')
        self.axis00.set_ylabel('t(n+{})'.format(i))
        self.axis00.text(0.1, 0.6, 'No. of points for averaging:{}'.format(
            NumPointsForAveraging))
        self.axis00.set_title('Bright times')

        for i in shift_range:
            shift = i
            T = np.append(T, dfAverage['offtimes'][shift:].values)
            T_shift = np.append(T_shift, dfAverage['offtimes'][:-shift].values)
        self.axis01.hist2d(T, T_shift,
                            bins=40, range=([range_off, range_off]),
                            norm=mpl.colors.LogNorm())
        self.axis01.set_xlabel('t(n)')
        self.axis01.set_ylabel('t(n+{})'.format(i))
        self.axis01.text(0.1, 0.6, 'No. of points for averaging:{}'.format(
            NumPointsForAveraging))
        self.axis01.set_title('Dark times')
        return

    def PlotTimeTrace(self, figsize=(10, 4)):
        # Define axis positions and numbers
        self.FigureTimetrace = plt.figure(figsize=figsize)
        nrows=1
        ncols= 1
        self.axis00 = plt.subplot2grid((nrows,ncols), (0,0), colspan=1)
        plot_timetrace(self.axis00, self.timestamps,
                        self.BintimeForIntTrace, color='b')
        self.axis00.set_xlim(self.TimeLimit)

    def PlotCorrelation(self):
        # Define axis positions and numbers
        self.FigureCorrelation = plt.figure(figsize=(6, 3))
        nrows=1
        ncols= 1
        self.axis00 = plt.subplot2grid((nrows,ncols), (0,0), colspan=1)
        
        df_durations = self.df_durations
        df_mean = df_durations.groupby(np.arange(len(df_durations))//
            self.NumPointsForAveraging).mean().dropna().reset_index(drop=True)
        df_roll_mean = df_durations.rolling(
            self.NumPointsForAveraging).mean().dropna().reset_index(drop=True)        
        dfAverage = df_durations
        self.CorrBright, self.CorrDark = CorrelationBrightDark(self.axis00,
                            dfAverage["ontimes"], dfAverage["offtimes"],
                            tlag_lim = self.LagTimeLimitCorrelation,
                            G_lim = self.ContrastLimitCorrelation)
    
    def PlotFCS(self):
        # update timestamps and nanotimes
        mask = np.logical_and(self.timestamps >= self.TimeLimit[0],
                                self.timestamps <= self.TimeLimit[1])
        timestamps = self.timestamps[mask]

        self.FigureCorrelation = plt.figure(figsize=(6, 3))
        nrows=1
        ncols= 1
        self.axis00 = plt.subplot2grid((nrows,ncols), (0,0), colspan=1)
        result = PlotFCS(self.axis00, timestamps,
                         self.RangeFCS, self.BinsFCS)
        [self.bin_lags, self.G] = result

    def LongTraceByParts(self,
        TimeWindow = 1e2, TimePeriod = 1e2,
        PhotonWindow = 1e4, PhotonPeriod = 1e4,
        by_photon=False):
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
                binned_trace = np.histogram(t_mic_temp, bins=100, range=(0, 8))
                t = binned_trace[1][:-1]
                n = binned_trace[0]
                df_lt['t'] = t
                df_lt[str(tleft)] = n / max(n)
                # FCS
                bin_lags = make_loglags(-5, 1, 20)
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

            # df_ts_temp = pd.DataFrame({str(tleft):t_mac_temp})
            # df_ts = pd.concat([df_ts_temp, df_ts], ignore_index=True, axis=1)
            # col_names.append(str(tright))
            # print(col_names)
            df_ts.columns = col_names
        self.df_ts = df_ts
        self.df_lt = df_lt
        self.df_fcs = df_fcs
        self.df_ip = df_ip
        return df_ts, df_lt, df_fcs, df_ip

    def PlotLongTraceByParts(self):
        try:
            print('looking for df_ts',len(self.df_ts))
        except:
            self.LongTraceByParts()
        df_ts = self.df_ts
        df_lt = self.df_lt
        df_fcs = self.df_fcs
        df_ip = self.df_ip
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
        columns = df_ts.columns
        colorList = []
        for column in columns:
            color = cmap(float(i)/N)
            colorList.append(color)
            i=i+1
            #plot all
            plot_timetrace(self.axis00, df_ts[column],
                           bintime = self.BintimeForIntTrace, color=color)
            #plot individual
            self.axis20.plot(df_fcs.iloc[:, 0], df_fcs[column], color=color)
            self.axis21.plot(df_lt.iloc[:, 0], df_lt[column], color=color)
            # self.axis20.plot(df_ip.iloc[:, 0], df_ip[column],color=color)
        # Zoomed in trace-1
        SelectColumn = columns[0]
        SelectColor = colorList[0]
        self.axis00.axvspan(min(df_ts[SelectColumn]),
                            max(df_ts[SelectColumn]),
                            color=SelectColor, alpha=0.3, lw=0)
        plot_timetrace(self.axis10, df_ts[SelectColumn],
                       bintime = self.BintimeForIntTrace, color=SelectColor)
        self.axis10.set_ylim(0, None)
        self.axis10.set_xlim(min(df_ts[SelectColumn]),
                             max(df_ts[SelectColumn]))
        self.axis10.legend(['Highlighted part of the trace'])
        # Zoomed in trace-2
        SelectColumn = columns[-1]
        SelectColor = colorList[-1]
        self.axis00.axvspan(min(df_ts[SelectColumn]),
                            max(df_ts[SelectColumn]),
                            color=SelectColor, alpha=0.3, lw=0)
        self.axis00.set_ylim(0, None)
        self.axis00.set_xlim(self.TimeLimit[0], max(df_ts[column]))

        plot_timetrace(self.axis11, df_ts[SelectColumn],
                       bintime = self.BintimeForIntTrace, color=SelectColor)
        self.axis11.set_ylim(0, None)
        self.axis11.set_xlim(min(df_ts[SelectColumn]),
                             max(df_ts[SelectColumn]))
        self.axis11.legend(['Highlighted part of the trace'])

        self.axis21.set_xlim(2, 8)
        self.axis21.set_ylim(1e-1, None)
        self.axis21.set_yscale('log')
        self.axis21.set_xlabel('Lifetime/ns')
        self.axis21.set_ylabel('#')
        self.axis21.text(3, 0.15, 'Life time histogram', style='italic')
        self.axis20.set_xlim(1e-5, 1)
        self.axis20.set_ylim(0, 4)
        self.axis20.set_xscale('log')
        self.axis20.set_xlabel('lag time/s')
        self.axis20.set_ylabel('G(t)-1')
        self.axis20.text(2e-5, 1,'FCS')
    
    def StatsLongTraceByParts(self):
        try:
            print('looking for df_ts',len(self.df_ts))
        except:
            self.LongTraceByParts()
        df_ts = self.df_ts
        df_lt = self.df_lt
        df_fcs = self.df_fcs
        df_ip = self.df_ip

        lag_time = df_fcs.iloc[:, 0]
        cols_fcs = ['A1', 'A1_err', 't_ac1', 't_ac1_err',
                    'ton1', 'ton1_err', 'toff1', 'toff1_err']
        df_fcs_fit = pd.DataFrame(columns=cols_fcs)
        cols_lt = ['ampl1', 'tau1', 'ampl2', 'tau2', 'baseline']
        cols_lt = cols_lt + [s + 'err' for s in cols_lt]
        df_lt_fit = pd.DataFrame(columns=cols_lt)
        for column in df_ts.columns:
            # fcs fit
            G = df_fcs[column]
            result_fcs = t_on_off_fromFCS(lag_time, G,
                                tmin=1e-5, tmax=1.0e0,
                                fitype='mono_exp')
            df_fcs_fit.loc[column] = list(result_fcs.values())
            # lifetime fit
            time_ns = df_lt['t'].values
            decay = df_lt[column].values
            result_lt = LifeTimeFitTail(time_ns, decay,
                                offset=None, model='biexp')
            [fit_res, time_ns_tail, decay_hist_tail] = result_lt
            params = fit_res.params
            fit_values = {k: v.value for k, v in params.items()}
            fit_errs = {k: v.stderr for k, v in params.items()}
            df_lt_fit.loc[column] = list(fit_values.values()) +\
                                    list(fit_errs.values())#append to rows
        self.df_fcs_fit = df_fcs_fit
        self.df_lt_fit = df_lt_fit
        return

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

def PlotFCS(axis, timestamps, RangeFCS, BinsFCS):
    bin_lags = make_loglags(RangeFCS[0], RangeFCS[1], BinsFCS)
    G = normalize_G(timestamps, timestamps, bin_lags)
    G = np.hstack((G[:1], G)) - 1
    axis.plot(bin_lags, G)
    axis.set_xlabel('lag time/s')
    axis.set_ylabel('G(0)-1')
    axis.set_xscale('log')
    axis.set_xlim(min(bin_lags))
    return bin_lags, G

def waitime_hist_inset(waitimes, axis, bins, binrange,
                       insetrange, fit_func, PlotInset,
                       color='b', barPlot=False):
    '''waitimes: list of on-times or off-times
    '''
    n,bins_hist = np.histogram(waitimes, bins=bins,
                            range=(binrange[0] , binrange[1]))#avoiding zero
    t=bins_hist[:-1]; n = n[:]
    t_fit = np.linspace(binrange[0], binrange[1], 1000)
    binwidth = np.mean(np.diff(t))
    #fit
    if fit_func.__code__.co_code == mono_exp.__code__.co_code:
        p0 = [10,1.1]
    elif fit_func.__code__.co_code == risetime_fit.__code__.co_code:
        p0=[10,1.1, 0.1]
    elif fit_func.__code__.co_code == streched_exp.__code__.co_code:
        p0=[10,0.5, 1000]        
    fit, pcov = curve_fit(fit_func, t, n, p0=p0, bounds=(0, np.inf))
    print('k1:'+str(fit[0]))
    print(fit)
    #plot as bar
    from matplotlib import pyplot, transforms
    rot = transforms.Affine2D().rotate_deg(90)
    if barPlot:
        axis.bar(t, n, width=binwidth, color='k', alpha=0.5)
    else:
        # axis.plot(t, n, 'o', color=color)
        axis.plot(t, n, drawstyle='steps-mid', lw=0.5, color=color)
    axis.plot(t_fit, fit_func(t_fit, *fit), 'k',
        lw=1,label='k1:'+str(fit[0])+'\n'+str(fit[1]))
    axis.set_xlim(0, None)
    axis.set_ylim(1e0, None)
    axis.set_yscale('log')
    #inset
    if PlotInset:
        axis_in = inset_axes(axis, height="50%", width="50%")
        if barPlot:
            axis_in.bar(t, n, width=binwidth, color='k', alpha=0.5)
            axis_in.plot(t, n, drawstyle='steps-mid', lw=0.5, color='k')
        else:
            axis_in.plot(t, n, 'o', color=color)
        axis_in.plot(t_fit, fit_func(t_fit, *fit), 'k',
            lw=1,label='k1:'+str(fit[0])+'\n'+str(fit[1]))
        axis_in.set_xlim(insetrange)
        axis_in.get_yaxis().set_ticklabels([])
    result = {'TimeBins': t,
            'PDF': n}
    return 

def MidPointPotentialTimeTrace(axis, t_av_on, t_av_off,
    t_abs, AppliedPotential, E0range=[None, None], TimeLimit = [None, None]):
    t_on_ratio = t_av_off/t_av_on
    E0_list = AppliedPotential - 59*np.log10(t_on_ratio)
    axis.plot(t_abs, E0_list, label=r'$E_0/mV$')
    axis.set_xlim(TimeLimit)
    axis.set_ylim(E0range)
    axis.set_ylabel('E_0/mV')
    axis.set_xlabel('time/s')
    axis.legend()
    # axis.yaxis.tick_right()
    return E0_list

def MidPointPotentialGaussFit(axis, E0_list, bins, E0range):
    E0Centers = np.linspace(E0range[0], E0range[1], bins)
    N, binList = np.histogram(E0_list, bins=bins, range=E0range)
    gaussModel = lmfit.Model(gaussian)
    params = gaussModel.make_params(amp=100, cen=50, sig=30)
    params['sig'].min = 0
    result = gaussModel.fit(N, params, x=E0Centers)

    binWidth = np.mean(np.diff(E0Centers))
    print(binWidth)
    axis.plot(N, E0Centers, '*m')
    axis.plot(gaussian(E0Centers, *result.values.values()),
              E0Centers, '--k')
    E0 = int(result.values['cen'])
    E0fwhm = int(2.35482 * result.values['sig'])
    E0Amp = int(result.values['amp'])
    print(E0Amp)
    axis.text(E0Amp / 3, E0 + 40,
              'E0: {} mV\nfwhm: {} mV'.format(E0, E0fwhm))
    axis.set_xlabel('#')
    # axis.set_ylabel('Potential/mV')
    axis.set_ylim(E0range)
    return result


def AverageDurationsLogNormalFit(axis, AvgDurations,
                                 Bins, BinRange,color='b',
                                 Plotting=False, PlotFit=False):
    BinCenters = np.linspace(BinRange[0], BinRange[1], Bins)
    N, binList = np.histogram(AvgDurations, bins=Bins, range=BinRange)
    mask = N != 0
    N = N[mask]
    BinCenters = BinCenters[mask]
    logNormalModel = lmfit.Model(logNormal)
    params = logNormalModel.make_params(amp=100, cen=0.1, sig=1)
    params['sig'].min = 0.1
    result = logNormalModel.fit(N, params, x=BinCenters)
    Sigma = np.round(result.values['sig'], 2)
    Center = result.values['cen']
    Mean = np.round(np.exp(Center + (Sigma**2 / 2)), 2)
    Amplitude = int(result.values['amp'])
    # print(Mean)
    if Plotting:
        axis.plot(N, BinCenters, '*', color=color)
        if PlotFit:
            axis.plot(logNormal(BinCenters, *result.values.values()),
                    BinCenters, '--k')
            axis.text(Amplitude, 2 * Mean,
                    'Mean: {} s\nSigma: {}'.format(Mean, Sigma))
        axis.set_xlabel('#')
        # axis.set_ylabel('Potential/mV')
        axis.set_ylim(BinRange)
    return result


def CorrelationBrightDark(axis, t_av_on, t_av_off, tlag_lim, G_lim):
    G_on = []
    G_off = []
    m=(len(t_av_on)/20)-5
    t_av_tot = np.average(t_av_on)+np.average(t_av_off)
    # On-time correlation
    G_on = autocorrelate(t_av_on[:-1], m=m ,deltat=1,normalize=True)
    axis.plot(G_on[:,0]*t_av_tot, G_on[:,1], 'b',label='On time correlation')
    ymax_lim_on = G_on[0,1]
    # off-time correlation-------
    m=(len(t_av_off)/2)-5
    G_off = autocorrelate(t_av_off[:-1], m=m ,deltat=1,normalize=True)
    axis.plot(G_off[:,0]*t_av_tot, G_off[:,1], 'r',label='Off time correlation')
    # axis settings
    axis.set_xscale('log')
    axis.set_xlabel('time/s')
    axis.set_ylabel('G(t)')
    ymax_lim_off = G_off[0,1]
    if ymax_lim_on > ymax_lim_off:
        ymax_lim = ymax_lim_on
    else:
        ymax_lim = ymax_lim_off
    axis.legend()
    axis.set_ylim(G_lim)#+0.01*ymax_lim
    axis.set_xlim(tlag_lim)       
    return G_on, G_off

def Plot2Ddurations(axis, df_durations, RollMean = True,
                    NumPointsForAveraging = 10,
                    shift_range = range(1, 10, 1),
                    ontimes = True,
                    bins=40, rangehist=[0, 1]):
    if RollMean:
        df_roll_mean = df_durations.rolling(
            NumPointsForAveraging).mean().dropna().reset_index(drop=True)        
        dfAverage = df_roll_mean
    else:
        df_mean = df_durations.groupby(np.arange(len(df_durations))//
            NumPointsForAveraging).mean().dropna().reset_index(drop=True)
        dfAverage = df_mean
    T=np.array([]); T_shift = np.array([])
    for i in shift_range:
        shift = i  
        if ontimes:
            T = np.append(T, dfAverage['ontimes'][shift:].values)
            T_shift = np.append(T_shift, dfAverage['ontimes'][:-shift].values)
        else:
            T = np.append(T, dfAverage['offtimes'][shift:].values)
            T_shift = np.append(T_shift, dfAverage['offtimes'][:-shift].values)            
        
    axis.hist2d(T, T_shift,
                bins=bins, range=([rangehist, rangehist]),
               norm=mpl.colors.LogNorm())
    axis.set_xlabel('t(n)')
    axis.set_ylabel('t(n+{})'.format(i))
    # axis.text(0.1, 0.6, 'No. of points for averaging:{}'.format(
    #     NumPointsForAveraging))
    return axis

def FCSCorrTimeAmp2D(ObjLongTraceMeasured, ObjLongTraceSimulated,
                     RangeCorrTime=[0, .4],
                     RangeCorrAmplitude=[1.5, 3.5]):
    # definitions for the axes
    left, width = 0.1, 0.55
    bottom, height = 0.1, 0.55
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.3]
    rect_histy = [left_h, bottom, 0.3, height]

    # start with a rectangular Figure
    fig = plt.figure(1, figsize=(8, 8))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    from matplotlib.ticker import NullFormatter
    nullfmt = NullFormatter()         # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    histtype = 'step'

    x = ObjLongTraceMeasured.df_fcs_fit['t_ac1'].values
    y = ObjLongTraceMeasured.df_fcs_fit['A1'].values
    colorData = 'r'
    colorscatt = np.array(ObjLongTraceMeasured.df_fcs_fit.index).astype('int')

    xSim = ObjLongTraceSimulated.df_fcs_fit['t_ac1'].values
    ySim = ObjLongTraceSimulated.df_fcs_fit['A1'].values
    colorSim = 'b'
    colorscattSim = np.array(
        ObjLongTraceSimulated.df_fcs_fit.index).astype('int')
    cmap = 'jet'
    bins = 80

    axScatter.scatter(x, y, marker='*', c=colorscatt, cmap=cmap)
    axHistx.hist(x, bins=bins, range=RangeCorrTime,
                 histtype=histtype, color=colorData)
    axHisty.hist(y, bins=bins, range=RangeCorrAmplitude,
                 orientation='horizontal', histtype=histtype, color=colorData)

    cax = axScatter.scatter(xSim, ySim, marker='.', c=colorscattSim, cmap=cmap)
    axHistx.hist(xSim, bins=bins, range=RangeCorrTime,
                 histtype=histtype, color=colorSim)
    axHisty.hist(ySim, bins=bins, range=RangeCorrAmplitude,
                 orientation='horizontal', histtype=histtype, color=colorSim)
    axScatter.set_xlim(RangeCorrTime)
    axScatter.set_ylim(RangeCorrAmplitude)
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())
    axScatter.set_xlabel('Correlation time/s')
    axScatter.set_ylabel('G(0)-1')
    plt.colorbar(cax)
    return
#===============fitting functions=============
def risetime_fit(t, k1, k2, A):
    return ((A*k1*k2/(k2-k1)) * (np.exp(-k1*t) - np.exp(-k2*t)))

def mono_exp(t, k1, A):
    return A*np.exp(-k1*t)


def gaussian(x, amp, cen, sig):
    return amp * np.exp(-(x - cen)**2 / sig**2)


def logNormal(x, amp, cen, sig):
    return (amp / (sig * 2 * np.pi * x)) * np.exp(
            -(np.log(x) - cen)**2 / (2 * sig**2))


def streched_exp(t, k, b, A):
    return A*np.exp(-(k*t)**b)
    
