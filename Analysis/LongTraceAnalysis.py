import os
import numpy as np
import scipy
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
from simulation import simulate_on_off_times, save_simtrace_longtrace

SimulatedFCStimes = '/home/biswajit/Research/reports-PhD/AzurinSM-MS4/Azurin_SM_repo/Analysis/SimulatedFCSOnOffTimes.xlsx'
class LongTraceClass(object):
    """docstring for LongTraceClass"""
    parameters = {"TimeLimit": [None, None],
                  "BintimeForIntTrace": 5e-3, # in seconds
                  "BinsForBrightHist": 20,
                  "RangeForBrightHist": [0.01, 2], # in seconds
                  "BinsForDarkHist": 20,
                  "RangeForDarkHist": [0.01, 10], # in seconds
                  "PlotInsetForDurationHist": False,
                  "InsetRangeBrightHist": [0, 0.05], # in seconds
                  "InsetRangeDarkHist": [0, 0.1], # in seconds
                  "AveragingType": 'mean',  # 'rolling' or 'mean' or 'noAveraging'
                  "NumPointsForAveraging": 10,
                  "SimulationTimeLength": 100000,  # in seconds
                  "BinsForAvgBrightHist": 50,
                  "BinsForAvgDarkHist": 50,
                  "Range2DHistBright": [0, 1], # in seconds
                  "Range2DHistDark": [0, 5], # in seconds
                  "RangeForMidPotentialHist": [0, 125], # in mV
                  "BinsForMidPotentialHist": 50,
                  "LagTimeLimitCorrelation": [10, 1000], # in seconds
                  "ContrastLimitCorrelation": (None, None),
                  "BintimeForCorrelation": 1e-2, # in seconds
                  "FigureSize": (12, 12),
                  "ChangePointParams": (1, 0.01, 0.99, 2),
                  "BinsFCS": 10,
                  "RangeFCS": [-5, 2], # exponent of 10 returns value in seconds
                  "BinsLifetime": 50,
                  "RangeLifetime": [0, 8], # in nano-seconds
                  "FigureTightLayout": False,
                  "Background": 200, #counts per second
                 }

    def __init__(self, file_path_hdf5, SimulatedHDF5,
                 Simulation=False, parameters=parameters):

        print('Input file is {}'.format(os.path.basename(file_path_hdf5)))
        self.Simulation = Simulation
        self.file_path_hdf5 = file_path_hdf5
        self.analysis_hdf5 = file_path_hdf5[:-4] + 'analysis.hdf5'
        # self.SimulatedHDF5 = SimulatedHDF5
        if self.Simulation:
            self.file_path_hdf5 = SimulatedHDF5
            self.SimulatedHDF5 = SimulatedHDF5
        else:
            self.SimulatedHDF5 = save_simtrace_longtrace(
                                        FilePathHdf5=file_path_hdf5,
                                        lifetime_on=3.8e-9,
                                        lifetime_off=0.6e-9,
                                        rewrite=False)            
        for key in parameters:
            setattr(self, key, parameters[key])
        self.FilePathYaml = self.file_path_hdf5[:-5] + '.yaml'
        with open(self.FilePathYaml) as f:
            dfyaml = yaml.load(f)
        tmin = dfyaml['TimeLimit']['MinTime']
        tmax = dfyaml['TimeLimit']['MaxTime']
        self.TimeLimit = [tmin, tmax]
        self.AppliedPotential = dfyaml['Potential']['Value']  # Unit: mV
        self.Background = dfyaml['Background']
        if self.Simulation:
            h5 = h5py.File(self.file_path_hdf5, 'r')
            self.timestamps = h5['onexp_offexp']['timestamps'][...]
            self.nanotimes = h5['onexp_offexp']['nanotimes'][...]
            h5.close()
        else:
            h5 = h5py.File(self.file_path_hdf5, 'r')
            self.unit = h5['photon_data']['timestamps_specs']['timestamps_unit'][...]
            self.tcspc_unit = h5['photon_data']['nanotimes_specs']['tcspc_unit'][...]
            self.timestamps = self.unit * h5['photon_data']['timestamps'][...]
            self.nanotimes = self.tcspc_unit * \
                h5['photon_data']['nanotimes'][...]
            h5.close()
        mask = np.logical_and(self.timestamps >= self.TimeLimit[0],
                              self.timestamps <= self.TimeLimit[1])
        self.timestamps = self.timestamps[mask]
        self.nanotimes = self.nanotimes[mask]
        # update min and max time limit
        self.TimeLimit[0] = min(self.timestamps)
        self.TimeLimit[1] = max(self.timestamps)
        h5 = h5py.File(self.SimulatedHDF5, 'r')
        self.timestamps_sim = h5['onexp_offexp']['timestamps'][...]
        self.nanotimes_sim = h5['onexp_offexp']['nanotimes'][...]
        h5.close()


    def PlotFCS(self):
        # update timestamps and nanotimes
        mask = np.logical_and(self.timestamps >= self.TimeLimit[0],
                              self.timestamps <= self.TimeLimit[1])
        timestamps = self.timestamps[mask]

        self.FigureCorrelation = plt.figure(figsize=(6, 3))
        nrows = 1
        ncols = 1
        self.axis00 = plt.subplot2grid((nrows, ncols), (0, 0), colspan=1)
        result = PlotFCS(self.axis00, timestamps,
                         self.RangeFCS, self.BinsFCS)
        [self.bin_lags, self.G] = result
        if self.FigureTightLayout:
            self.FigureCorrelation.tight_layout()

    def LongTraceByParts(self,
                         TimeWindow=1e2, TimePeriod=1e2,
                         PhotonWindow=1e4, PhotonPeriod=1e4,
                         by_photon=False):
        '''
        Arguments:
        timestamps and nanotimes should be of equal length and
        both in the units of seconds
        window and period in number of photons
        by_photon: by defaul it is false and it devides the trace by time.
        If true, it will devide the trace by the number of photons
        '''
        self.TimeWindow = TimeWindow
        self.TimePeriod = TimePeriod
        self.PhotonWindow = PhotonWindow
        self.PhotonPeriod = PhotonPeriod

        out = long_trace_byparts(self.timestamps, self.nanotimes,
                                 self.TimeLimit,
                                 self.TimeWindow, self.TimePeriod,
                                 self.PhotonWindow, self.PhotonPeriod,
                                 self.BinsLifetime, self.RangeLifetime)
        [df_ts, df_lt, df_fcs, df_ip] = out
        self.df_ts = df_ts
        self.df_lt = df_lt
        self.df_fcs = df_fcs
        self.df_ip = df_ip
        out = long_trace_byparts(self.timestamps_sim, self.nanotimes_sim,
                                 self.TimeLimit,
                                 self.TimeWindow, self.TimePeriod,
                                 self.PhotonWindow, self.PhotonPeriod,
                                 self.BinsLifetime, self.RangeLifetime)
        [df_ts, df_lt, df_fcs, df_ip] = out
        self.df_ts_sim = df_ts
        self.df_lt_sim = df_lt
        self.df_fcs_sim = df_fcs
        self.df_ip_sim = df_ip
        return df_ts, df_lt, df_fcs, df_ip

    def StatsLongTraceByParts(self):
        try:
            print('looking for df_ts', len(self.df_ts))
        except:
            self.LongTraceByParts()
        df_ts = self.df_ts
        df_lt = self.df_lt
        df_fcs = self.df_fcs
        df_ip = self.df_ip
        if self.Simulation:
            out = stats_long_trace_byparts(self.df_ts, self.df_lt,
                                self.df_fcs, self.df_ip,
                                bg=self.Background, from_cp_values=False)
        else:
            h5_analysis = h5py.File(self.analysis_hdf5, 'r')
            df = h5_analysis['changepoint']['cp_0.01_0.99_100s'][...]
            cp_values = pd.DataFrame(df, columns=['cp_index', 'cp_ts', 'cp_state', 'cp_countrate']) # a dataframe
            h5_analysis.close()
            out = stats_long_trace_byparts(self.df_ts, self.df_lt,
                                        self.df_fcs, self.df_ip,
                                        bg=self.Background,
                                        from_cp_values=True, cp_values=cp_values)
        [self.df_fcs_fit, self.df_lt_fit] = out
        Potential = self.AppliedPotential
        tons = self.df_fcs_fit['ton1'].values.astype('float')
        toffs = self.df_fcs_fit['toff1'].values.astype('float')
        self.df_fcs_fit['E0'] = Potential - 59 * np.log10(toffs / tons)
        out = stats_long_trace_byparts(self.df_ts_sim, self.df_lt_sim,
                                       self.df_fcs_sim, self.df_ip_sim,
                                       self.Background)
        [self.df_fcs_fit_sim, self.df_lt_fit_sim] = out
        return

    def PlotLongTraceByParts(self, RangeBrightTime, RangeDarkTime,
                             SimulatedFCStimes=SimulatedFCStimes,
                             indexRangeAveraging=[None, None]):
        try:
            print('looking for df_fcs_fit', len(self.df_fcs_fit))
        except:
            self.StatsLongTraceByParts()
        df_ts = self.df_ts
        df_lt = self.df_lt
        df_fcs = self.df_fcs
        df_ip = self.df_ip
        nrows = 1 + 3
        ncols = 2
        self.FigureByParts = plt.figure(figsize=(10, 10))
        self.axis00 = plt.subplot2grid((nrows, ncols), (0, 0), colspan=2)
        self.axis10 = plt.subplot2grid((nrows, ncols), (1, 0))
        self.axis11 = plt.subplot2grid((nrows, ncols), (1, 1))
        self.axis20 = plt.subplot2grid((nrows, ncols), (2, 0), rowspan=2)
        self.axis21 = plt.subplot2grid((nrows, ncols), (2, 1))
        self.axis31 = plt.subplot2grid((nrows, ncols), (3, 1))

        cmap = plt.get_cmap('jet')  # jet_r
        N = len(df_ts.columns)
        i = 1
        columns = df_ts.columns
        colorList = []
        for column in columns:
            color = cmap(float(i) / N)
            colorList.append(color)
            i = i + 1
            #plot all
            plot_timetrace(self.axis00, df_ts[column],
                           bintime=self.BintimeForIntTrace, color=color)
            #plot individual
            self.axis21.plot(df_fcs.iloc[:, 0], df_fcs[column], color=color)
            self.axis31.plot(df_lt.iloc[:, 0], df_lt[column], color=color)
            # self.axis21.plot(df_ip.iloc[:, 0], df_ip[column],color=color)
        # Zoomed in trace-1
        SelectColumn = columns[0]
        SelectColor = colorList[0]
        self.axis00.axvspan(min(df_ts[SelectColumn]),
                            max(df_ts[SelectColumn]),
                            color=SelectColor, alpha=0.3, lw=0)
        plot_timetrace(self.axis10, df_ts[SelectColumn],
                       bintime=self.BintimeForIntTrace, color=SelectColor)
        self.axis10.set_ylim(0, None)
        self.axis10.set_xlim(min(df_ts[SelectColumn]),
                             max(df_ts[SelectColumn]))
        # Zoomed in trace-2
        SelectColumn = columns[-1]
        SelectColor = colorList[-1]
        self.axis00.axvspan(min(df_ts[SelectColumn]),
                            max(df_ts[SelectColumn]),
                            color=SelectColor, alpha=0.3, lw=0)
        self.axis00.set_ylim(0, None)
        self.axis00.set_xlim(self.TimeLimit[0], max(df_ts[column]))

        plot_timetrace(self.axis11, df_ts[SelectColumn],
                       bintime=self.BintimeForIntTrace, color=SelectColor)
        self.axis11.set_ylim(0, None)
        self.axis11.set_xlim(min(df_ts[SelectColumn]),
                             max(df_ts[SelectColumn]))

        self.axis31.set_xlim(2, 8)
        self.axis31.set_ylim(1e-1, None)
        self.axis31.set_yscale('log')
        self.axis31.set_xlabel('Lifetime/ns')
        self.axis31.set_ylabel('#')
        # self.axis31.text(3, 0.15, 'Life time histogram', style='italic')
        self.axis21.set_xlim(1e-5, 1)
        self.axis21.set_ylim(0, 4)
        self.axis21.set_xscale('log')
        self.axis21.set_xlabel('lag time/s')
        self.axis21.set_ylabel('G(t)-1')
        self.axis21.text(2e-5, 1, 'FCS')
        # plot 2D scatter
        x = self.df_fcs_fit['ton1'].values.astype('float')
        y = self.df_fcs_fit['toff1'].values.astype('float')
        x_avg = np.round(np.average(x[indexRangeAveraging[0]:indexRangeAveraging[1]]), 3)
        x_std = np.round(np.std(x), 2)
        y_avg = np.round(np.average(y[indexRangeAveraging[0]:indexRangeAveraging[1]]), 3)
        y_std = np.round(np.std(y), 2)
        print("Average bright time:{}; std:{}\n Average dark time:{}; std:{}".format(x_avg,x_std, y_avg, y_std))
        # xSim = self.df_fcs_fit_sim['ton1'].values.astype('float')
        # ySim = self.df_fcs_fit_sim['toff1'].values.astype('float')
        colorscatt = np.array(self.df_fcs_fit.index).astype('int')        
        # colorscattSim = np.array(self.df_fcs_fit.index).astype('int')
        cmap = 'jet'
        # simulated FCS times
        df = pd.read_excel(SimulatedFCStimes)
        xSim = df['ton1'].values
        xSim = x_avg * xSim/np.average(xSim)
        ySim = df['toff1'].values
        ySim = y_avg * ySim/np.average(ySim)
        cax = self.axis20.scatter(xSim, ySim, marker='o', c='k', alpha=0.2)
        # Real FCS times
        self.axis20.scatter(x, y, marker='v', s=200, facecolors='none',
                            c=colorList, edgecolor='k', cmap=cmap) #colorscatt
        self.axis20.set_xlim(RangeBrightTime)
        self.axis20.set_ylim(RangeDarkTime)
        self.axis20.set_xlabel('Bright times/s')
        self.axis20.set_ylabel('Dark times/s')

        # Add Figures (axis) numbers
        axis_list = self.FigureByParts.axes
        title_list = ['A', 'B', 'C', 'F', 'D', 'E']
        for index, axis in enumerate(axis_list):
            axis.text(.9, .9, title_list[index], horizontalalignment='center',
                      transform=axis.transAxes)
        if self.FigureTightLayout:
            self.FigureByParts.tight_layout()

    def DigitizePhotons(self, int_photon=False,
                        nanotimes_bool=False,
                        duration_cp=False):
        if self.Simulation:
            df_dig = digitize_photonstamps(file_path_hdf5=self.file_path_hdf5,
                                           pars=self.ChangePointParams,
                                           time_sect=100,
                                           Simulated=True,
                                           time_lim=self.TimeLimit,
                                           bintime=self.BintimeForIntTrace,
                                           cp_no=True,
                                           int_photon=int_photon,
                                           nanotimes_bool=nanotimes_bool,
                                           duration_cp=duration_cp)
        else:
            df_dig = digitize_photonstamps(file_path_hdf5=self.file_path_hdf5,
                                           pars=self.ChangePointParams,
                                           time_sect=100,
                                           time_lim=self.TimeLimit,
                                           bintime=self.BintimeForIntTrace,
                                           cp_no=True,
                                           int_photon=int_photon,
                                           nanotimes_bool=nanotimes_bool,
                                           duration_cp=duration_cp)
        return df_dig

    def onoff_values(self):
        df_dig = self.DigitizePhotons()
        df_on = df_dig[df_dig['state'] == 2].reset_index(drop=True)
        df_off = df_dig[df_dig['state'] == 1].reset_index(drop=True)
        time_left = df_on.groupby('cp_no').timestamps.min()
        time_right = df_on.groupby('cp_no').timestamps.max()
        abstime_on = df_on.groupby('cp_no').timestamps.mean()
        ontimes = time_right - time_left

        time_left = df_off.groupby('cp_no').timestamps.min()
        time_right = df_off.groupby('cp_no').timestamps.max()
        abstime_off = df_off.groupby('cp_no').timestamps.mean()
        offtimes = time_right - time_left
        l = len(abstime_on) - 2
        abstime_on = abstime_on.values[:l]
        ontimes = ontimes.values[:l]
        abstime_off = abstime_off.values[:l]
        offtimes = offtimes.values[:l]
        self.df_durations = pd.DataFrame()
        self.df_durations['abstime_on'] = abstime_on
        self.df_durations['ontimes'] = ontimes
        self.df_durations['abstime_off'] = abstime_off
        self.df_durations['offtimes'] = offtimes
        # for simulated file
        out = simulate_on_off_times(ton1=np.average(ontimes), ton2=0.002,
                                    toff1=np.average(offtimes), toff2=0.005,
                                    time_len=self.SimulationTimeLength,
                                    plotting=False)
        [ontimes_exp_1, ontimes_exp_rise, offtimes_exp_1, offtimes_exp_rise] = out
        abstime = np.cumsum(ontimes_exp_1) + np.cumsum(offtimes_exp_1)
        self.dfDurationSimulated = pd.DataFrame({'abstime_on': abstime,
                                                 'ontimes': ontimes_exp_1,
                                                 'abstime_off': abstime,
                                                 'offtimes': offtimes_exp_1,
                                                 })
        return

    def PlotDurationsVsTime(self):
        # Define axis positions and numbers
        self.FigureDuration = plt.figure(figsize=self.FigureSize)
        nrows = 4
        ncols = 4
        self.axis00 = plt.subplot2grid((nrows, ncols), (0, 0), colspan=3)
        self.axis03 = plt.subplot2grid((nrows, ncols), (0, 3))
        self.axis10 = plt.subplot2grid((nrows, ncols), (1, 0), colspan=3)
        self.axis13 = plt.subplot2grid((nrows, ncols), (1, 3))
        self.axis20 = plt.subplot2grid((nrows, ncols), (2, 0), colspan=3)
        self.axis23 = plt.subplot2grid((nrows, ncols), (2, 3))
        self.axis30 = plt.subplot2grid((nrows, ncols), (3, 0))
        self.axis31 = plt.subplot2grid((nrows, ncols), (3, 1))
        self.axis32 = plt.subplot2grid((nrows, ncols), (3, 2))
        self.axis33 = plt.subplot2grid((nrows, ncols), (3, 3))
        # get Bright and Dark times
        df_durations = self.df_durations
        dfDurationSimulated = self.dfDurationSimulated
        if self.AveragingType == 'noAveraging':
            dfAverage = self.df_durations
            dfAverageSimulated = dfDurationSimulated
        elif self.AveragingType == 'mean':
            dfAverage = df_durations.groupby(np.arange(len(df_durations)) //
                                             self.NumPointsForAveraging).mean().dropna().reset_index(drop=True)
            dfAverageSimulated = dfDurationSimulated.groupby(np.arange(len(dfDurationSimulated)) //
                                                             self.NumPointsForAveraging).mean().dropna().reset_index(drop=True)
        elif self.AveragingType == 'rolling':
            dfAverage = df_durations.rolling(
                self.NumPointsForAveraging).mean().dropna().reset_index(drop=True)
            dfAverageSimulated = dfDurationSimulated.rolling(
                self.NumPointsForAveraging).mean().dropna().reset_index(drop=True)
        # MidPotential calculation
        TimeRatio = dfAverage['offtimes'] / dfAverage['ontimes']
        dfE0Average = pd.DataFrame()
        dfE0Average['E0'] = self.AppliedPotential - 59 * np.log10(TimeRatio)
        dfE0Average['abstime'] = dfAverage['abstime_on']
        TimeRatioSimulated = dfAverageSimulated['offtimes'] / \
            dfAverageSimulated['ontimes']
        dfE0AverageSimulated = pd.DataFrame()
        dfE0AverageSimulated['E0'] = self.AppliedPotential - \
            59 * np.log10(TimeRatioSimulated)
        dfE0AverageSimulated['abstime'] = dfAverageSimulated['abstime_on']
        # PLOTTING STARTS HERE #
        self.axis00.plot(dfAverage['abstime_on'], dfAverage['ontimes'], 'b')
        mask = np.logical_and(dfAverageSimulated['abstime_on'] > self.TimeLimit[0],
                              dfAverageSimulated['abstime_on'] < self.TimeLimit[1])
        self.axis00.plot(dfAverageSimulated['abstime_on'].values[mask],
                         dfAverageSimulated['ontimes'].values[mask],
                         'b', alpha=0.4)
        self.axis00.set_xlim(self.TimeLimit)
        self.axis00.set_ylabel(r'$Avg \tau_b/s$')
        self.axis00.set_xticklabels([])
        self.axis00.legend(['Average Bright times', '_'],
                           frameon=False, loc='upper left')
        ylimAxis00 = self.axis00.get_ylim()
        brightHist = np.histogram(dfAverage['ontimes'].values,
                                  bins=50,
                                  range=self.axis00.get_ylim(),
                                  density=True)
        self.axis03.plot(brightHist[0], brightHist[1][:-1], 'b')
        # plot simulated for comparision
        brightHistSim = np.histogram(dfAverageSimulated['ontimes'].values,
                                     bins=50,
                                     range=self.axis00.get_ylim(),
                                     density=True)
        # self.axis03.plot(brightHistSim[0], brightHistSim[1][:-1], '--b')
        self.axis03.fill(brightHistSim[0], brightHistSim[1][:-1],
                         'b', alpha=0.2)
        self.axis03.set_xlim(0, )
        self.axis03.set_yticklabels([])
        self.axis03.set_xticklabels([])

        self.axis10.plot(dfAverage['abstime_off'], dfAverage['offtimes'], 'r')
        mask = np.logical_and(dfAverageSimulated['abstime_off'] > self.TimeLimit[0],
                              dfAverageSimulated['abstime_off'] < self.TimeLimit[1])
        self.axis10.plot(dfAverageSimulated['abstime_off'].values[mask],
                         dfAverageSimulated['offtimes'].values[mask],
                         'r', alpha=0.4)
        self.axis10.set_xlim(self.TimeLimit)
        self.axis10.set_ylabel(r'$Avg \tau_d/s$')
        self.axis10.set_xticklabels([])
        self.axis10.legend(['Average Dark times', '_'],
                           frameon=False, loc='upper left')
        ylimAxis = self.axis10.get_ylim()
        darkHist = np.histogram(dfAverage['offtimes'].values,
                                bins=50,
                                range=self.axis10.get_ylim(),
                                density=True)
        self.axis13.plot(darkHist[0], darkHist[1][:-1], 'r')
        darkHistSim = np.histogram(dfAverageSimulated['offtimes'].values,
                                   bins=50,
                                   range=self.axis10.get_ylim(),
                                   density=True)
        # self.axis13.plot(darkHistSim[0], darkHistSim[1][:-1], '--r')
        self.axis13.fill(darkHistSim[0], darkHistSim[1][:-1], 'r', alpha=0.2)
        self.axis13.set_xlim(0, )
        self.axis13.set_yticklabels([])
        self.axis13.set_xticklabels([])

        self.axis20.plot(dfE0Average['abstime'], dfE0Average['E0'],
                         'm', label=r'$E_0/mV$')
        mask = np.logical_and(dfE0AverageSimulated['abstime'] > self.TimeLimit[0],
                              dfE0AverageSimulated['abstime'] < self.TimeLimit[1])
        self.axis20.plot(dfE0AverageSimulated['abstime'].values[mask],
                         dfE0AverageSimulated['E0'].values[mask],
                         'm', alpha=0.2, label='Simulated')
        self.axis20.set_xlabel('time/s')
        self.axis20.set_ylabel(r'$E_0/mV$')
        self.axis20.set_xlim(self.TimeLimit)
        self.axis20.legend([r'$E_0 / mV$', '_'],
                           frameon=False, loc='upper left')

        E0lim = self.axis20.get_ylim()
        E0Hist = np.histogram(dfE0Average['E0'].values,
                              bins=50,
                              range=self.axis20.get_ylim(),
                              density=True)
        self.axis23.plot(E0Hist[0], E0Hist[1][:-1], 'm')

        E0Hist = np.histogram(dfE0AverageSimulated['E0'].values,
                              bins=50,
                              range=self.axis20.get_ylim(),
                              density=True)
        # self.axis23.plot(E0Hist[0], E0Hist[1][:-1], '--m')
        self.axis23.fill(E0Hist[0], E0Hist[1][:-1], '--m', alpha=0.2)

        self.axis23.set_xlim(0, )
        self.axis23.set_yticklabels([])
        self.axis23.set_xticklabels([])
        self.axis23.set_xlabel('#')
        self.axis23.set_ylabel('')

        self.BrightStrechedFit = strechexp_lmfit(waitimes=df_durations['ontimes'],
                                     axis=self.axis30,
                                     bins=self.BinsForBrightHist,
                                     binrange=self.RangeForBrightHist,
                                     color='b', minimizer=True, barPlot=False)
        self.axis30.legend(['Bright times', '_fit'],
                           loc='upper left', framealpha=0.5)
        self.axis30.set_xlabel('time/s', color='b')
        self.axis30.tick_params('x', direction='in', colors='b')

        axis30_up = plt.twiny(ax=self.axis30)
        self.DarkStrechedFit = strechexp_lmfit(waitimes=df_durations['offtimes'],
                                     axis=axis30_up,
                                     bins=self.BinsForDarkHist,
                                     binrange=self.RangeForDarkHist,
                                     color='r', minimizer=True, barPlot=False)
        axis30_up.legend(['Dark times', '_fit'],
                         loc='lower left', framealpha=0.5)
        axis30_up.set_xlabel('')
        axis30_up.tick_params('x', direction='in', colors='r')
        self.axis30.set_yticklabels([])
        self.axis30.set_ylabel('log(PDF)')

        Plot2Ddurations(self.axis31, dfAverage,
                        shift_range=range(1, 10, 1),
                        ontimes=True,
                        bins=40, rangehist=self.Range2DHistBright)
        self.axis31.text(0.2 * self.Range2DHistBright[1],
                         0.8 * self.Range2DHistBright[1], 'Bright times')
        q05 = dfAverageSimulated.quantile(q=0.05)
        q95 = dfAverageSimulated.quantile(q=0.95)
        cent = 0.5 * (q05['ontimes'] + q95['ontimes'])
        rad = 0.5 * (q95['ontimes'] - q05['ontimes'])
        print("Circle for Bright time; radius={}, centre={}".format(rad, cent))
        circle1 = plt.Circle((cent, cent), rad, fill=False,
                             linestyle='-', edgecolor='white', linewidth=6)
        self.axis31.add_artist(circle1)
        circle1 = plt.Circle((cent, cent), rad, fill=False,
                             linestyle='-', edgecolor='k', linewidth=2)
        self.axis31.add_artist(circle1)

        Plot2Ddurations(self.axis32, dfAverage,
                        shift_range=range(1, 10, 1),
                        ontimes=False,
                        bins=40, rangehist=self.Range2DHistDark)
        self.axis32.text(0.2 * self.Range2DHistDark[1],
                         0.8 * self.Range2DHistDark[1],
                         'Dark times')
        q05 = dfAverageSimulated.quantile(q=0.05)
        q95 = dfAverageSimulated.quantile(q=0.95)
        cent = 0.5 * (q05['offtimes'] + q95['offtimes'])
        rad = 0.5 * (q95['offtimes'] - q05['offtimes'])
        print("Circle for Dark time; radius={}, centre={}".format(rad, cent))
        circle1 = plt.Circle((cent, cent), rad, fill=False,
                             linestyle='--', edgecolor='white', linewidth=6)
        self.axis32.add_artist(circle1)
        circle1 = plt.Circle((cent, cent), rad, fill=False,
                             linestyle='--', edgecolor='k', linewidth=2)
        self.axis32.add_artist(circle1)

        out = PlotCorrelationBrightDark(self.axis33,
                                        df_durations["ontimes"],
                                        df_durations["offtimes"],
                                        lagtime_lim=self.LagTimeLimitCorrelation,
                                        G_lim=self.ContrastLimitCorrelation,
                                        bintime=self.BintimeForCorrelation)
        [self.CorrBright, self.CorrDark] = out
        out = PlotCorrelationBrightDark(self.axis33,
                                        dfDurationSimulated["ontimes"][:2000],
                                        dfDurationSimulated["offtimes"][:2000],
                                        lagtime_lim=self.LagTimeLimitCorrelation,
                                        G_lim=self.ContrastLimitCorrelation,
                                        bintime=self.BintimeForCorrelation,
                                        colorBright='k', colorDark='k')
        self.axis33.legend(
            ['Bright', 'Dark', 'Simulated', '_'], loc='upper left')

        # Add Figures (axis) numbers
        axis_list = self.FigureDuration.axes
        title_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G',
                      'H', 'I', 'J', 'K', 'L']
        for index, axis in enumerate(axis_list):
            axis.text(.9, .9, title_list[index], horizontalalignment='center',
                      transform=axis.transAxes)
        if self.FigureTightLayout:
            self.FigureDuration.tight_layout()
        FileToSave = 'temp.xlsx'
        writer = pd.ExcelWriter(FileToSave)
        df_durations.to_excel(writer, 'Average', index=False)
        writer.save()
        return

    def Plot2Ddurations(self, RollMean=False,
                        range_on=[0, 1],
                        range_off=[0, 1],
                        NumPointsForAveraging=10,
                        shift_range=range(1, 10, 1)):
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
        if self.FigureTightLayout:
            self.Figure2Ddurations.tight_layout()
        return

    def PlotTimeTrace(self, figsize=(10, 4)):
        # Define axis positions and numbers
        self.FigureTimetrace = plt.figure(figsize=figsize)
        nrows = 1
        ncols = 1
        self.axis00 = plt.subplot2grid((nrows, ncols), (0, 0), colspan=1)
        plot_timetrace(self.axis00, self.timestamps,
                       self.BintimeForIntTrace, color='b')
        self.axis00.set_xlim(self.TimeLimit)
        if self.FigureTightLayout:
            self.FigureTimetrace.tight_layout()

    def PlotCorrelation(self):
        # Define axis positions and numbers
        self.FigureCorrelation = plt.figure(figsize=(6, 3))
        nrows = 1
        ncols = 1
        self.axis00 = plt.subplot2grid((nrows, ncols), (0, 0), colspan=1)

        df_durations = self.df_durations
        df_mean = df_durations.groupby(np.arange(len(df_durations)) //
                                       self.NumPointsForAveraging).mean().dropna().reset_index(drop=True)
        df_roll_mean = df_durations.rolling(
            self.NumPointsForAveraging).mean().dropna().reset_index(drop=True)
        dfAverage = df_durations
        self.CorrBright, self.CorrDark = CorrelationBrightDark(self.axis00,
                                                               dfAverage["ontimes"], dfAverage["offtimes"],
                                                               tlag_lim=self.LagTimeLimitCorrelation,
                                                               G_lim=self.ContrastLimitCorrelation)
        if self.FigureTightLayout:
            self.FigureCorrelation.tight_layout()


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
    n, bins_hist = np.histogram(waitimes, bins=bins,
                                range=(binrange[0], binrange[1]))  # avoiding zero
    t = bins_hist[:-1]
    n = n[:]
    t_fit = np.linspace(binrange[0], binrange[1], 1000)
    binwidth = np.mean(np.diff(t))
    #fit
    if fit_func.__code__.co_code == mono_exp.__code__.co_code:
        p0 = [10, 1.1]
    elif fit_func.__code__.co_code == risetime_fit.__code__.co_code:
        p0 = [10, 1.1, 0.1]
    elif fit_func.__code__.co_code == streched_exp.__code__.co_code:
        p0 = [10, 0.5, 1000]
    fit, pcov = curve_fit(fit_func, t, n, p0=p0, bounds=(0, np.inf))
    print('k1:' + str(fit[0]))
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
              lw=1, label='k1:' + str(fit[0]) + '\n' + str(fit[1]))
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
                     lw=1, label='k1:' + str(fit[0]) + '\n' + str(fit[1]))
        axis_in.set_xlim(insetrange)
        axis_in.get_yaxis().set_ticklabels([])
    result = {'TimeBins': t,
              'PDF': n}
    return


def strechexp_lmfit(waitimes, axis, bins, binrange,
                    color='b', minimizer=True, barPlot=False):
    '''waitimes: list of on-times or off-times
    '''
    def log_streched_exp(t, k, b, A):
        '''P = A*exp(-(kt)^b)
        log(P) = log(A)-(kt)^b'''
        return np.log(A) - (k * t)**b

    def streched_exp(t, k, b, A):
        return A * np.exp(-(k * t)**b)

    def residual_streched_exp(params, t, n, weights):
        k = params['k']
        b = params['b']
        A = params['A']
        return (n - (A * np.exp(-(k * t)**b))) * weights
    n, bins_hist = np.histogram(waitimes, bins=bins,
                                range=(binrange[0], binrange[1]))  # avoiding zero
    t = bins_hist[:-1]
    mask = n != 0
    n = n[mask]
    t = t[mask]
    t_fit = np.linspace(binrange[0], binrange[1], 1000)
    binwidth = np.mean(np.diff(t))
    # fit
    strechModel = lmfit.Model(log_streched_exp)
    params = lmfit.Parameters()
    params.add('k', 10, min=0)
    params.add('b', 0.9, min=0, max=1)
    params.add('A', 1, min=0)
    result = strechModel.fit(np.log(n), params, t=t)
    if minimizer:
        # minimize
        params = lmfit.Parameters()
        params.add('k', 10, min=0)
        params.add('b', 0.9, min=0, max=1)
        params.add('A', 1, min=0)
        weights = 1. / np.sqrt(n)
        result = lmfit.minimize(residual_streched_exp, params,
                                args=(t, n, weights),
                                method='leastsq')
    # extract parameters
    fit_report = {'k': result.params['k'].value,
                  'k_err': result.params['k'].stderr,
                  'b': result.params['b'].value,
                  'b_err': result.params['b'].stderr,
                  'A': result.params['A'].value,
                  'A_err': result.params['A'].stderr
                 }
    tau = ((1 / fit_report['k']) / fit_report['b']) * scipy.special.gamma(fit_report['b'])
    fit_report['tau'] = tau
    # print values
    print(fit_report)
    #plot as bar
    if barPlot:
        axis.bar(t, n, width=binwidth, color='k', alpha=0.5)
    else:
        axis.plot(t, n, 'o', color=color)
        axis.plot(t, n, drawstyle='steps-mid', lw=0.5, color=color)
    axis.plot(t_fit, streched_exp(t_fit, fit_report['k'],
                                  fit_report['b'],
                                  fit_report['A']),
              color=color, lw=1)  # , label='k1:' + result.values['k']
    axis.set_xlim(0, None)
    axis.set_ylim(1e0, None)
    axis.set_yscale('log')
    return fit_report


def MidPointPotentialTimeTrace(axis, t_av_on, t_av_off,
                               t_abs, AppliedPotential, E0range=[None, None], TimeLimit=[None, None]):
    t_on_ratio = t_av_off / t_av_on
    E0_list = AppliedPotential - 59 * np.log10(t_on_ratio)
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
                                 Bins, BinRange, color='b',
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


def correlation_durations(ontimes, offtimes,
                          bintime=1e-2, lagtime_lim=[1, 1e3]):
    if len(ontimes) < len(offtimes):
        l = len(ontimes)
    else:
        l = len(offtimes)
    ontimes = ontimes[:l]
    offtimes = offtimes[:l]
    OnPlusOff = ((ontimes + offtimes) /
                 bintime).astype('int')  # in millisecond
    ontimes_gen = np.array([])
    offtimes_gen = np.array([])
    for index, duration in enumerate(OnPlusOff):
        ontimes_i = ontimes[index]
        offtimes_i = offtimes[index]
        ontimes_gen = np.concatenate((ontimes_gen,
                                      np.ones(duration) * ontimes_i),
                                     axis=0)
        offtimes_gen = np.concatenate((offtimes_gen,
                                       np.ones(duration) * offtimes_i),
                                      axis=0)
    from autocorrelate import autocorrelate
    # m=(len(ontimes_gen)/20)-5
    G_on = autocorrelate(ontimes_gen, m=40, deltat=1, normalize=True)
    G_on[:, 0] = G_on[:, 0] * bintime
    mask = np.logical_and(G_on[:, 0] > lagtime_lim[0],
                          G_on[:, 0] < lagtime_lim[1])
    G_on = G_on[mask]
    G_off = autocorrelate(offtimes_gen, m=40, deltat=1, normalize=True)
    G_off[:, 0] = G_off[:, 0] * bintime
    mask = np.logical_and(G_off[:, 0] > lagtime_lim[0],
                          G_off[:, 0] < lagtime_lim[1])
    G_off = G_off[mask]
    return G_on, G_off


def PlotCorrelationBrightDark(axis, ontimes, offtimes,
                              lagtime_lim, G_lim, bintime=1e-2,
                              colorBright='b', colorDark='r'):
    G_on, G_off = correlation_durations(ontimes, offtimes,
                                        bintime=bintime,
                                        lagtime_lim=lagtime_lim)

    axis.plot(G_on[:, 0], G_on[:, 1], color=colorBright,
              label='On time correlation')
    axis.plot(G_off[:, 0], G_off[:, 1], color=colorDark,
              label='Off time correlation')
    # axis settings
    axis.set_xscale('log')
    axis.set_xlabel('time/s')
    axis.set_ylabel('G(t)')
    ymax_lim_on = G_on[0, 1]
    ymax_lim_off = G_off[0, 1]
    if ymax_lim_on > ymax_lim_off:
        ymax_lim = ymax_lim_on
    else:
        ymax_lim = ymax_lim_off
    axis.legend()
    axis.set_ylim(G_lim)  # +0.01*ymax_lim
    axis.set_xlim(lagtime_lim)
    return G_on, G_off


def Plot2Ddurations(axis, df_durations,
                    shift_range=range(1, 10, 1),
                    ontimes=True,
                    bins=40, rangehist=[0, 1]):
    T = np.array([])
    T_shift = np.array([])
    for i in shift_range:
        shift = i
        if ontimes:
            T = np.append(T, df_durations['ontimes'][shift:].values)
            T_shift = np.append(
                T_shift, df_durations['ontimes'][:-shift].values)
        else:
            T = np.append(T, df_durations['offtimes'][shift:].values)
            T_shift = np.append(
                T_shift, df_durations['offtimes'][:-shift].values)

    axis.hist2d(T, T_shift,
                bins=bins, range=([rangehist, rangehist]),
                norm=mpl.colors.LogNorm())
    axis.set_xlabel('t(n)')
    axis.set_ylabel('t(n+{})'.format(i))
    # axis.text(0.1, 0.6, 'No. of points for averaging:{}'.format(
    #     NumPointsForAveraging))
    return axis


def FCSCorrTimeAmp2D(ObjLongTraceMeasured,
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
    fig = plt.figure(1, figsize=(4, 4))

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

    xSim = ObjLongTraceMeasured.df_fcs_fit_sim['t_ac1'].values
    ySim = ObjLongTraceMeasured.df_fcs_fit_sim['A1'].values
    colorSim = 'b'
    colorscattSim = np.array(
        ObjLongTraceMeasured.df_fcs_fit_sim.index).astype('int')
    cmap = 'jet'
    bins = 15

    axScatter.scatter(x, y, marker='v', s=80, facecolors='none',
                      c=colorscatt, cmap=cmap)
    axHistx.hist(x, bins=bins, range=RangeCorrTime,
                 histtype=histtype, color=colorData)
    axHisty.hist(y, bins=bins, range=RangeCorrAmplitude,
                 orientation='horizontal', histtype=histtype, color=colorData)

    bins_sim = 80
    # , c=colorscattSim, cmap=cmap
    cax = axScatter.scatter(xSim, ySim, marker='.', c='k')
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
    # plt.colorbar(cax)
    return fig


def FCSBrightDark2D(ObjLongTraceMeasured,
                    RangeBrightTime=[0, 0.8],
                    RangeDarkTime=[0, 1.5]):
    # definitions for the axes
    # definitions for the axes
    left, width = 0.1, 0.55
    bottom, height = 0.1, 0.55
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.3]
    rect_histy = [left_h, bottom, 0.3, height]

    # start with a rectangular Figure
    fig = plt.figure(1, figsize=(4, 4))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    from matplotlib.ticker import NullFormatter
    nullfmt = NullFormatter()         # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    histtype = 'step'

    x = ObjLongTraceMeasured.df_fcs_fit['ton1'].values.astype('float')
    y = ObjLongTraceMeasured.df_fcs_fit['toff1'].values.astype('float')

    colorData = 'r'
    colorscatt = np.array(ObjLongTraceMeasured.df_fcs_fit.index).astype('int')

    xSim = ObjLongTraceMeasured.df_fcs_fit_sim['ton1'].values.astype('float')
    ySim = ObjLongTraceMeasured.df_fcs_fit_sim['toff1'].values.astype('float')
    colorSim = 'b'
    colorscattSim = np.array(
        ObjLongTraceMeasured.df_fcs_fit.index).astype('int')
    cmap = 'jet'
    bins = 15

    axScatter.scatter(x, y, marker='v', s=80, facecolors='none',
                      c=colorscatt, cmap=cmap)
    axHistx.hist(x, bins=bins, range=RangeBrightTime,
                 histtype=histtype, color=colorData)
    axHisty.hist(y, bins=bins, range=RangeDarkTime,
                 orientation='horizontal', histtype=histtype, color=colorData)

    # , c=colorscattSim, cmap=cmap
    cax = axScatter.scatter(xSim, ySim, marker='.', c='k')
    axHistx.hist(xSim, bins=bins, range=RangeBrightTime,
                 histtype=histtype, color=colorSim)
    axHisty.hist(ySim, bins=bins, range=RangeDarkTime,
                 orientation='horizontal', histtype=histtype, color=colorSim)
    axScatter.set_xlim(RangeBrightTime)
    axScatter.set_ylim(RangeDarkTime)
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())
    axScatter.set_xlabel('Bright times/s')
    axScatter.set_ylabel('Dark times/s')
    # plt.colorbar(cax)
    return fig

def long_trace_byparts(timestamps, nanotimes, TimeLimit,
                       TimeWindow=1e2, TimePeriod=1e2,
                       PhotonWindow=1e4, PhotonPeriod=1e4,
                       BinsLifetime=50, RangeLifetime=[0, 8],
                       by_photon=False):
    '''
    Arguments:
    timestamps and nanotimes should be of equal length and
    both in the units of seconds
    window and period in number of photons
    by_photon: by defaul it is false and it devides the trace by time. If true, it will devide the trace by the number of photons
    '''
    # update timestamps and nanotimes
    mask = np.logical_and(timestamps >= TimeLimit[0],
                            timestamps <= TimeLimit[1])
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

        while length_update > PhotonWindow:
            tleft = int(index_left)
            tright = int(index_left + PhotonWindow)
            # change "period" to "window" to avoid rolling
            index_left = int(index_left + PhotonPeriod)
            length_update = int(length - index_left)

            t_mac_temp = timestamps[tleft:tright]
            t_mic_temp = 1e9 * nanotimes[tleft:tright]

            df_ts['t'] = t_mac_temp
            df_ts[str(tleft)] = t_mac_temp

            # interphoton histogram
            t_diff = np.diff(t_mac_temp)
            binned_trace = np.histogram(
                t_diff, bins=500, range=(1e-5, 1e-1))
            t = binned_trace[1][:-1]
            n = binned_trace[0]
            df_ip['t'] = t
            df_ip[str(tleft)] = n / max(n)
            # lifetime histogram
            binned_trace = np.histogram(t_mic_temp,
                                        bins=BinsLifetime,
                                        range=RangeLifetime)
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
        while length_update > TimeWindow:
            tleft = int(index_left)
            tright = int(index_left + TimeWindow)
            # change "period" to "window" to avoid rolling
            index_left = int(index_left + TimePeriod)
            length_update = int(length - index_left)

            mask = np.logical_and(
                timestamps >= tleft, timestamps <= tright)
            t_mac_temp = timestamps[mask]
            t_mic_temp = 1e9 * nanotimes[mask]
            # df_ts['t'] = t_mac_temp
            # df_ts[str(tleft)] = t_mac_temp
            df_ts_temp = pd.DataFrame({str(tleft): t_mac_temp})
            # print(len(df_ts_temp))
            df_ts = pd.concat([df_ts, df_ts_temp],
                                ignore_index=True, axis=1)
            # interphoton histogram
            t_diff = np.diff(t_mac_temp)
            binned_trace = np.histogram(
                t_diff, bins=500, range=(1e-5, 1e-1))
            t = binned_trace[1][:-1]
            n = binned_trace[0]
            df_ip['t'] = t
            df_ip[str(tleft)] = n / max(n)
            # lifetime histogram
            binned_trace = np.histogram(t_mic_temp,
                                        bins=BinsLifetime,
                                        range=RangeLifetime)
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
    return df_ts, df_lt, df_fcs, df_ip


def stats_long_trace_byparts(df_ts, df_lt, df_fcs, df_ip, bg=200, from_cp_values=False, cp_values=None):
    '''bg: background in counts per second'''
    lag_time = df_fcs.iloc[:, 0]
    cols_fcs = ['A1', 'A1_err', 't_ac1', 't_ac1_err',
                'ton1', 'ton1_err', 'toff1', 'toff1_err']
    df_fcs_fit = pd.DataFrame(columns=cols_fcs)
    cols_lt = ['ampl1', 'tau1', 'ampl2', 'tau2', 'baseline']
    cols_lt = cols_lt + [s + 'err' for s in cols_lt]
    df_lt_fit = pd.DataFrame(columns=cols_lt)
    for column in df_ts.columns:
        c_f = float(column)
        if from_cp_values:
            df = cp_values[(cp_values['cp_ts'] >= c_f) & 
                           (cp_values['cp_ts'] <= c_f+100) &
                           (cp_values['cp_state'] == 1)]
            bg = df['cp_countrate'].mean()
        # fcs fit
        G = df_fcs[column]
        timestamps = df_ts[column]
        signal = len(timestamps)/(np.max(timestamps) - np.min(timestamps))
        signal = signal - bg
        result_fcs = t_on_off_fromFCS(lag_time, G,
                                      tmin=1e-5, tmax=1.0e0,
                                      fitype='mono_exp',
                                      signal=signal, bg=bg,
                                      bg_corr=True)
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
            list(fit_errs.values())  # append to rows
    print('signal: {} counts/second'.format(signal))
    return df_fcs_fit, df_lt_fit

#===============fitting functions=============

def risetime_fit(t, k1, k2, A):
    return ((A * k1 * k2 / (k2 - k1)) * (np.exp(-k1 * t) - np.exp(-k2 * t)))


def mono_exp(t, k1, A):
    return A * np.exp(-k1 * t)


def gaussian(x, amp, cen, sig):
    return amp * np.exp(-(x - cen)**2 / sig**2)


def logNormal(x, amp, cen, sig):
    return (amp / (sig * 2 * np.pi * x)) * np.exp(
        -(np.log(x) - cen)**2 / (2 * sig**2))


def streched_exp(t, k, b, A):
    return A * np.exp(-(k * t)**b)
