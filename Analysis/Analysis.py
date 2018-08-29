import os
import numpy as np
import pandas as pd
pd.set_option('precision', 9)
import h5py
# import re
from scipy.optimize import curve_fit
from lmfit import  Model, Parameter, Parameters
import matplotlib.pyplot as plt
from pylab import *
import yaml
import datetime
from pycorrelate import fcs_photonhdf5, t_on_off_fromFCS
from ChangePointProcess import *


mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.size"] = "12"
# =========Get the pointnumber, datn, emplot, FCS files with their filepath in a "GIVEN FOLDER"=====
foldername = r'/home/biswajit/Research/Reports_ppt/reports/AzurinSM-MS4/data/201702_S101toS104/S101d14Feb17_60.5_635_A2_CuAzu655'

def MetaFromPt3Hdf5(FilePathHdf5):
    FileNameHdf5 = os.path.basename(FilePathHdf5)
    string_pt3 = '.pt3'
    string_mV = 'mV'
    position_num = FileNameHdf5.find(string_pt3)
    pos_num_val_1 = FileNameHdf5[position_num-1]
    pos_num_val_2 = FileNameHdf5[position_num-2]
    pos_num_val_3 = FileNameHdf5[position_num-3]
    if not pos_num_val_1.isdigit():
        #print('Point number in %s is\\
        #  not properly placed(found): ' %FileNameHdf5)
        point_number = 1000
    elif pos_num_val_2 in ['_']:
        point_number = pos_num_val_1
    elif pos_num_val_3 in ['_']:
        point_number = pos_num_val_2 + pos_num_val_1
    point_number = int(point_number)
    #print(point_number)
    #potential extraction
    position_pot = FileNameHdf5.find(string_mV)
    pos_pot_val_1 = FileNameHdf5[position_pot-1]
    pos_pot_val_2 = FileNameHdf5[position_pot-2]
    pos_pot_val_3 = FileNameHdf5[position_pot-3]
    pos_pot_val_4 = FileNameHdf5[position_pot-4]
    pos_pot_val_5 = FileNameHdf5[position_pot-5]
    if not pos_num_val_1.isdigit():
        #print('potential value in %s is\\
        #  not properly defined' %FileNameHdf5)
        potentail_val = 1000
    elif pos_pot_val_2 in ['_']:
        potentail_val = pos_pot_val_1
    elif pos_pot_val_3 in ['_']:
        potentail_val = pos_pot_val_2 + pos_pot_val_1
    elif pos_pot_val_4 in ['_']:
        potentail_val = pos_pot_val_3 + pos_pot_val_2 + pos_pot_val_1
    elif pos_pot_val_5 in ['_']:
        potentail_val = pos_pot_val_4 +pos_pot_val_3 + pos_pot_val_2 + pos_pot_val_1
    potentail_val = int(potentail_val)
    MetaData = {'PointNumber': point_number,
                'Potential': potentail_val}
    return MetaData

def ListsPt3Hdf5(foldername):
    extensions_pt3 = [".pt3.hdf5"] #file extensions we are interested in
    string_pt3 = '.pt3.hdf5'
    columns = ['PointNumber', 'Potential', 'FileName', 'FilePathHdf5']
    pt3hdf5list = pd.DataFrame(index=None, columns=columns)
    for dirpath, dirnames, filenames in os.walk(foldername):
        for filename in [f for f in
                         filenames if f.endswith(tuple(extensions_pt3))]:
            MetaData = MetaFromPt3Hdf5(filename)
            FilePathHdf5 = os.path.join(dirpath, filename)
            templist = pd.DataFrame([[MetaData['PointNumber'],
                                      MetaData['Potential'],
                                      filename, FilePathHdf5]],
                                    columns=columns)
            pt3hdf5list = pt3hdf5list.append(templist, ignore_index=True)
    return pt3hdf5list

def CreateYamlForPt3Hdf5(FilePathHdf5, Rewrite=False):
    os.path.isfile(FilePathHdf5)
    FilePathYaml = os.path.splitext(FilePathHdf5)[0] + '.yaml'
    # Load yaml file or Create if doesn't exist
    if os.path.isfile(FilePathYaml):
        # print('exists')
        with open(FilePathYaml) as f:
            dfyaml = yaml.load(f)
    else:
        with open(FilePathYaml, 'w') as f:
            dfyaml = {'FileName': os.path.basename(FilePathHdf5),
                      'FilePath': FilePathHdf5}
            yaml.dump(dfyaml, f, default_flow_style=False)
        with open(FilePathYaml) as f:
            dfyaml = yaml.load(f)
    # add info to yaml REAL EDITING
    date = datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    dfyaml['Date Modified'] = date
    dfyaml['FilePath'] = FilePathHdf5
    # Write and dump again
    with open(FilePathYaml, 'w') as f:
        yaml.dump(dfyaml, f, default_flow_style=False)
    return dfyaml, FilePathYaml

def CreateYamlFromPt3Hdf5Folderwise(folderpath):
    '''Run it once and catiously
    Especially once the datn file deleted, don't run it.
    '''
    extensions_pt3 = [".pt3.hdf5"] #file extensions we are interested in
    for dirpath, dirnames, filenames in os.walk(folderpath):
        for filename in [f for f in filenames if
                         f.endswith(tuple(extensions_pt3))]:
            FilePathHdf5 = os.path.join(dirpath, filename)
            dfyaml, FilePathYaml = CreateYamlForPt3Hdf5(FilePathHdf5)
            MetaData = MetaFromPt3Hdf5(FilePathHdf5)
            # write potential and point number of that molecule
            dfyaml['PointNumber'] = MetaData['PointNumber']
            dfyaml['Potential'] = MetaData['Potential']
            dfyaml['Potential'] = {'Value': MetaData['Potential'],
                                   'Unit': 'mV'}
            # Get time limit from .pt3.datn
            FilePathDatn = os.path.splitext(FilePathHdf5)[0] + '.datn'
            if os.path.isfile(FilePathDatn):
                try:
                    df = pd.read_csv(FilePathDatn, header=None, sep='\t')
                    MinTime = min(df[0])
                    MaxTime = max(df[0])
                except:
                    print(FilePathDatn)
            else:
                h5 = h5py.File(FilePathHdf5, 'r')
                unit = h5['photon_data']['timestamps_specs'][
                    'timestamps_unit'][...]
                timestamps = unit * h5['photon_data']['timestamps'][...]
                MinTime = min(timestamps)
                MaxTime = max(timestamps)
                h5.close()
            dfyaml['TimeLimit'] = {'MinTime':MinTime,
                                   'MaxTime':MaxTime}
            # Write and dump again
            with open(FilePathYaml, 'w') as f:
                yaml.dump(dfyaml, f, default_flow_style=False)
    return

def GetSpecificPoints(foldername=foldername,
                      input_potential=[0, 25, 50, 100],
                      pointnumbers=[1]):
    """bin=1 in millisecond
    foldername should be a path to the folder
    """
    df_pt3hdf5list = ListsPt3Hdf5(foldername)
    df_specific = df_pt3hdf5list[df_pt3hdf5list['PointNumber'].isin(
        pointnumbers)]  # keep all the points that exist
    df_specific = df_specific[df_specific['Potential'].isin(input_potential)]
    df_specific.reset_index(drop=True, inplace=True)
    df_specific = df_specific.sort_values(by=['Potential'], ascending=True)
    df_specific.reset_index(drop=True, inplace=True)
    return df_specific
#=========== TIME_TRACE_PLOT and FCS_PLOT: given folder name, point number and list of potential, plot time traces at diff potentialof same molecule ======
def timetraceplot_potentials(foldername=foldername,
                             input_potential=[0, 25, 50, 100],
                             pointnumbers=[2], x_lim_min=0, x_lim_max=5,
                             y_lim_min=0, y_lim_max=3, bintime=5e-3,
                             show_changepoint=True, figsize=(10, 8)):
    """bin=1 in millisecond
    foldername should be a path to the folder
    """
    df_specific = GetSpecificPoints(foldername,
                                    input_potential=input_potential,
                                    pointnumbers=pointnumbers)
    fig, ax = plt.subplots(figsize=figsize, sharex=True, sharey=True)
    subplots_adjust(hspace=0.000)
    for i in range(len(df_specific)):
        given_potential = df_specific['Potential'][i]
        file_path_hdf5 = df_specific['FilePathHdf5'][i]

        ax = subplot(len(df_specific), 1, i + 1)
        out = changepoint_photonhdf5(file_path_hdf5, time_sect=100,
                                     pars=(1, 0.01, 0.99, 2))
        [hdf5_anal, timestamps, changepoint_output] = out
        plot_changepoint_trace(ax, timestamps, changepoint_output,
                               bintime, x_lim_min=0, x_lim_max=5,
                               y_lim_min=0, y_lim_max=6,
                               show_changepoint=True)
        ax.legend([str(given_potential) + " mV"],
                  fontsize=16, frameon=False)
        xlim(x_lim_min, x_lim_max)
        ylim(0, y_lim_max)  # 1.5*max(df[1]/1000)

        xticks([])
        yticks(range(0, y_lim_max, 2), fontsize=16)
        if i == len(df_specific) - 1:
            xticks(range(0, x_lim_max + 1, 1), fontsize=16)
            ax.set_xlabel('time/s', fontsize=16)
    fig.text(0.04, 0.5, 'Fluorescence(kcps)', va='center',
             rotation='vertical', fontsize=16)
    return fig

def fcsplot_potentials(foldername=foldername,
                       input_potential=[0, 25, 50, 100],
                       pointnumbers=[2], tmin=1e-5, tmax=1.0e0,
                       V_th=60, figsize=(5, 10), same_axis=True):
    '''
    Arguments:
    V_th: Potential above which monoexponential fit will be used
          and below which biexponential fit will be used
    '''
    df_specific = GetSpecificPoints(foldername,
                                    input_potential=input_potential,
                                    pointnumbers=pointnumbers)
    fig, ax = plt.subplots(figsize=figsize, sharex=True, sharey=True)
    subplots_adjust(hspace=0.000)
    pot_leg = []
    for i in range(len(df_specific)):
        potential = df_specific['Potential'][i]
        file_path_hdf5 = df_specific['FilePathHdf5'][i]
        out = fcs_photonhdf5(file_path_hdf5, tmin=None, tmax=None,
                             t_fcsrange=[1e-6, 10], nbins=100)
        [file_path_hdf5analysis, fcs_out] = out
        lag_time = fcs_out['lag_time']
        Gn = fcs_out['G(t)-1']
        if not same_axis:
            ax = subplot(len(df_specific), 1, i + 1)
        if potential > V_th:
            out = t_on_off_fromFCS(lag_time, Gn, tmin=tmin, tmax=tmax,
                                   fitype='mono_exp', bg_corr=False,
                                   plotting=True, ax=ax)
            ax.set_title([str(potential)])
            pot_leg.append(str(potential))
            pot_leg.append('fit')
        if potential < V_th:
            out = t_on_off_fromFCS(lag_time, Gn, tmin=tmin, tmax=tmax,
                                   fitype='bi_exp', bg_corr=False,
                                   plotting=True, ax=ax)
            ax.set_title([str(potential)])
            pot_leg.append(str(potential))
            pot_leg.append('fit')
    if same_axis:
        ax.legend(pot_leg)
    fig.text(0.04, 0.5, 'G(t)-1', va='center',
             rotation='vertical', fontsize=16)
    return fig
# ============== ON/OFF times from changepoint and FCS ==================

def on_off_all_folder(folderlist,
                      input_potential=[100],
                      pointnumbers=range(100),
                      pars=(1, 0.01, 0.99, 2)):
    t_ons = []
    t_offs = []
    n_on = []
    n_off = []
    for folder in folderlist:
        df_pt3hdf5list = ListsPt3Hdf5(foldername=folder)
        df_specific = df_pt3hdf5list[df_pt3hdf5list['PointNumber'].isin(
            pointnumbers)]
        df_specific = df_specific[df_specific['Potential'].isin(
            input_potential)]
        df_specific.reset_index(drop=True, inplace=True)
        for i in range(len(df_specific)):
            Point_number = df_specific['PointNumber'][i]
            file_path_hdf5 = df_specific['FilePathHdf5'][i]
            try:
                out = changepoint_photonhdf5(file_path_hdf5,
                                             time_sect=100,
                                             pars=pars,
                                             overwrite=False)
                [hdf5_anal, timestamps, cp_out] = out
                onoff_out = onoff_fromCP(cp_out, timestamps)
                df_ton = onoff_out['ontimes']
                df_toff = onoff_out['offtimes']
            except:
                df_ton = []
                df_toff = []
                pass
            t_ons = np.concatenate((t_ons, df_ton), axis=0)
            t_offs = np.concatenate((t_offs, df_toff))
    return t_ons, t_offs

def hist2D_on_off(foldername=foldername, input_potential=[100],
                  pointnumbers=[24], bins_on=40, range_on=[0, 0.01],
                  bins_off=50, range_off=[0, 1], x_shift=10,
                  plots=True, figsize=(16, 8)):
    t_ons = []; t_offs = []
    for i in pointnumbers:
        out = histogram_on_off_1mol(foldername=foldername,
                                    input_potential=input_potential,
                                    pointnumbers=[i], bins_on=bins_on,
                                    range_on=range_on, bins_off=bins_off,
                                    range_off=range_off, plotting=False)
        [t_on_temp, t_off_temp] = out
        t_ons = np.concatenate((t_ons, t_on_temp), axis=0)
        t_offs = np.concatenate((t_offs, t_off_temp), axis=0)

    t_ons = pd.Series(t_ons); t_offs = pd.Series(t_offs)
    t_on_shifted_1 = t_ons.shift(+1) ## shift up
    t_on_delay_1 = pd.DataFrame([t_on_shifted_1, t_ons])
    t_on_delay_1 = t_on_delay_1.T
    t_on_delay_1 = t_on_delay_1.dropna()
    t_off_shifted_1 = t_offs.shift(+1) ## shift up

    t_on_shifted_x = t_ons.shift(+x_shift) ## shift up
    t_off_shifted_x = t_offs.shift(+x_shift) ## shift up
    print('Number of on events: %d' %len(t_ons))
    print('Number of off events: %d' %len(t_offs))
    if plots == True:
        import matplotlib as mpl
        colormap = mpl.cm.jet
        fig = plt.figure(figsize=figsize)

        ax1 = fig.add_subplot(2, 3, 1)#2,2,1
        out = hist2d(t_on_shifted_1[1:], t_ons[1:],
                     range=[range_on, range_on],
                     bins=bins_on,
                     norm=mpl.colors.LogNorm(), cmap=colormap)
        [C_on_1, Ex_on_1, Ey_on_1, fig] = out
        Ex_on_1, Ey_on_1 = meshgrid(Ex_on_1, Ey_on_1)
        # ax1.pcolormesh(Ex_on_1, Ey_on_1, C_on_1, cmap=colormap)#,norm=mpl.colors.LogNorm()
        colorbar()
        ax1.set_title('ON time Cu-Azu %smV' %input_potential)
        ax1.set_xlabel(r'$\tau_{on}/s$')
        ax1.set_ylabel(r'$\tau_{on}+1/s$')

        ax2 = fig.add_subplot(2, 3, 2)#2,2,1
        out = hist2d(t_on_shifted_x[x_shift:], t_ons[x_shift:],
                     range=[range_on, range_on],
                     bins=bins_on,
                     norm=mpl.colors.LogNorm(), cmap=colormap)
        [C_on_x, Ex_on_x, Ey_on_x, fig] = out
        Ex_on_x, Ey_on_x = meshgrid(Ex_on_x, Ey_on_x)
        # ax2.pcolormesh(Ex_on_x, Ey_on_x, C_on_x, cmap=colormap)#,norm=mpl.colors.LogNorm()
        colorbar()
        ax2.set_title('ON time Cu-Azu %smV' %input_potential)
        ax2.set_xlabel(r'$\tau_{on}/s$')
        ax2.set_ylabel(r'$\tau_{on}+%s/s$'%x_shift)

        ax3 = fig.add_subplot(2, 3, 3)
        C_on_diff = C_on_1-C_on_x
        pcm = ax3.pcolormesh(Ex_on_x, Ey_on_x, C_on_diff,
                             norm=mpl.colors.SymLogNorm(linthresh=2,
                             linscale=2, vmin=C_on_diff.min(), vmax=C_on_diff.max()),
                             cmap=colormap)
        fig.colorbar(pcm, ax=ax3, extend='max')

        ax4 = fig.add_subplot(2, 3, 4)
        out = hist2d(t_off_shifted_1[1:], t_offs[1:],
                     range=[range_off, range_off],
                     bins=bins_off,
                     norm=mpl.colors.LogNorm(), cmap=colormap)
        [C_off_1, Ex_off_1, Ey_off_1, figu] = out
        Ex_off_1, Ey_off_1 = meshgrid(Ex_off_1, Ey_off_1)
        colorbar()
        ax4.set_title('OFF time Cu-Azu %smV' %input_potential)
        ax4.set_xlabel(r'$\tau_{off}/s$')
        ax4.set_ylabel(r'$\tau_{off}+1/s$')

        ax5 = fig.add_subplot(2, 3, 5)
        out = hist2d(t_off_shifted_x[x_shift:], t_offs[x_shift:],
                     range=[range_off, range_off],
                     bins=bins_off,
                     norm=mpl.colors.LogNorm(), cmap=colormap)
        [C_off_x, Ex_off_x, Ey_off_x, figu] = out
        Ex_off_x, Ey_off_x = meshgrid(Ex_off_x, Ey_off_x)
        colorbar()
        ax5.set_title('OFF time Cu-Azu %smV' %input_potential)
        ax5.set_xlabel(r'$\tau_{off}/s$')
        ax5.set_ylabel(r'$\tau_{off}+%s/s$'%x_shift)

        ax6 = fig.add_subplot(2, 3, 6)
        C_off_diff = C_off_1-C_off_x
        pcm = ax6.pcolormesh(Ex_off_x, Ey_off_x, C_off_diff,
                             norm=mpl.colors.SymLogNorm(linthresh=0.1,
                                                        linscale=0.1,
                                                        vmin=C_off_diff.min(),
                                                        vmax=C_off_diff.max()),
                             cmap=colormap)
        fig.colorbar(pcm, ax=ax6, extend='max')
        plt.tight_layout()
    return(t_ons, t_offs)
#==================== TIME TRACE OUTPUT ==============
potential = 35
pointnumbers = range(100)
potentialist = range(-100, 200, 1)
def cp_outputs_folderwise(folderpath = foldername,
                          pointnumbers=[1],
                          potentialist=potentialist,
                          pars=(1, 0.01, 0.99, 2)):
    df_pt3hdf5list = ListsPt3Hdf5(foldername=folderpath)
    df_specific = df_pt3hdf5list[df_pt3hdf5list['PointNumber'].isin(
                                pointnumbers)]
    df_specific = df_specific[df_specific['Potential'].isin(potentialist)]
    out_total = pd.DataFrame()  # initiating empty output matrix
    for input_number in pointnumbers:
        df_specific_i = df_specific[df_specific['PointNumber']
                                    == input_number]
        df_specific_i = df_specific_i.sort_values(
            by=['Potential'], ascending=True)
        df_specific_i.reset_index(drop=True, inplace=True)
        if not df_specific_i.empty:
            #---------Create Pandas array to save outputs----------
            indices = np.ones(7)
            indices = indices.astype(str)
            Point_number = 'Point_' + str(input_number)
            indices[:] = Point_number
            subgroup = ['Potential', 't_onav', 't_onaverr',
                        't_offav', 't_offaverr', 't_ratio', 't_ratioerr']
            arrays = [indices, subgroup]
            col = pd.MultiIndex.from_arrays(arrays)
            length = (len(df_specific_i))  # for defining dimension of out_mat
            # create zeroes which will be replaced by proper values
            out_point = pd.DataFrame(
                zeros((length, len(subgroup))), columns=col)

            #---------Pandas array created to save outputs----
            out_point[Point_number]['Potential'] = df_specific_i['Potential']
            poten_array = []
            t_onav = []
            t_onaverr = []
            t_offav = []
            t_offaverr = []
            ratio = []
            ratioerr = []  # Empty forms for timetrace output
            for i in range(length):
                potential = df_specific_i['Potential'][i]
                file_path_hdf5 = df_specific_i['FilePathHdf5'][i]
                # read and analyze change point
                out = changepoint_photonhdf5(file_path_hdf5, time_sect=100,
                                             pars=pars,
                                             overwrite=False)
                [hdf5_anal, timestamps, cp_out] = out
                onoff_out = onoff_fromCP(cp_out, timestamps)
                ton = onoff_out['tonav']
                ton_err = onoff_out['tonav_err']
                toff = onoff_out['toffav']
                toff_err = onoff_out['toffav_err']
                t_ratio = toff / ton
                t_ratio_err = (toff / ton) * sqrt(((toff_err / toff)**2) +
                                                  ((ton_err / ton)**2))
                # append to the main array
                poten_array.append(potential)
                t_onav.append(ton)
                t_onaverr.append(ton_err)  # needs to be changed
                t_offav.append(toff)
                t_offaverr.append(toff_err)  # needs to be changed
                ratio.append(t_ratio)
                ratioerr.append(t_ratio_err)  # needs to be changed

                df_create = array([poten_array, t_onav, t_onaverr,
                                   t_offav, t_offaverr, ratio, ratioerr])
                df_create = df_create.astype(float64)  # ;a=pd.DataFrame(a)
                df_create = pd.DataFrame(df_create.T, columns=subgroup)
                out_point[Point_number] = df_create
            out_total = pd.concat([out_total, out_point], axis=1)
    return out_total

def fcs_outputs_folderwise(folderpath=foldername,
                           pointnumbers=[1],
                           potentialist=potentialist):
    df_pt3hdf5list = ListsPt3Hdf5(foldername=folderpath)
    df_specific = df_pt3hdf5list[df_pt3hdf5list['PointNumber'].isin(
                                pointnumbers)]
    df_specific = df_specific[df_specific['Potential'].isin(potentialist)]
    out_total = pd.DataFrame()  # initiating empty output matrix
    for input_number in pointnumbers:
        df_specific_i = df_specific[df_specific['PointNumber']
                                    == input_number]
        df_specific_i = df_specific_i.sort_values(
            by=['Potential'], ascending=True)
        df_specific_i.reset_index(drop=True, inplace=True)
        if not df_specific_i.empty:
            #---------Create Pandas array to save outputs----------
            indices = np.ones(7)
            indices = indices.astype(str)
            Point_number = 'Point_' + str(input_number)
            indices[:] = Point_number
            subgroup = ['Potential', 't_onav', 't_onaverr',
                        't_offav', 't_offaverr', 't_ratio', 't_ratioerr']
            arrays = [indices, subgroup]
            col = pd.MultiIndex.from_arrays(arrays)
            length = (len(df_specific_i))  # for defining dimension of out_mat
            # create zeroes which will be replaced by proper values
            out_point = pd.DataFrame(
                zeros((length, len(subgroup))), columns=col)

            #---------Pandas array created to save outputs----
            out_point[Point_number]['Potential'] = df_specific_i['Potential']
            poten_array = []
            t_onav = []
            t_onaverr = []
            t_offav = []
            t_offaverr = []
            ratio = []
            ratioerr = []  # Empty forms for timetrace output
            for i in range(length):
                potential = df_specific_i['Potential'][i]
                file_path_hdf5 = df_specific_i['FilePathHdf5'][i]
                # read and analyze change point

                # read fcs and fit fcs
                out = fcs_photonhdf5(file_path_hdf5, tmin=None, tmax=None,
                                     t_fcsrange=[1e-6, 10], nbins=100,
                                     overwrite=False)
                [file_path_hdf5analysis, fcs_out] = out
                lag_time = fcs_out['lag_time']
                Gn = fcs_out['G(t)-1']
                try:
                    fcs_fit_result = t_on_off_fromFCS(lag_time, Gn,
                                                      tmin=1e-5, tmax=1.0e3,
                                                      bg_corr=False)
                    ton = float(fcs_fit_result['ton1'])
                    ton_err = float(fcs_fit_result['ton1_err'])
                    toff = float(fcs_fit_result['toff1'])
                    toff_err = float(fcs_fit_result['toff1_err'])
                    t_ratio = toff / ton
                    t_ratio_err = (toff / ton) * sqrt(((toff_err / toff)**2) +
                                                      ((ton_err / ton)**2))
                    # append to the main array
                    poten_array.append(potential)
                    t_onav.append(ton)
                    t_onaverr.append(ton_err)  # needs to be changed
                    t_offav.append(toff)
                    t_offaverr.append(toff_err)  # needs to be changed
                    ratio.append(t_ratio)
                    ratioerr.append(t_ratio_err)  # needs to be changed

                    df_create = array([poten_array, t_onav, t_onaverr,
                                       t_offav, t_offaverr, ratio, ratioerr])
                    df_create = df_create.astype(float64)  # ;a=pd.DataFrame(a)
                    df_create = pd.DataFrame(df_create.T, columns=subgroup)
                    out_point[Point_number] = df_create
                except:
                    print('fcs fitting of %s with potential %s didnot succeed' % (Point_number, potential))
            out_total = pd.concat([out_total, out_point], axis=1)
    return out_total

# ============== Midpoint potential ==============
def Mid_potentials_slopem_lmfit(folderpath=foldername, pointnumbers=range(5),
                          process='cp', plotting=True,
                          min_pot=40, max_pot=150, min_pot_num=1):
    '''
    Argument: 
    process: 'fcs' or 'cp'
    '''
    if process == 'cp':
        out = cp_outputs_folderwise(folderpath=folderpath,
                                    pointnumbers=pointnumbers,
                                    potentialist=potentialist)
    if process == 'fcs':
        out = fcs_outputs_folderwise(folderpath=folderpath,
                                     pointnumbers=pointnumbers,
                                     potentialist=potentialist)
    # Nernst Equation definition        
    def nernst(x, a):
        '''x is potential
        a: E0/midpoint potential(parameter)
        returns ratio(t_oxd/t_red)'''
        return(10**((x-a) / 0.059))
    nernst_mod = Model(nernst)
    params_nernst = nernst_mod.make_params(a=0.02)
    def nernst_slopem(x, a, m):
        return(10**((x-a) / m))
    nernst_slopem_mod = Model(nernst_slopem)
    params_ner_slop = nernst_slopem_mod.make_params(a=0.02, m=0.059)
    columns_E0 = ['Point number', 'E0_fit', 'E0_err']
    columns_E0_m = ['Point number', 'E0_fit', 'E0_err', 'slope', 'slope_error']
    E0_list = pd.DataFrame(index=None, columns=columns_E0)
    E0_m_list = pd.DataFrame(index=None, columns=columns_E0_m)
    #--figure initiation----
    if plotting == True:
        fig = plt.figure(figsize=(10,4))
        nrows = 1; ncols = 2
        ax00 = plt.subplot2grid((nrows, ncols), (0, 0))
        ax01 = plt.subplot2grid((nrows, ncols), (0, 1))
        cmap = plt.get_cmap('jet')#jet_r
        N = len(out.columns.levels[0])
    for i in range(len(out.columns.levels[0])):
        point = out.columns.levels[0][i]
        PointNumber = point[6:]
        point_output_tot = out[point].dropna()
        point_output = point_output_tot[point_output_tot[
                       'Potential'] >= min_pot] #select a potential threshold
        point_output = point_output_tot[point_output_tot[
                       'Potential'] <= max_pot] #select a potential threshold
        point_output.reset_index(drop=True, inplace=True)
        if len(point_output) > min_pot_num:
            potential = point_output['Potential']
            t_onav = point_output['t_onav']
            t_onaverr = point_output['t_onaverr']
            t_offav = point_output['t_offav']
            t_offaverr = point_output['t_offaverr']
            t_ratio = point_output['t_ratio']
            t_ratioerr = point_output['t_ratioerr']
            E = potential*0.001 #converting to mV
            #--------fitting nernst----------------
            res_nernst = nernst_mod.fit(t_ratio, params_nernst, x=E)
            out_params = str((res_nernst.params['a'], 'value'))
            E0 = res_nernst.best_values['a']
            E0_err = float(out_params.split('+/- ')[1].split(', bounds')[0])
            #---------append to list---------
            E0_list_temp = pd.DataFrame([[PointNumber, E0, E0_err]],
                                        columns=columns_E0)
            E0_list = E0_list.append(E0_list_temp, ignore_index=True)
            #--------fitting nernst_slopem------
            res_nernst_slop = nernst_slopem_mod.fit(t_ratio, params_ner_slop, x=E)
            E0_m = res_nernst_slop.best_values['a']
            out_params = str((res_nernst_slop.params['a'], 'value'))
            E0_m_err = float(out_params.split('+/- ')[1].split(', bounds')[0])
            slope_m = res_nernst_slop.best_values['m']
            out_params_m = str((res_nernst_slop.params['m'], 'value'))
            slope_m_err = float(out_params_m.split('+/- ')[1].split(', bounds')[0])
            E0_m_list_temp = pd.DataFrame([[PointNumber, E0_m, E0_m_err, slope_m, slope_m_err]],
                                          columns=columns_E0_m)
            E0_m_list = E0_m_list.append(E0_m_list_temp, ignore_index=True)
            #-----plot------
            if plotting == True:
                color = cmap(float(i)/N)
                # ax00.errorbar(point_output_tot['Potential'], point_output_tot['t_ratio'],
                # yerr=point_output_tot['t_ratioerr'], fmt='o', color=color, label=point)#plot raw outputs
                ax00.plot(point_output_tot['Potential'],
                          point_output_tot['t_ratio'], 'o',
                          color=color, label=point)#plot raw outputs
                ax00.plot(linspace(min(potential)-10, max(potential)+10),
                          nernst(0.001*linspace(min(potential)-10,
                          max(potential)+10), E0), color=color, linewidth=2.0)#color
                #plot(E*1000, nernst(E, *E0_fit), color=color, linewidth=2.0)
                ax01.errorbar(point_output_tot['Potential'],
                              point_output_tot['t_ratio'],
                              yerr = point_output_tot['t_ratioerr'],
                              fmt='o', color=color, label=point)#plot raw outputs
                ax01.plot(linspace(min(potential)-10, max(potential)+10),
                          nernst_slopem(0.001*linspace(min(potential)-10,
                          max(potential)+10), E0_m, slope_m),
                          color=color, linewidth=2.0)
                ax00.set_yscale('log')
                ax00.set_xlim(min_pot-10, None)
                ax00.set_xlabel('$Potential [V]$', fontsize=20)
                ax00.set_ylabel('$T_{dark}/T_{bright}$', fontsize=20)
                ax00.tick_params(axis='both', which='major', labelsize=16)
                ax00.set_title(r'$E=E_0 + 0.059log(\frac{t_{off}}{t_{on}})$')
                ax01.set_yscale('log')
                ax01.set_xlabel('$Potential [V]$', fontsize=20)
                ax01.set_ylabel('$T_{dark}/T_{bright}$', fontsize=20)
                ax01.tick_params(axis='both', which='major', labelsize=16)
                ax01.set_title(r'$E=E_0 + m.log(\frac{t_{off}}{t_{on}})$')
                tight_layout()
    return(E0_list, E0_m_list)

# =========== FOLDERWISE ==============

def fcs_folderwise(folderpath, t_fcsrange=[1e-6, 1], nbins=100,
                   overwrite=False):
    start_time = time.time()
    pt3hdf5_extension = [".pt3.hdf5", ".t3r.hdf5"]
    for dirpath, dirname, filenames in os.walk(folderpath):
        for filename in [f for f in filenames if 
                         f.endswith(tuple(pt3hdf5_extension))]:
            FilePathHdf5 = os.path.join(dirpath, filename)
            FilePathYaml = FilePathHdf5[:-4] + 'yaml'
            with open(FilePathYaml) as f:
                dfyaml = yaml.load(f)
            tmin = dfyaml['TimeLimit']['MinTime']
            tmax = dfyaml['TimeLimit']['MaxTime']

            start_time_i = time.time()
            print("---%.1f : fcs calculation started for %s\n" %
                  (start_time_i, FilePathHdf5))
            try:
                out = fcs_photonhdf5(FilePathHdf5, tmin=tmin, tmax=tmax,
                                     t_fcsrange=t_fcsrange, nbins=nbins,
                                     overwrite=overwrite)
            except:
                out = fcs_photonhdf5(FilePathHdf5, tmin=None, tmax=None,
                                     t_fcsrange=t_fcsrange, nbins=nbins,
                                     overwrite=overwrite)
            processtime = time.time() - start_time_i
            print("---TOTAL time took for the file: %s IS: %s seconds ---\n" %
                  (FilePathHdf5, processtime))
    print("---TOTAL time took for the folder: %s seconds ---\n" %
          (time.time() - start_time))
    return
