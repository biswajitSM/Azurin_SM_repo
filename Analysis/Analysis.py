import os
import numpy as np
import pandas as pd
pd.set_option('precision', 9)
import h5py
import re
from scipy.optimize import curve_fit
from lmfit import  Model, Parameter, Parameters
import matplotlib.pyplot as plt
from pylab import *

from pycorrelate import *
from changepoint_process import *


mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.size"] = "14"
# =========Get the pointnumber, datn, emplot, FCS files with their filepath in a "GIVEN FOLDER"=====
foldername = r'/home/biswajit/Research/Reports_ppt/reports/AzurinSM-MS4/data/201702_S101toS104/S101d14Feb17_60.5_635_A2_CuAzu655'
def dir_mV_molNo_pt3(foldername=foldername):
    extensions_pt3 = [".pt3"] #file extensions we are interested in
    string_pt3 = '.pt3'
    string_mV = 'mV'
    columns = ['Point number', 'Potential', 'filename[pt3]', 'filepath[pt3]']
    pt3_list = pd.DataFrame(index=None, columns=columns)
    for dirpath, dirnames, filenames in os.walk(foldername):
        for filename in [f for f in filenames if f.endswith(tuple(extensions_pt3))]:
            position_num = filename.find(string_pt3)
            pos_num_val_1 = filename[position_num-1]
            pos_num_val_2 = filename[position_num-2]
            pos_num_val_3 = filename[position_num-3]
            if not pos_num_val_1.isdigit():
                #print('Point number in %s is not properly placed(found): ' %filename)
                point_number = 1000
            elif pos_num_val_2 in ['_']:
                point_number = pos_num_val_1
            elif pos_num_val_3 in ['_']:
                point_number = pos_num_val_2 + pos_num_val_1
            point_number = int(point_number)
            #print(point_number)
            #potential extraction
            position_pot = filename.find(string_mV)
            pos_pot_val_1 = filename[position_pot-1]
            pos_pot_val_2 = filename[position_pot-2]
            pos_pot_val_3 = filename[position_pot-3]
            pos_pot_val_4 = filename[position_pot-4]
            pos_pot_val_5 = filename[position_pot-5]
            if not pos_num_val_1.isdigit():
                #print('potential value in %s is not properly defined' %filename)
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
            file_pt3_path = os.path.join(dirpath, filename)
            temp_pt3list = pd.DataFrame([[point_number, potentail_val, filename, file_pt3_path]], columns=columns)
            pt3_list = pt3_list.append(temp_pt3list, ignore_index=True)
    return pt3_list
def dir_mV_molNo(foldername=foldername):
    pt3_list = dir_mV_molNo_pt3(foldername=foldername)
    extensions = [".dat", ".datn"]
    datn_em_list = pd.DataFrame()
    for i in range(len(pt3_list)):
        pt3_filename = pt3_list['filename[pt3]'][i]
        point_number = int(pt3_list['Point number'][i])
        potential = int(pt3_list['Potential'][i])
        pt3_path = pt3_list['filepath[pt3]'][i]
        # check if datn exist
        datn_file_path = pt3_path + '.datn'
        #add .datn and .em.plot
        if os.path.isfile(datn_file_path):
            emplot_path = datn_file_path + '.em.plot'
            filename_hdf5 = pt3_filename[:-3] + 'hdf5'
            hdf5_filepath = pt3_path[:-3] + 'hdf5'
            columns_datn_em = ['Point number', 'Potential',
                                              'pt3_filename', 'pt3_path',
                                              'filepath[.datn]',
                                              'filepath[.em.plot]',
                                              'filepath[.hdf5]'
                                            ]
            temp_datn_list = pd.DataFrame([[point_number, potential,
                                                                  pt3_filename, pt3_path,
                                                                  datn_file_path,
                                                                  emplot_path,
                                                                  hdf5_filepath]],
                                          columns=columns_datn_em)
            datn_em_list = datn_em_list.append(temp_datn_list, ignore_index=True)
    pt3_list = pt3_list.sort_values(by=['Point number'], ascending=True)
    pt3_list.reset_index(drop=True, inplace=True)
    datn_em_list = datn_em_list.sort_values(by=['Point number'], ascending=True)
    datn_em_list.reset_index(drop=True, inplace=True)
    return datn_em_list
def point_list(foldername = foldername, pointnumbers=range(100)):
    df_datn_emplot = dir_mV_molNo(foldername=foldername)
    df_datn_specific = df_datn_emplot[df_datn_emplot['Point number'].isin(pointnumbers)]
    df_groupby = df_datn_specific.groupby(['Point number'])
    out_total = pd.DataFrame()
    for name, group in df_groupby:
        temp = df_groupby.get_group(name)
        temp = temp['Potential']
        df_point = pd.DataFrame(sort(array(temp)), columns=['Point_'+str(int(name))])
        out_total=pd.concat([out_total, df_point], axis=1);
    out_total = out_total.replace(np.nan, '', regex=True)
    return(out_total)
def point_not_working(foldername = foldername, pointnumbers=range(100)):
    df_datn_emplot = dir_mV_molNo(foldername=foldername)
    df_datn_specific = df_datn_emplot[df_datn_emplot['Point number'].isin(pointnumbers)]
    df_datn_path = df_datn_specific['filepath[.em.plot]']
    output = np.array([])
    for i in df_datn_path:
        try:
            df=i
            df = pd.read_csv(df, header=None, sep='\t')
        except:
            output = np.concatenate((output, np.array([i])))
            pass
    #os.chdir(parentdir)
    #savetxt('notworking.dat', output, fmt='%s')
    return(output)
def removeifemplotdoesntexist(foldername = foldername, pointnumbers=range(100)):
    df_datn_emplot, df_FCS, folderpath = dir_mV_molNo(foldername=foldername)
    #df_datn_specific = df_datn_emplot[df_datn_emplot['Point number'].isin(pointnumbers)]

    for i in range(len(df_datn_emplot['filepath[.em.plot]'])):
        if not os.path.isfile(df_datn_emplot['filepath[.em.plot]'][i]):
            df_datn_emplot['filepath[.em.plot]'][i] = 0
            df_FCS['filepath[FCS]'][i] = 0
    df_datn_em = df_datn_emplot[df_datn_emplot['filepath[.em.plot]'] != 0]
    df_FCS = df_FCS[df_datn_emplot['filepath[.em.plot]'] != 0]
    df_datn_path = df_datn_em['filepath[.em.plot]']
    output = np.array([])
    for i in df_datn_path:
        try:
            df=i
            df = pd.read_csv(df, header=None, sep='\t')
        except:
            output = np.concatenate((output, np.array([i])))
            pass
    print('Output should be empty')
    #os.chdir(parentdir)
    #savetxt('notworking.dat', output, fmt='%s')
    return(output)
def get_point_specifics(foldername= foldername, input_potential=[0, 25, 50, 100], pointnumbers=[1]):
    """bin=1 in millisecond
    foldername should be given as r'D:\Research\...'
    """
    df_datn_emplot = dir_mV_molNo(foldername)
    df_specific = df_datn_emplot[df_datn_emplot['Point number'].isin(pointnumbers)]#keep all the points that exist
    df_specific = df_specific[df_specific['Potential'].isin(input_potential)]; df_specific.reset_index(drop=True, inplace=True)
    df_specific =df_specific.sort_values(by=['Potential'], ascending=True); df_specific.reset_index(drop=True, inplace=True)
    return df_specific
#---------TIME_TRACE_PLOT and FCS_PLOT: given folder name, point number and list of potential, plot time traces at diff potentialof same molecule------
def timetraceplot_potentials(foldername=foldername, input_potential=[0, 25, 50, 100],
                             pointnumbers=[2], x_lim_min=0, x_lim_max=5,
                             y_lim_min=0, y_lim_max=3, bintime=5e-3,
                             show_changepoint=True, figsize=(10, 8)):
    """bin=1 in millisecond
    foldername should be given as r'D:\Research\...'
    """
    df_specific = get_point_specifics(foldername, input_potential=input_potential,
                                      pointnumbers=pointnumbers)
    fig, ax = plt.subplots(figsize=figsize, sharex=True, sharey=True)
    subplots_adjust(hspace=0.000)
    for i in range(len(df_specific)):
        given_potential = df_specific['Potential'][i]
        #f_datn = df_specific['filepath[.datn]'][i]
        f_emplot = df_specific['filepath[.em.plot]'][i]
        file_path_hdf5 = df_specific['filepath[.hdf5]'][i]

        ax = subplot(len(df_specific), 1, i + 1)
        out = changepoint_photonhdf5(file_path_hdf5, time_sect=25,
                                     pars=(1, 0.01, 0.99, 2))
        [hdf5_anal, timestamps, changepoint_output] = out
        plot_changepoint_trace(ax, timestamps, changepoint_output,
                               bintime, x_lim_min=0, x_lim_max=5,
                               y_lim_min=0, y_lim_max=6,
                               show_changepoint=True)
        ax.legend([str(given_potential) + " mV"], fontsize=16, framealpha=0.5)
        xlim(x_lim_min, x_lim_max)
        ylim(0, y_lim_max)  # 1.5*max(df[1]/1000)
        # xticks([])
        yticks(range(0, y_lim_max, 2), fontsize=16)
        if i == len(df_specific) - 1:
            # xticks(range(0, x_lim_max+1, 1), fontsize=16)
            ax.set_xlabel('time/s', fontsize=16)
    fig.text(0.04, 0.5, 'Fluorescence(kcps)', va='center',
             rotation='vertical', fontsize=16)
    return fig
def fcsplot_potentials(foldername=foldername, input_potential=[0, 25, 50, 100],
                       pointnumbers=[2], tmin=1e-5, tmax=1.0e0,
                      V_th = 60, figsize=(5, 10), same_axis=True):
    '''
    Arguments:
    V_th: Potential above which monoexponential fit will be used
          and below which biexponential fit will be used
    '''
    df_specific = get_point_specifics(foldername, input_potential=input_potential,
                                      pointnumbers=pointnumbers)
    fig, ax = plt.subplots(figsize=figsize, sharex=True, sharey=True)
    subplots_adjust(hspace=0.000)
    pot_leg = []
    for i in range(len(df_specific)):
        potential = df_specific['Potential'][i]
        file_path_hdf5 = df_specific['filepath[.hdf5]'][i]
        out = fcs_photonhdf5(file_path_hdf5, tmin=None, tmax=None,
                             t_fcsrange=[1e-6, 10], nbins=100);
        [file_path_hdf5analysis, fcs_out] = out       
        lag_time = fcs_out['lag_time'];
        Gn = fcs_out['G(t)-1'];
        if not same_axis:
            ax = subplot(len(df_specific), 1, i + 1)
        if potential > V_th:
            out = t_on_off_fromFCS(lag_time, Gn, tmin=tmin, tmax=tmax,
                                     fitype='mono_exp', bg_corr=False,
                                     plotting=True, ax=ax);
            ax.set_title([str(potential)])
            pot_leg.append(str(potential))
            pot_leg.append('fit')
        if potential < V_th:
            out = t_on_off_fromFCS(lag_time, Gn, tmin=tmin, tmax=tmax,
                                     fitype='bi_exp', bg_corr=False,
                                     plotting=True, ax=ax);
            ax.set_title([str(potential)])
            pot_leg.append(str(potential))
            pot_leg.append('fit')
    if same_axis:
        ax.legend(pot_leg)
    fig.text(0.04, 0.5, 'G(t)-1', va='center',
             rotation='vertical', fontsize=16)
    return fig
# ==============ON/OFF times from changepoint and FCS==================
def histogram_on_off_folder(foldername= foldername, input_potential=[100], pointnumbers=range(100),
                          bins_on=50, range_on=[0, 0.2], bins_off=50, range_off=[0, 0.5], plotting=False):
    df_datn_emplot, df_FCS, folder = dir_mV_molNo(foldername)
    df_specific = df_datn_emplot[df_datn_emplot['Point number'].isin(pointnumbers)]#keep all the points that exist
    df_specific = df_specific[df_specific['Potential'].isin(input_potential)]; df_specific.reset_index(drop=True, inplace=True)
    t_ons = []; t_offs = [];
    for i in range(len(df_specific)):
        f_datn_path = df_specific['filepath[.datn]'].values[i]
        f_emplot_path = df_specific['filepath[.em.plot]'].values[i]
        df_ton, df_toff, average_ton, average_toff, average_ton_err, average_toff_err = t_on_off_fromCP(f_emplot_path);
        t_ons = np.concatenate((t_ons, df_ton), axis=0);
        t_offs = np.concatenate((t_offs, df_toff));
    if plotting == True:
        fig, axes = plt.subplots(1, 2, figsize=(10,4))
        n,bins_on,patches = axes[0].hist(t_ons, range=range_on,bins=bins_on)
        axes[0].set_xlabel(r'$\tau_{on}$')
        axes[0].set_ylabel('#')
        #axes[0].set_yscale('log')
        axes[0].set_title("ON time histogram at %s mV" %input_potential[0])
        n,bins_off,patches = axes[1].hist(t_offs, range=range_off,bins=bins_off)
        axes[1].set_xlabel(r'$\tau_{off}$')
        axes[1].set_ylabel('#')
        #axes[1].set_yscale('log')
        axes[1].set_title("OFF time histogram at %s mV" %input_potential[0])
    return(t_ons, t_offs)
def hist2D_on_off(foldername=foldername, input_potential=[100],
                  pointnumbers=[24], bins_on=40, range_on=[0, 0.01],
                  bins_off=50, range_off=[0, 1], x_shift=10,
                  plots=True, figsize=(16, 8)):
    t_ons = []; t_offs=[];
    for i in pointnumbers:
        out = histogram_on_off_1mol(foldername= foldername,
                                    input_potential=input_potential,
                                    pointnumbers=[i], bins_on=bins_on,
                                    range_on=range_on, bins_off=bins_off,
                                    range_off=range_off, plotting=False)
        [t_on_temp, t_off_temp,n_on, bins_on, n_off, bins_off] = out
        t_ons = np.concatenate((t_ons, t_on_temp), axis=0)
        t_offs = np.concatenate((t_offs, t_off_temp), axis=0)

    t_ons=pd.Series(t_ons);t_offs=pd.Series(t_offs)
    t_on_shifted_1 = t_ons.shift(+1) ## shift up
    t_on_delay_1 = pd.DataFrame([t_on_shifted_1, t_ons]);
    t_on_delay_1 = t_on_delay_1.T
    t_on_delay_1 = t_on_delay_1.dropna();
    t_off_shifted_1 = t_offs.shift(+1) ## shift up

    t_on_shifted_x = t_ons.shift(+x_shift) ## shift up
    t_off_shifted_x = t_offs.shift(+x_shift) ## shift up
    print('Number of on events: %d' %len(t_ons))
    print('Number of off events: %d' %len(t_offs))
    if plots == True:
        import matplotlib as mpl
        colormap=mpl.cm.jet
        fig = plt.figure(figsize=figsize)

        ax1 = fig.add_subplot(2,3,1)#2,2,1
        C_on_1,Ex_on_1,Ey_on_1, figu = hist2d(t_on_shifted_1[1:], t_ons[1:], range=[range_on, range_on], bins=bins_on, norm=mpl.colors.LogNorm(), cmap=colormap)
        Ex_on_1,Ey_on_1 = meshgrid(Ex_on_1,Ey_on_1)
        # ax1.pcolormesh(Ex_on_1, Ey_on_1, C_on_1, cmap=colormap)#,norm=mpl.colors.LogNorm()
        colorbar()
        ax1.set_title('ON time Cu-Azu %smV' %input_potential)
        ax1.set_xlabel(r'$\tau_{on}/s$')
        ax1.set_ylabel(r'$\tau_{on}+1/s$')

        ax2 = fig.add_subplot(2,3,2)#2,2,1
        C_on_x,Ex_on_x,Ey_on_x, figu = hist2d(t_on_shifted_x[x_shift:], t_ons[x_shift:], range=[range_on, range_on], bins=bins_on, norm=mpl.colors.LogNorm(), cmap=colormap)
        Ex_on_x,Ey_on_x = meshgrid(Ex_on_x,Ey_on_x)
        # ax2.pcolormesh(Ex_on_x, Ey_on_x, C_on_x, cmap=colormap)#,norm=mpl.colors.LogNorm()
        colorbar()
        ax2.set_title('ON time Cu-Azu %smV' %input_potential)
        ax2.set_xlabel(r'$\tau_{on}/s$')
        ax2.set_ylabel(r'$\tau_{on}+%s/s$'%x_shift)

        ax3 = fig.add_subplot(2,3,3)
        C_on_diff = C_on_1-C_on_x;
        pcm=ax3.pcolormesh(Ex_on_x, Ey_on_x, C_on_diff,
                       norm=mpl.colors.SymLogNorm(linthresh=2, linscale=2,vmin=C_on_diff.min(), vmax=C_on_diff.max()), cmap=colormap)
        fig.colorbar(pcm, ax=ax3, extend='max')

        ax4 = fig.add_subplot(2,3,4)
        C_off_1, Ex_off_1, Ey_off_1, figu= hist2d(t_off_shifted_1[1:], t_offs[1:], range=[range_off, range_off],bins=bins_off, norm=mpl.colors.LogNorm(), cmap=colormap);#, norm=mpl.colors.LogNorm()
        Ex_off_1, Ey_off_1 = meshgrid(Ex_off_1, Ey_off_1)
        colorbar()
        ax4.set_title('OFF time Cu-Azu %smV' %input_potential)
        ax4.set_xlabel(r'$\tau_{off}/s$')
        ax4.set_ylabel(r'$\tau_{off}+1/s$')

        ax5 = fig.add_subplot(2,3,5)
        C_off_x,Ex_off_x,Ey_off_x, figu = hist2d(t_off_shifted_x[x_shift:], t_offs[x_shift:], range=[range_off, range_off],bins=bins_off, norm=mpl.colors.LogNorm(), cmap=colormap);#, norm=mpl.colors.LogNorm()
        Ex_off_x,Ey_off_x = meshgrid(Ex_off_x,Ey_off_x)
        colorbar()
        ax5.set_title('OFF time Cu-Azu %smV' %input_potential)
        ax5.set_xlabel(r'$\tau_{off}/s$')
        ax5.set_ylabel(r'$\tau_{off}+%s/s$'%x_shift)

        ax6 = fig.add_subplot(2,3,6)
        C_off_diff=C_off_1-C_off_x
        pcm=ax6.pcolormesh(Ex_off_x, Ey_off_x, C_off_diff,
                           norm = mpl.colors.SymLogNorm(linthresh=0.1,linscale=0.1,vmin=C_off_diff.min(), vmax=C_off_diff.max()), cmap=colormap)
        fig.colorbar(pcm, ax=ax6, extend='max')
        plt.tight_layout()
    return(t_ons, t_offs)
#====================TIME TRACE OUTPUT==============
potential = 35
pointnumbers=range(100)
potentialist = np.linspace(-100, 200, 1 + (200 - (-100)) / 5)
potentialist = potentialist.astype('int')
def cp_outputs_folderwise(folderpath=foldername, pointnumbers=[1], potentialist=potentialist):
    df_datn_emplot = dir_mV_molNo(foldername=folderpath)
    df_specific = df_datn_emplot[df_datn_emplot['Point number'].isin(pointnumbers)]
    df_specific = df_specific[df_specific['Potential'].isin(potentialist)]
    out_total = pd.DataFrame()  # initiating empty output matrix
    for input_number in pointnumbers:
        df_specific_i = df_specific[df_specific['Point number']
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
                # potential = out_point[Point_number]['Potential'][i]
                potential = df_specific_i['Potential'][i]
                file_path_hdf5 = df_specific_i['filepath[.hdf5]'][i]
                # read and analyze change point
                out = changepoint_photonhdf5(file_path_hdf5, time_sect=25,
                                             pars=(1, 0.01, 0.99, 2),
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
def fcs_outputs_folderwise(folderpath=foldername, pointnumbers=[1], potentialist=potentialist):
    df_datn_emplot = dir_mV_molNo(foldername=folderpath)
    df_specific = df_datn_emplot[df_datn_emplot['Point number'].isin(
        pointnumbers)]
    df_specific = df_specific[df_specific['Potential'].isin(potentialist)]
    out_total = pd.DataFrame()  # initiating empty output matrix
    for input_number in pointnumbers:
        df_specific_i = df_specific[df_specific['Point number']
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
                # potential = out_point[Point_number]['Potential'][i]
                potential = df_specific_i['Potential'][i]
                file_path_hdf5 = df_specific_i['filepath[.hdf5]'][i]
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
                    print('fcs fitting of %s with potential %s didnot succeed' % (
                        Point_number, potential))
            out_total = pd.concat([out_total, out_point], axis=1)
    return out_total
# ============== Midpoint potential ==============
def Mid_potentials_slopem(folderpath=foldername, pointnumbers=range(5),
                          process='cp', plotting=True,
                          min_pot=40, min_pot_num=1):
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
        return(10**((x - a) / 0.059))
    def nernst_slopem(x, a, m):
        return(10**((x - a) / m))
    columns_E0 = ['Point number', 'E0_fit', 'E0_err']
    columns_E0_m = ['Point number', 'E0_fit', 'E0_err', 'slope', 'slope_error']
    E0_list = pd.DataFrame(index=None, columns=columns_E0)
    E0_m_list = pd.DataFrame(index=None, columns=columns_E0_m)
    #--figure initiation----
    if plotting == True:
        fig = plt.figure(figsize=(10,4))
        nrows=1; ncols=2;
        ax00 = plt.subplot2grid((nrows, ncols), (0,0))
        ax01 = plt.subplot2grid((nrows, ncols), (0,1))
        cmap = plt.get_cmap('jet')#jet_r
        N=len(out.columns.levels[0])
    for i in range(len(out.columns.levels[0])):
        point = out.columns.levels[0][i]
        point_output_tot = out[point].dropna()
        point_output = point_output_tot[point_output_tot['Potential'] >= min_pot] #select a potential threshold
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
            E0_fit, E0_var = curve_fit(nernst, E, t_ratio, p0=0.02)
            E0_err = np.sqrt(np.diag(E0_var));
            E0 = E0_fit[0]; E0_err=E0_err[0]
            #---------append to list---------
            E0_list_temp = pd.DataFrame([[point, E0, E0_err]], columns=columns_E0)
            E0_list=E0_list.append(E0_list_temp, ignore_index=True)
            #--------fitting nernst_slopem------
            E0_m_fit, E0_m_var = curve_fit(nernst_slopem, E, t_ratio, p0=[0.02, 0.059])
            E0_m_err = np.sqrt(np.diag(E0_m_var))
            E0_m = E0_m_fit[0];E0_m_fit_err = E0_m_err[0];
            slope_m = E0_m_fit[1]; slope_m_err = E0_m_err[1];
            E0_m_list_temp = pd.DataFrame([[point, E0_m, E0_m_fit_err, slope_m, slope_m_err]], columns=columns_E0_m)
            E0_m_list = E0_m_list.append(E0_m_list_temp, ignore_index=True)
            #-----plot------
            if plotting == True:
                color = cmap(float(i)/N)
                ax00.errorbar(point_output_tot['Potential'], point_output_tot['t_ratio'],
                         yerr=point_output_tot['t_ratioerr'], fmt='o', color=color, label=point, ms=5)#plot raw outputs
                ax00.plot(linspace(-25, max(potential)+10), nernst(0.001*linspace(-25, max(potential)+10), *E0_fit), color=color, linewidth=2)#color
                ax00.plot(linspace(-25, max(potential)+10), ones(len(linspace(-25, max(potential)+10))), '--k', linewidth=2)

                
                ax01.errorbar(point_output_tot['Potential'], point_output_tot['t_ratio'],
                         yerr=point_output_tot['t_ratioerr'], fmt='o', color=color, label=point, ms=5)#plot raw outputs
                ax01.plot(linspace(-25, max(potential)+10), nernst_slopem(0.001*linspace(-25, max(potential)+10), *E0_m_fit), color=color, linewidth=2)
                ax01.plot(linspace(-25, max(potential)+10), ones(len(linspace(-25, max(potential)+10))), '--k', linewidth=2)
                
                ax00.set_yscale('log')
                ax00.set_xlim(-25, 125)
                ax01.set_xlim(-25, 125)
                ax00.set_xlabel('$Potential [V]$', fontsize=20)
                ax00.set_ylabel('$T_{OFF}/T_{ON}$', fontsize=20)
                ax00.tick_params(axis='both', which='major', labelsize=16)
                ax00.set_title(r'$E=E_0 + 0.059log(\frac{t_{off}}{t_{on}})$')
                ax01.set_yscale('log')
                ax01.set_xlabel('$Potential [V]$', fontsize=20)
                ax01.set_ylabel('$T_{OFF}/T_{ON}$', fontsize=20)
                ax01.tick_params(axis='both', which='major', labelsize=16)
                ax01.set_title(r'$E=E_0 + m.log(\frac{t_{off}}{t_{on}})$')
                fig.tight_layout()
                #legend(bbox_to_anchor=(0.9, 0.3), fontsize=16)
    return(E0_list, E0_m_list)

def Mid_potentials_slopem_lmfit(folderpath=foldername, pointnumbers=range(5),
                          process='cp', plotting=True,
                          min_pot=40, min_pot_num=1):
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
        nrows=1; ncols=2;
        ax00 = plt.subplot2grid((nrows, ncols), (0,0))
        ax01 = plt.subplot2grid((nrows, ncols), (0,1))
        cmap = plt.get_cmap('jet')#jet_r
        N=len(out.columns.levels[0])
    for i in range(len(out.columns.levels[0])):
        point = out.columns.levels[0][i]
        point_output_tot = out[point].dropna()
        point_output = point_output_tot[point_output_tot['Potential'] >= min_pot] #select a potential threshold
        point_output.reset_index(drop=True, inplace=True)
        if len(point_output)>min_pot_num:
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
            out_params = str((res_nernst.params['a'],'value'));
            E0 = res_nernst.best_values['a']
            E0_err = float(out_params.split('+/- ')[1].split(', bounds')[0]);
            #---------append to list---------
            E0_list_temp = pd.DataFrame([[point, E0, E0_err]], columns=columns_E0)
            E0_list=E0_list.append(E0_list_temp, ignore_index=True)
            #--------fitting nernst_slopem------
            res_nernst_slop = nernst_slopem_mod.fit(t_ratio, params_ner_slop, x=E)
            E0_m = res_nernst_slop.best_values['a']
            out_params = str((res_nernst_slop.params['a'],'value'));
            E0_m_err = float(out_params.split('+/- ')[1].split(', bounds')[0]);
            slope_m = res_nernst_slop.best_values['m']
            out_params_m = str((res_nernst_slop.params['m'],'value'));
            slope_m_err = float(out_params_m.split('+/- ')[1].split(', bounds')[0]);
            E0_m_list_temp = pd.DataFrame([[point, E0_m, E0_m_err, slope_m, slope_m_err]], columns=columns_E0_m)
            E0_m_list = E0_m_list.append(E0_m_list_temp, ignore_index=True)
            #-----plot------
            if plotting == True:
                color = cmap(float(i)/N)
                ax00.errorbar(point_output_tot['Potential'], point_output_tot['t_ratio'],
                         yerr=point_output_tot['t_ratioerr'], fmt='o', color=color, label=point)#plot raw outputs
                ax00.plot(linspace(min(potential)-10, max(potential)+10), nernst(0.001*linspace(min(potential)-10, max(potential)+10), E0), color=color, linewidth=2.0)#color
                #plot(E*1000, nernst(E, *E0_fit), color=color, linewidth=2.0)

                
                ax01.errorbar(point_output_tot['Potential'], point_output_tot['t_ratio'],
                         yerr=point_output_tot['t_ratioerr'], fmt='o', color=color, label=point)#plot raw outputs
                ax01.plot(linspace(min(potential)-10, max(potential)+10), nernst_slopem(0.001*linspace(min(potential)-10, max(potential)+10), E0_m, slope_m), color=color, linewidth=2.0)
                ax00.set_yscale('log')
                ax00.set_xlabel('$Potential [V]$', fontsize=20)
                ax00.set_ylabel('$T_{OFF}/T_{ON}$', fontsize=20)
                ax00.tick_params(axis='both', which='major', labelsize=16)
                ax00.set_title(r'$E=E_0 + 0.059log(\frac{t_{off}}{t_{on}})$')
                ax01.set_yscale('log')
                ax01.set_xlabel('$Potential [V]$', fontsize=20)
                ax01.set_ylabel('$T_{OFF}/T_{ON}$', fontsize=20)
                ax01.tick_params(axis='both', which='major', labelsize=16)
                ax01.set_title(r'$E=E_0 + m.log(\frac{t_{off}}{t_{on}})$')
                tight_layout()
    return(E0_list, E0_m_list)