import numpy as np
import pandas as pd
import h5py
import os.path
from pylab import *
import os
import re
from scipy.optimize import curve_fit
global pointnumber
from lmfit import  Model, Parameter, Parameters
import matplotlib.pyplot as plt
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
            point_number=int(point_number)
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
            pt3_list = pt3_list.append(temp_pt3list, ignore_index = True)
    return(pt3_list)
def dir_mV_molNo(foldername=foldername):
    pt3_list = dir_mV_molNo_pt3(foldername=foldername)
    extensions = [".dat", ".datn"]
    columns_FCS = ['Point number', 'Potential', 'filename[FCS]', 'filepath[FCS]']
    FCS_list = pd.DataFrame(index=None, columns=columns_FCS)
    datn_em_list = pd.DataFrame()
    for i in range(len(pt3_list)):
        pt3_filename = pt3_list['filename[pt3]'][i]
        point_number = pt3_list['Point number'][i]
        point_number = int(point_number)
        potential = pt3_list['Potential'][i]
        potential = int(potential)
        pt3_path = pt3_list['filepath[pt3]'][i]
        for dirpath, dirnames, filenames in os.walk(foldername):
            for filename in [f for f in filenames if f.endswith(tuple(extensions))]:
                #add FCS to dataframe
                if pt3_filename[:-5] in filename and 'FCS' in filename and '_'+str(point_number)+'_' in filename:
                    fcs_file_path = os.path.join(dirpath, filename)
                    temp_FCS_list = pd.DataFrame([[point_number, potential, filename, fcs_file_path]], columns=columns_FCS)
                    FCS_list = FCS_list.append(temp_FCS_list, ignore_index=True)
                #add .datn and .em.plot
                if pt3_filename[:-3] in filename and 'datn' in filename:
                    filename_datn = filename;
                    datn_file_path = os.path.join(dirpath, filename_datn)
                    filename_emplot = filename+'.em.plot'
                    emplot_path = os.path.join(dirpath, filename_emplot)
                    filename_hdf5 = pt3_filename[:-3]+'hdf5'
                    hdf5_filepath = os.path.join(dirpath, filename_hdf5)
                    columns_datn_em=['Point number', 'Potential',
                                     'filename[.datn]', 'filepath[.datn]',
                                     'filename[.em.plot]', 'filepath[.em.plot]',
                                     'filename[.hdf5]', 'filepath[.hdf5]'
                                    ]
                    temp_datn_list = pd.DataFrame([[point_number, potential,
                                                    filename_datn, datn_file_path, 
                                                    filename_emplot, emplot_path, 
                                                    filename_hdf5, hdf5_filepath]],
                                                  columns=columns_datn_em)
                    datn_em_list = datn_em_list.append(temp_datn_list, ignore_index=True)
    pt3_list = pt3_list.sort(['Point number'], ascending=[1])
    pt3_list.reset_index(drop=True, inplace=True)
    FCS_list = FCS_list.sort(['Point number'], ascending=[1])
    FCS_list.reset_index(drop=True, inplace=True)
    datn_em_list = datn_em_list.sort(['Point number'], ascending=[1])
    datn_em_list.reset_index(drop=True, inplace=True)
    return(datn_em_list, FCS_list, pt3_list)
def point_list(foldername = foldername, pointnumbers=range(100)):
    df_datn_emplot, df_FCS, folderpath = dir_mV_molNo(foldername=foldername)
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
    df_datn_emplot, df_FCS, folderpath = dir_mV_molNo(foldername=foldername)
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
def check_missingFCSfiles(foldername=foldername):
    datn_em_list, FCS_list, pt3_list = dir_mV_molNo(foldername=foldername)
    for i in range(len(datn_em_list)):
        file_name = datn_em_list['filename[.datn]'][i]
        point_number = datn_em_list['Point number'][i]
        potential = datn_em_list['Potential'][i]
        FCS = FCS_list[FCS_list['Point number'].isin([point_number])]
        FCS = FCS[FCS['Potential'].isin([potential])]
        if FCS.empty:
            print('FCS of %s with Point number %s with potential %s doesn''nt exist' %(file_name,point_number, potential) )
    return
#=================get_point_specifics=======================
def get_point_specifics(foldername= foldername, input_potential=[0, 25, 50, 100], pointnumbers=[1]):
    """bin=1 in millisecond
    foldername should be given as r'D:\Research\...'
    """
    df_datn_emplot, df_FCS, folder = dir_mV_molNo(foldername)
    df_specific = df_datn_emplot[df_datn_emplot['Point number'].isin(pointnumbers)]#keep all the points that exist
    df_specific = df_specific[df_specific['Potential'].isin(input_potential)]; df_specific.reset_index(drop=True, inplace=True)
    df_datn_em_specific = df_specific

    df_fcs_specific = df_FCS[df_FCS['Point number'].isin(pointnumbers)]#keep all the points that exist
    df_fcs_specific = df_fcs_specific[df_fcs_specific['Potential'].isin(input_potential)]; df_fcs_specific.reset_index(drop=True, inplace=True)
    return(df_datn_em_specific, df_fcs_specific)
#---------TIME_TRACE_PLOT: given folder name, point number and list of potential, plot time traces at diff potentialof same molecule------
def time_trace_plot(foldername= foldername, input_potential=[0, 25, 50, 100],
                    pointnumbers=[1], x_lim_min=0, y_lim_min=0, x_lim_max=5, y_lim_max=6, bin=5, show_changepoint=True, figsize=(10, 8)):
    """bin=1 in millisecond
    foldername should be given as r'D:\Research\...'
    """
    df_datn_emplot, df_FCS, folder = dir_mV_molNo(foldername)
    df_specific = df_datn_emplot[df_datn_emplot['Point number'].isin(pointnumbers)]#keep all the points that exist
    df_specific = df_specific[df_specific['Potential'].isin(input_potential)];
    df_specific=df_specific.sort(['Potential'], ascending=[1]);df_specific.reset_index(drop=True, inplace=True)
    fig, ax = plt.subplots(figsize = figsize,sharex=True, sharey=True)
    subplots_adjust(hspace=0.000);
    for i in range(len(df_specific)):
        given_potential = df_specific['Potential'][i]
        #f_datn = df_specific['filepath[.datn]'][i]
        f_emplot = df_specific['filepath[.em.plot]'][i]
        f_hdf5 = df_specific['filepath[.hdf5]'][i]
        
        ax = subplot(len(df_specific),1,i+1)
        #df = pd.read_csv(f_datn, header=None)#Original data
        import h5py
        h5 = h5py.File(f_hdf5)
        unit = 12.5e-9;
        df = unit * h5['photon_data']['timestamps'][:]
        df=pd.DataFrame(df)
        h5.close()
        tt_length=max(df[0])-min(df[0])
        tt_length = round(tt_length, 0)
        binpts=tt_length*1000/bin
        df_hist = histogram(df[0], bins=binpts,range=(min(df[0]), max(df[0])))
        plot(df_hist[1][:-1], df_hist[0]/bin, 'b', label=str(given_potential)+" mV")#original data
        #----time trace overlapped with change-points
        if os.path.isfile(f_emplot):
            df = pd.read_csv(f_emplot, header=None, sep='\t') #change-point
            if show_changepoint == True:
                plot(df[0], df[1]*0.8/1000, 'r', linewidth=2, label='')#change-point analysis
        xlim(x_lim_min, x_lim_max)
        ylim(0, y_lim_max)# 1.5*max(df[1]/1000)
        # xticks([])
        yticks(range(0, y_lim_max, 2), fontsize=16)
        if i == len(df_specific)-1:
            # xticks(range(0, x_lim_max+1, 1), fontsize=16)
            ax.set_xlabel('time/s', fontsize=16)
        legend(fontsize=16, framealpha=0.5)
    fig.text(0.04, 0.5, 'Fluorescence(kcps)', va='center', rotation='vertical', fontsize=16)
    return(fig)
#===============getting all tags fo each photon===================
def lifetime_dig(foldername= foldername, input_potential=[0, 25, 50, 100], pointnumbers=[1], bintime = 10e-3):
    """bin=1 in millisecond
    foldername should be given as r'D:\Research\...'
    """
    def find_closest(A, target):
        # https://stackoverflow.com/questions/8914491/finding-the-nearest-value-and-return-the-index-of-array-in-python
        #A must be sorted
        idx = A.searchsorted(target)
        idx = np.clip(idx, 1, len(A)-1)
        left = A[idx-1]
        right = A[idx]
        idx -= target - left < right - target
        return idx
    df_datn_emplot, df_FCS, folder = dir_mV_molNo(foldername)
    df_specific = df_datn_emplot[df_datn_emplot['Point number'].isin(pointnumbers)]#keep all the points that exist
    df_specific = df_specific[df_specific['Potential'].isin(input_potential)];
    df_specific=df_specific.sort(['Potential'], ascending=[1]);df_specific.reset_index(drop=True, inplace=True)
    for i in range(len(df_specific)):
        given_potential = df_specific['Potential'][i]
        f_emplot = df_specific['filepath[.em.plot]'][i]
        f_hdf5 = df_specific['filepath[.hdf5]'][i]
        if os.path.isfile(f_emplot):
            df_emplot = pd.read_csv(f_emplot, header=None, sep='\t') #change-point
            df_tag = df_emplot[[0, 1]]
            df_tag = pd.DataFrame({'time': df_tag[0][1:],
                                   'diff_count': diff(df_tag[1])});
            df_tag = df_tag[df_tag['diff_count'] != 0]
            df_tag.reset_index(drop=True, inplace=True)
            h5 = h5py.File(f_hdf5)
            unit = h5['photon_data']['timestamps_specs']['timestamps_unit'][...]
            tcspc_unit = h5['photon_data']['nanotimes_specs']['tcspc_unit'][...]
            timestamps = unit * h5['photon_data']['timestamps'][...];
            nanotimes = tcspc_unit * h5['photon_data']['nanotimes'][...]
            mask = np.logical_and(timestamps>=min(df_tag['time']), timestamps<=max(df_tag['time']))
            timestamps = timestamps[mask]
            nanotimes = nanotimes[mask]
            h5.close()
            idx_closest = find_closest(timestamps, df_tag['time']);
            timestamps_closest = timestamps[idx_closest]
            dig_cp = np.digitize(timestamps, timestamps_closest);
            dig_uniq = np.unique(dig_cp)
            #dig_uniq_count = np.unique(uniq)
            bins = int((max(timestamps)-min(timestamps))/bintime)
            binned_trace = np.histogram(timestamps, bins=bins)
            dig_bin = np.digitize(timestamps, binned_trace[1][:-1]); # digitize every bintime
            
            df_dig = pd.DataFrame({'timestamps': timestamps,
                                   'dig_bin': dig_bin,
                                   'dig_cp': dig_cp,
                                   'count_rate': dig_cp,
                                   'nanotimes': nanotimes
                                  })
            df_dig['count_rate'] = df_dig['count_rate'].replace(dig_uniq, df_tag['time'])
            df_dig['count_rate'] = df_dig['count_rate'].replace(df_tag['time'].values, df_tag['diff_count'].values)
            real_count = [min(df_emplot[[1]].values), max(df_emplot[[1]].values)]
            tagged_count = [min(df_dig['count_rate']), max(df_dig['count_rate'])]
            df_dig['count_rate'] = df_dig['count_rate'].replace(tagged_count, real_count)
            return(df_dig)
#===FCS fit functions AND FCS plot for a molecule(s) at different potentials=======
def FCS_mono_fit(filename,tmin,tmax):
    df_fcs = pd.read_csv(filename, index_col=False, names=None, skiprows=1, header=None, sep='\s+');
    df_fcs = df_fcs[df_fcs[0]>=tmin];
    df_fcs = df_fcs[df_fcs[0]<=tmax];
    xdata=df_fcs[0];
    ydata=df_fcs[1];
    def mono_exp(x, A1, A2, t1):
        return((A1+A2*exp(-x/t1)))

    monofit, pcov_mono = curve_fit(mono_exp, xdata, ydata, p0 = [10, 1, 1], bounds=(0, np.inf))
    return(monofit)
def FCS_bi_fit(filename,tmin,tmax):
    df_fcs = pd.read_csv(filename, index_col=False, names=None, skiprows=1, header=None, sep='\s+');
    df_fcs = df_fcs[df_fcs[0]>=tmin];
    df_fcs = df_fcs[df_fcs[0]<=tmax];
    xdata=df_fcs[0];
    ydata=df_fcs[1];
    def biexp(x, A1, A2, t1, A3, t2):
        return(A1+A2*exp(-x/t1))+A3*exp(-x/t2)
    try:
        bifit, pcov_bi = curve_fit(biexp, xdata, ydata, p0 = [10, 1, 1, 0.5, 1], bounds=(0, np.inf))
    except RuntimeError:
        bifit = [NaN,NaN,NaN,NaN,NaN]
        print('Runtime Error %s' %filename)
    return(bifit)

def FCS_plot(foldername= foldername, input_potential=[0, 25, 50, 100],
                    pointnumbers=[1], tmin=0.005, tmax=1000, kind='bi'):
    """
    foldername should be given as r'D:\Research\...'
    """
    for i in range(len(pointnumbers)):
        input_number = pointnumbers[i]
        df_datn_emplot, df_FCS, folder = dir_mV_molNo(foldername)
        df_specific = df_FCS[df_FCS['Point number']==input_number]
        df_specific = df_specific[df_specific['Potential'].isin(input_potential)]; df_specific.reset_index(drop=True, inplace=True)
        fig, ax = plt.subplots(figsize = (10, 8),sharex=True, sharey=True)
        subplots_adjust(hspace=0.000);
        for i in range(len(df_specific)):
            given_potential = df_specific['Potential'][i]
            print(given_potential)
            df_specific_V = df_specific[df_specific['Potential'] == given_potential]
            f_FCS_path = df_specific_V['filepath[FCS]'].values[0]
            f_FCS = f_FCS_path
            df_fcs = pd.read_csv(f_FCS, index_col=False, names=None, skiprows=1, header=None, sep='\s+');
            df_fcs = df_fcs[df_fcs[0]>=tmin]; df_fcs = df_fcs[df_fcs[0]<=tmax];
            xdata=df_fcs[0];ydata=df_fcs[1];
            ax.plot(xdata,ydata, label=str(input_number)+':'+str(given_potential)+'mV')
            ax.set_xscale('log')

            def biexp(x, A1, A2, t1, A3, t2): #fit bi-exp
                return(A1+A2*exp(-x/t1))+A3*exp(-x/t2)
            def mono_exp(x, A1, A2, t1): #fit mono
                return((A1+A2*exp(-x/t1)))
            if kind in ['mono', 'Mono']: #determine what you want to plot
                monofit = FCS_mono_fit(f_FCS, tmin, tmax)
                plt.plot(xdata, mono_exp(xdata, *monofit), color = 'b', linewidth=2.0)
                print('g(t) = %s + %s * exp(-t/%s)' %(monofit[0], monofit[1], monofit[2]))
            if kind in ['bi', 'Bi']:
                bifit = FCS_bi_fit(f_FCS, tmin, tmax)
                plt.plot(xdata, biexp(xdata, *bifit), color = 'r', linewidth=2.0)
                print('g(t) = %s + %s * exp(-t/%s) + %s * exp(-t/%s)' %(bifit[0], bifit[1], bifit[2], bifit[3], bifit[4]))
            if kind in ['both', 'Both']:
                monofit = FCS_mono_fit(f_FCS, tmin, tmax)
                plt.plot(xdata, mono_exp(xdata, *monofit), color = 'b', linewidth=2.0)
                print('g(t) = %s + %s * exp(-t/%s)' %(monofit[0], monofit[1], monofit[2]))
                bifit = FCS_bi_fit(f_FCS, tmin, tmax)
                plt.plot(xdata, biexp(xdata, *bifit), color = 'r', linewidth=2.0)
                print('g(t) = %s + %s * exp(-t/%s) + %s * exp(-t/%s)' %(bifit[0], bifit[1], bifit[2], bifit[3], bifit[4]))
            plt.xscale('log')
    return()
#==============ON/OFF times from changepoint and FCS==================
def t_on_off_fromCP(f_emplot):
    #expt data
    # df = pd.read_csv(f_datn, header=None)
    # binpts=5000; mi=min(df[0]); ma=mi+10;
    # df_hist = histogram(df[0], bins=binpts, range=(mi, ma))
    #change point
    df = pd.read_csv(f_emplot, header=None, sep='\t')
    df_diff= diff(df[0])
    #calculating Ton and Toff
    df_tag = df[[0, 1]]; # df_ton = df_ton[1:]
    df_tag = pd.DataFrame([df_tag[0][1:], diff(df_tag[1])]); df_tag = df_tag.T;
    df_tag.columns = [0, 1];
    df_tag = df_tag[df_tag[1] != 0];
    df_tag.reset_index(drop=True, inplace=True);
    if df_tag[1][0] < 0:
        df_tag = df_tag[1:]
        df_tag.reset_index(drop=True, inplace=True);
    df_tag_pos = df_tag[df_tag[1]==max(df_tag[1])];df_tag_pos.reset_index(drop=True, inplace=True);
    df_tag_neg = df_tag[df_tag[1]==min(df_tag[1])];df_tag_neg.reset_index(drop=True, inplace=True);

    df_ton = df_tag_neg[0]-df_tag_pos[0];df_ton.reset_index(drop=True, inplace=True);
    t1=df_tag_pos[0][1:]; t1.reset_index(drop=True, inplace=True);
    t2=df_tag_neg[0]; t1.reset_index(drop=True, inplace=True);
    df_toff = t1 - t2; df_toff = df_toff[:df_toff.shape[0]-2];df_ton.reset_index(drop=True, inplace=True)
    # remove NAN values:
    df_ton = df_ton[~np.isnan(df_ton)];
    df_toff = df_toff[~np.isnan(df_toff)];
    average_ton = 1000 * np.average(df_ton);# also converts to millisecond
    average_ton = np.round(average_ton, 2)
    lambda_ton = 1/average_ton;
    lambda_ton_low = lambda_ton * (1-(1.96/np.sqrt(len(df_ton))))
    lambda_ton_upp = lambda_ton * (1+(1.96/np.sqrt(len(df_ton))))
    average_ton_err = (1/lambda_ton_low) - (1/lambda_ton_upp);
    average_ton_err = np.round(average_ton_err, 2)

    average_toff = 1000 * np.average(df_toff);# also converts to millisecond
    average_toff = np.round(average_toff, 2)
    lambda_toff = 1/average_toff;
    lambda_toff_low = lambda_toff * (1-(1.96/np.sqrt(len(df_toff))))
    lambda_toff_upp = lambda_toff * (1+(1.96/np.sqrt(len(df_toff))))
    average_toff_err = (1/lambda_toff_low) - (1/lambda_toff_upp);
    average_toff_err = np.round(average_toff_err, 2)
    return(df_ton, df_toff, average_ton, average_toff, average_ton_err, average_toff_err)
# ---HISTOGRAM ON/OFF: given folder name, potential and list of point number, plot histogram at certain potentialof ----
def histogram_on_off_1mol(foldername= foldername, input_potential=[100], pointnumbers=[1],
                          bins_on=50, range_on=[0, 0.2], bins_off=50, range_off=[0, 0.5], plotting=False):
    df_datn_emplot, df_FCS, folder = dir_mV_molNo(foldername)
    df_specific = df_datn_emplot[df_datn_emplot['Point number'].isin(pointnumbers)]#keep all the points that exist
    df_specific = df_specific[df_specific['Potential'].isin(input_potential)]; df_specific.reset_index(drop=True, inplace=True)
    f_emplot_path = 'x'; f_datn_path='x'; t_ons=[];t_offs=[];n_on = []; n_off = []
    if not df_specific.empty:
        f_emplot_path = df_specific['filepath[.em.plot]'].values[0]
    if os.path.isfile(f_emplot_path):
        try:
            df_ton, df_toff, average_ton, average_toff, average_ton_err, average_toff_err = t_on_off_fromCP(f_emplot_path)
            t_ons = np.array(df_ton);
            t_offs = np.array(df_toff)
            n_on = []; n_off = []
            n_on,bins_on = histogram(t_ons, range=range_on,bins=bins_on);
            n_off,bins_off = histogram(t_offs, range=range_off,bins=bins_off)
            if plotting == True:
                fig, axes = plt.subplots(1, 2, figsize= (10,4))
                n_on,bins_on,patches = axes[0].hist(t_ons, range=range_on, bins=bins_on, color='k', alpha=0.5)
                n_on,bins_on,patches = axes[0].hist(t_ons, range=range_on, bins=bins_on, color='k', histtype='step')
                axes[0].set_xlabel(r'$\tau_{on}/s$')
                axes[0].set_ylabel('PDF')
                axes[0].set_xlim(0, None)
                #axes[0].set_yscale('log')
                axes[0].set_title("ON time histogram at %s mV" %input_potential[0])
                n_off,bins_off,patches = axes[1].hist(t_offs, range=range_off,bins=bins_off, color='k', alpha=0.5)
                n_off,bins_off,patches = axes[1].hist(t_offs, range=range_off,bins=bins_off, color='k', histtype='step')
                axes[1].set_xlabel(r'$\tau_{off}/s$')
                axes[1].set_ylabel('PDF')
                axes[1].set_xlim(0, None)
                #axes[1].set_yscale('log')
                axes[1].set_title("OFF time histogram at %s mV" %input_potential[0])
        except:
            print('em.plot file: %s doesn''t contain proper data' %f_emplot_path)
            #potential=np.nan # This row will be removed in later processing
            pass
    return(t_ons, t_offs, n_on, bins_on, n_off, bins_off)

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
def hist2D_on_off(foldername=foldername, input_potential=[100], pointnumbers=[24], bins_on=40, range_on=[0, 0.01], bins_off=50, range_off=[0, 1], x_shift=10, plots=True, figsize=(16, 8)):
    t_ons = []; t_offs=[];
    for i in pointnumbers:
        t_on_temp, t_off_temp,n_on, bins_on, n_off, bins_off = histogram_on_off_1mol(foldername= foldername, input_potential=input_potential, pointnumbers=[i], 
            bins_on=bins_on, range_on=range_on, bins_off=bins_off, range_off=range_off, plotting=False)
        t_ons = np.concatenate((t_ons, t_on_temp), axis=0)
        t_offs = np.concatenate((t_offs, t_off_temp), axis=0)

    t_ons=pd.Series(t_ons);t_offs=pd.Series(t_offs)
    t_on_shifted_1 = t_ons.shift(+1) ## shift up
    t_on_delay_1 = pd.DataFrame([t_on_shifted_1, t_ons]); t_on_delay_1=t_on_delay_1.T
    t_on_delay_1 = t_on_delay_1.dropna();
    t_off_shifted_1 = t_offs.shift(+1) ## shift up

    t_on_shifted_x = t_ons.shift(+x_shift) ## shift up
    t_off_shifted_x = t_offs.shift(+x_shift) ## shift up
    print('Number of on events: %d' %len(t_ons))
    print('Number of off events: %d' %len(t_offs))
    if plots==True:
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
        pcm=ax6.pcolormesh(Ex_off_x, Ey_off_x, C_off_diff, norm=mpl.colors.SymLogNorm(linthresh=0.1, linscale=0.1,vmin=C_off_diff.min(), vmax=C_off_diff.max()), cmap=colormap)
        fig.colorbar(pcm, ax=ax6, extend='max')
        plt.tight_layout()
    return(t_ons, t_offs)

#====================TIME TRACE OUTPUT==============
pointnumbers = linspace(1, 40, 40);pointnumbers = pointnumbers.astype(int);
potentialist = linspace(-100, 200, 1+(200-(-100))/5);
def timetrace_outputs_folderwise(folderpath=foldername, pointnumbers=[1], potentialist=potentialist, kind=['timetrace']):
    '''
    kind=['timetrace'] or kind=['fcs']'''
    df_datn_emplot, df_FCS, pt3_list = dir_mV_molNo(foldername=folderpath)
    df_datn_emplot = df_datn_emplot[df_datn_emplot['Point number'].isin(pointnumbers)]#keep all the points that exist
    df_datn_emplot = df_datn_emplot[df_datn_emplot['Potential'].isin(potentialist)]
    out_point = []#pd.DataFrame()
    out_total = pd.DataFrame()#initiating empty output matrix
    for input_number in pointnumbers:
        df_datnem_specific = df_datn_emplot[df_datn_emplot['Point number']==input_number]
        df_datnem_specific = df_datnem_specific.sort(['Potential'], ascending=[1])
        df_datnem_specific.reset_index(drop=True, inplace=True)

        if not df_datnem_specific.empty:
            #---------Create Pandas array to save outputs----------
            indices = np.ones(7); indices=indices.astype(str)
            Point_number = 'Point_'+str(input_number)
            indices[:]=Point_number
            subgroup = ['Potential','t_onav', 't_onaverr', 't_offav','t_offaverr','t_ratio', 't_ratioerr']
            arrays = [indices, subgroup]
            col = pd.MultiIndex.from_arrays(arrays)
            length=(len(df_datnem_specific))#for defining dimension of out_mat
            out_point = pd.DataFrame(zeros((length, len(subgroup))), columns=col)#create zeroes which will be replaced by proper values

            #---------Pandas array created to save outputs----
            out_point[Point_number]['Potential']=df_datnem_specific['Potential']
            poten_array = [];t_onav=[];t_onaverr=[]; t_offav=[]; t_offaverr=[]; t_ratio=[]; t_ratioerr=[] #Empty forms for timetrace output
            for i in range(length):
                potential = out_point[Point_number]['Potential'][i]
                df_datn_path = df_datnem_specific['filepath[.datn]'][i]
                df_em_path = df_datnem_specific['filepath[.em.plot]'][i]
                df_emplot_filename = df_datnem_specific['filename[.em.plot]'][i]

                if os.path.isfile(df_em_path):
                    try:
                        df_ton, df_toff, average_ton, average_toff, average_ton_err, average_toff_err = t_on_off_fromCP(df_em_path)
                        ratio_off_on = average_toff/average_ton;
                        ratio_off_on_err = (average_toff/average_ton)*sqrt(((average_toff_err/average_toff)**2)+((average_ton_err/average_ton)**2))
                    except:
                        print('em.plot file: %s doesn''t contain proper data' %df_emplot_filename)
                        potential=np.nan # This row will be removed in later processing
                        pass
                else:
                    print('em.plot file of %s with potential %s doesn''t exist' %(Point_number, potential))
                    potential=np.nan #, # This row will be removed in later processing
                poten_array.append(potential)
                t_onav.append(average_ton)
                t_onaverr.append(average_ton_err)#needs to be changed
                t_offav.append(average_toff)
                t_offaverr.append(average_toff_err)#needs to be changed
                t_ratio.append(ratio_off_on)
                t_ratioerr.append(ratio_off_on_err)#needs to be changed
            df_create=array([poten_array, t_onav, t_onaverr, t_offav, t_offaverr, t_ratio, t_ratioerr])
            df_create=df_create.astype(float64)#;a=pd.DataFrame(a)
            df_create = pd.DataFrame(df_create.T, columns=subgroup)
            out_point[Point_number]=df_create
            out_total=pd.concat([out_total, out_point], axis=1);
    return(out_total)
#=====================fcs output=====================
potential = 35
def t_on_off_fromFCS(df_fcs, tmin=0.05,tmax=1000, V= potential, V_th=40):
    df_fcs = pd.read_csv(df_fcs, index_col=False, names=None, skiprows=1, header=None, sep='\s+');
    df_fcs = df_fcs[df_fcs[0]>=tmin];
    df_fcs = df_fcs[df_fcs[0]<=tmax];
    xdata=df_fcs[0];
    ydata=df_fcs[1];
    def mono_exp(x, A, t_ac):
        return (A*exp(-x/t_ac))
    def bi_exp(x, A1, t1, A2, t2):
        return (A1*exp(-x/t1) + A2*exp(-x/t2))
    if V >= V_th:
        monofit, pcov = curve_fit(mono_exp, xdata, ydata, p0 = [1, 1], bounds=(0, np.inf))
        perr = np.sqrt(np.diag(pcov))
        A=monofit[0]; t_ac = monofit[1]; t_er = perr[1]
        toff = t_ac*(1+A); ton = t_ac*(1+(1/A));
        toff_er = t_er*(1+A); ton_er = t_er*(1+(1/A));
    else:
        bifit, pcov = curve_fit(bi_exp, xdata, ydata, p0 = [1, 1, 1, 1], bounds=(0, np.inf))
        perr = np.sqrt(np.diag(pcov))
        if bifit[1]>bifit[3]:
            A=bifit[0]; t_ac = bifit[1]; t_er = perr[1]
        else:
            A=bifit[2]; t_ac = bifit[3]; t_er = perr[3]
        toff = t_ac*(1+A); ton = t_ac*(1+(1/A));
        toff_er = t_er*(1+A); ton_er = t_er*(1+(1/A));
    return(ton, toff, ton_er, toff_er)#, t_on_err, t_off_err
def fcs_outputs_folderwise(folderpath=foldername, pointnumbers=[1], potentialist=potentialist, V_th=40):
    df_datn_emplot, df_FCS, pt3_list = dir_mV_molNo(foldername=folderpath)
    # df_FCS = df_FCS[df_FCS['Point number'].isin(pointnumbers)]
    # df_FCS = df_FCS[df_FCS['Point number'].isin(potentialist)
    out_total = pd.DataFrame()#initiating empty output matrix
    for input_number in pointnumbers:
        df_fcs_specific = df_FCS[df_FCS['Point number']==input_number]
        df_fcs_specific = df_fcs_specific.sort(['Potential'], ascending=[1])
        df_fcs_specific.reset_index(drop=True, inplace=True)

        if not df_fcs_specific.empty:
            #---------Create Pandas array to save outputs----------
            indices = np.ones(7); indices=indices.astype(str)
            Point_number = 'Point_'+str(input_number)
            indices[:]=Point_number
            subgroup = ['Potential','t_onav', 't_onaverr', 't_offav','t_offaverr','t_ratio', 't_ratioerr']
            arrays = [indices, subgroup]
            col = pd.MultiIndex.from_arrays(arrays)
            length=(len(df_fcs_specific))#for defining dimension of out_mat
            out_point = pd.DataFrame(zeros((length, len(subgroup))), columns=col)#create zeroes which will be replaced by proper values

            #---------Pandas array created to save outputs----
            out_point[Point_number]['Potential']=df_fcs_specific['Potential']
            poten_array = [];t_onav=[];t_onaverr=[]; t_offav=[]; t_offaverr=[]; t_ratio=[]; t_ratioerr=[] #Empty forms for timetrace output
            for i in range(length):
                potential = out_point[Point_number]['Potential'][i]
                df_fcs_path = df_fcs_specific['filepath[FCS]'][i]

                if os.path.isfile(df_fcs_path):
                    try:
                        average_ton, average_toff, average_ton_err, average_toff_err = t_on_off_fromFCS(df_fcs_path, tmin=0.05,tmax=1000, V= potential, V_th=40)
                        ratio_off_on = average_toff/average_ton;
                        ratio_off_on_err = (average_toff/average_ton)*sqrt(((average_toff_err/average_toff)**2)+((average_ton_err/average_ton)**2))
                    except:
                        print('fcs file: %s doesn''t contain proper data' %df_fcs_path)
                        potential=np.nan # This row will be removed in later processing
                        pass
                else:
                    print('fcs file of %s with potential %s doesn''t exist' %(Point_number, potential))
                    potential=np.nan #, # This row will be removed in later processing
                poten_array.append(potential)
                t_onav.append(average_ton)
                t_onaverr.append(average_ton_err)#needs to be changed
                t_offav.append(average_toff)
                t_offaverr.append(average_toff_err)#needs to be changed
                t_ratio.append(ratio_off_on)
                t_ratioerr.append(ratio_off_on_err)#needs to be changed
            df_create=array([poten_array, t_onav, t_onaverr, t_offav, t_offaverr, t_ratio, t_ratioerr])
            df_create=df_create.astype(float64)#;a=pd.DataFrame(a)
            df_create = pd.DataFrame(df_create.T, columns=subgroup)
            out_point[Point_number]=df_create
            out_total=pd.concat([out_total, out_point], axis=1);
    return(df_FCS, out_total)
#------------Midpoint potential---------------------
def Mid_potentials(folderpath=foldername, pointnumbers=range(20), plotting=True, min_pot=40):
    timetrace_output = timetrace_outputs_folderwise(folderpath=folderpath, pointnumbers=pointnumbers, potentialist=potentialist)
    def nernst(x, a):
        '''x is potential
        a: E0/midpoint potential(parameter)
        returns ratio(t_oxd/t_red)'''
        return(10**((x - a) / 0.059))
    columns_E0 = ['Point number', 'E0_fit', 'E0_err']
    E0_list = pd.DataFrame(index=None, columns=columns_E0)
    cmap = plt.get_cmap('hsv')#jet_r
    N=len(timetrace_output.columns.levels[0])
    for i in range(len(timetrace_output.columns.levels[0])):
        point = timetrace_output.columns.levels[0][i]
        point_output_tot = timetrace_output[point].dropna()
        point_output = point_output_tot[point_output_tot['Potential'] >= min_pot] #select a potential threshold
        if len(point_output)>2:
            potential = point_output['Potential']
            t_onav = point_output['t_onav']
            t_onaverr = point_output['t_onaverr']
            t_offav = point_output['t_offav']
            t_offaverr = point_output['t_offaverr']
            t_ratio = point_output['t_ratio']
            t_ratioerr = point_output['t_ratioerr']
            #--------fitting--------------
            E = potential*0.001 #converting to mV
            E0_fit, E0_var = curve_fit(nernst, E, t_ratio, p0=0.02)
            E0_err = np.sqrt(np.diag(E0_var));
            E0 = E0_fit[0]; E0_err=E0_err[0]
            #---------append to list---------
            E0_list_temp = pd.DataFrame([[point, E0, E0_err]], columns=columns_E0)
            E0_list=E0_list.append(E0_list_temp, ignore_index=True)
            #-----plot------
            if plotting == True:
                color = cmap(float(i)/N)
                errorbar(point_output_tot['Potential'], point_output_tot['t_ratio'],
                         yerr=point_output_tot['t_ratioerr'], fmt='o', color=color, label=point)#plot raw outputs
                plot(1000*linspace(-0.025, 0.11), nernst(linspace(-0.025, 0.11), *E0_fit), color=color, linewidth=2.0)
                yscale('log')
                xlabel('$Potential [V]$', fontsize=20)
                ylabel('$T_{OFF}/T_{ON}$', fontsize=20)
                tick_params(axis='both', which='major', labelsize=16)
                tight_layout()
                #legend(bbox_to_anchor=(0.9, 0.3), fontsize=16)
    return(E0_list)
#================Autocorrelation function==========================