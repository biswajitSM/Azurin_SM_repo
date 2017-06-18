import numpy as np
import pandas as pd
import os.path
from pylab import *
import glob
import os
import re
from xlwt import Workbook
from scipy.optimize import curve_fit
global pointnumber
from numpy import sqrt, pi, exp, linspace, loadtxt
from lmfit import  Model, Parameter, Parameters
import matplotlib.pyplot as plt
#--------------Get the pointnumber, datn, emplot, FCS files with their filepath in a "GIVEN FOLDER"---------------------
foldername = r'/home/biswajit/Research/Reports_ppt/reports/AzurinSM-MS4/data/201702_S101toS104/S101d14Feb17_60.5_635_A2_CuAzu655'
def dir_mV_molNo_temp(foldername=foldername):
    """input: Path of the Folder name as: foldername = r'D:\Research\Experimental\Analysis\2017analysis\201702\Analysis_Sebby_March_2017\S101d14Feb17_60.5_635_A2_CuAzu655'
	-----
	Output: df_datn_emplot, df_FCS, foldername
	e.g. df_datn_emplot, df_FCS, foldername = dir_mV_molNo(foldername)
	df_datn_emplot gives the list of points with their number, potential values and pathdirectory
	"""
    os.chdir(foldername)
    extensions = [".datn", ".dat"] #file extensions we are interested in
    columns=['Point number', 'Potential', 'filename[.datn]', 'filepath[.datn]','filename[.em.plot]', 'filepath[.em.plot]']
    df_datn_emplot = pd.DataFrame(index=None, columns=columns)
    columns_FCS = ['Point number', 'Potential', 'filename[FCS]', 'filepath[FCS]']
    df_FCS = pd.DataFrame(index=None, columns=columns_FCS)
    pointnumber = str(0)
    for dirpath, dirnames, filenames in os.walk("."):
        for filename in [f for f in filenames if f.endswith(tuple(extensions))]:
            #looking through all folders
            string_1 = 'mV'
            string_2 = 'FCS'
            position_FCS = filename.find(string_2)
            #determine whether or not it is FCS file
            if position_FCS in [-1]: #no FCS in name --> time trace file
                number1 = filename[-11:-10] #first number
                number2 = filename[-10:-9] #second number
                if number1.isdigit(): #check if 1 or 2-digit number
                    pointnumber = int(number1 + number2)
                elif number2.isdigit():
                    pointnumber = int(number2)
                position_potential = filename.find(string_1) #determine the place where the potential number is in the filename
                if position_potential in [-1]: #mV does not appear in the name
                    print('Potential in %s is not properly defined.' %filename)
                else:  #determine the potential (between -999 and +999mV)
                    pot_number1 = filename[position_potential-1:position_potential]
                    pot_number2 = filename[position_potential-2:position_potential-1]
                    pot_number3 = filename[position_potential-3:position_potential-2]
                    pot_number4 = filename[position_potential-4:position_potential-3]
                    pot_number5 = filename[position_potential-5:position_potential-4]
                if pot_number2 in ['_']: #filename X_voltagemV_Y
                    potentential = pot_number1
                elif pot_number3 in ['_']:
                    potentential = pot_number2 + pot_number1
                elif pot_number4 in ['_']:
                    potentential = pot_number3 + pot_number2 + pot_number1
                elif pot_number5 in ['_']:
                    potentential = pot_number4 + pot_number3 + pot_number2 + pot_number1
                potentential = int(potentential) #reading the potential
                if filename.endswith('.datn'):
                    f_datn = filename
                    f_datn_path = os.path.abspath(dirpath+'/'+f_datn)#change '/' to '\' in windows
                    f_emplot = re.sub('.datn$','.datn.em.plot',f_datn)
                    f_emplot_path = os.path.abspath(dirpath+'/'+f_emplot)
                    temp_output = pd.DataFrame([[pointnumber, int(potentential), f_datn, f_datn_path, f_emplot, f_emplot_path]], columns=columns)
                    df_datn_emplot = df_datn_emplot.append(temp_output, ignore_index=True)

            else:
                point_number1_FCS = filename[position_FCS-2:position_FCS-1]
                point_number2_FCS = filename[position_FCS-3:position_FCS-2]
                if point_number2_FCS in ['_']: #filename X_voltagemV_Y
                    pointnumberFCS = int(point_number1_FCS)
                else:
                    pointnumberFCS = int(point_number2_FCS + point_number1_FCS)
                position_potential_FCS = filename.find(string_1)
                if position_potential_FCS in [-1]: #mV does not appear in the name
                    print('Potential in %s is not properly defined.' %filename)
                else:  #determine the potential (between -999 and +999mV)
                    pot_number1_FCS = filename[position_potential_FCS-1:position_potential_FCS]
                    pot_number2_FCS = filename[position_potential_FCS-2:position_potential_FCS-1]
                    pot_number3_FCS = filename[position_potential_FCS-3:position_potential_FCS-2]
                    pot_number4_FCS = filename[position_potential_FCS-4:position_potential_FCS-3]
                    pot_number5_FCS = filename[position_potential_FCS-5:position_potential_FCS-4]
                if pot_number2_FCS in ['_']: #filename X_voltagemV_Y
                    potentential_FCS = pot_number1_FCS
                elif pot_number3_FCS in ['_']:
                    potentential_FCS = pot_number2_FCS + pot_number1_FCS
                elif pot_number4_FCS in ['_']:
                    potentential_FCS = pot_number3_FCS + pot_number2_FCS + pot_number1_FCS
                elif pot_number5_FCS in ['_']:
                    potentential_FCS = pot_number4_FCS + pot_number3_FCS + pot_number2_FCS + pot_number1_FCS
                potential_FCS = int(potentential_FCS) #reading the potential
                f_FCS = filename
                f_FCS_path = os.path.abspath(dirpath+'/'+f_FCS)
                temp_outputFCS = pd.DataFrame([[int(pointnumberFCS), int(potentential_FCS), f_FCS, f_FCS_path]], columns=columns_FCS)
                df_FCS = df_FCS.append(temp_outputFCS, ignore_index=True)

    os.chdir(foldername)
    return(df_datn_emplot, df_FCS, foldername)
def dir_mV_molNo_pt3(foldername=foldername):
    os.chdir(foldername)
    extensions_pt3 = [".pt3"] #file extensions we are interested in
    string_pt3 = '.pt3'
    string_mV = 'mV'
    columns = ['Point number', 'Potential', 'filename[pt3]', 'filepath[pt3]']
    pt3_list = pd.DataFrame(index=None, columns=columns)
    for dirpath, dirnames, filenames in os.walk("."):
        for filename in [f for f in filenames if f.endswith(tuple(extensions_pt3))]:
            position_num = filename.find(string_pt3)
            pos_num_val_1 = filename[position_num-1]
            pos_num_val_2 = filename[position_num-2]
            pos_num_val_3 = filename[position_num-3]
            if not pos_num_val_1.isdigit():
                print('Point number in %s is not properly placed(found): ' %filename)
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
                print('potential value in %s is not properly defined' %filename)
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
            file_pt3 = filename;
            temp_pt3list = pd.DataFrame([[point_number, potentail_val, filename, foldername+'/'+dirpath+'/'+filename]], columns=columns)
            pt3_list = pt3_list.append(temp_pt3list, ignore_index = True)
    return(pt3_list)
def dir_mV_molNo(foldername=foldername):
    pt3_list = dir_mV_molNo_pt3(foldername=foldername)
    extensions = [".dat", ".datn"]
    columns_FCS = ['Point number', 'Potential', 'filename[FCS]', 'filepath[FCS]']
    FCS_list = pd.DataFrame(index=None, columns=columns_FCS)
    columns_datn_em=['Point number', 'Potential', 'filename[.datn]', 'filepath[.datn]','filename[.em.plot]', 'filepath[.em.plot]']
    datn_em_list = pd.DataFrame(index=None, columns=columns_datn_em)
    for i in range(len(pt3_list)):
        pt3_filename = pt3_list['filename[pt3]'][i]
        point_number = pt3_list['Point number'][i]
        point_number = int(point_number)
        potential = pt3_list['Potential'][i]
        potential = int(potential)
        pt3_path = pt3_list['filepath[pt3]'][i]
        for dirpath, dirnames, filenames in os.walk("."):
            for filename in [f for f in filenames if f.endswith(tuple(extensions))]:
                if pt3_filename[:-5] in filename and 'FCS' in filename and '_'+str(point_number)+'_' in filename:
                    temp_FCS_list = pd.DataFrame([[point_number, potential, filename, foldername+'/'+dirpath+'/'+filename]], columns=columns_FCS)
                    FCS_list = FCS_list.append(temp_FCS_list, ignore_index=True)
                if pt3_filename[:-3] in filename and 'datn' in filename:
                    filename_datn = filename
                    filename_emplot = filename+'.em.plot'
                    temp_datn_list = pd.DataFrame([[point_number, potential, filename_datn, foldername+'/'+dirpath+'/'+filename_datn,
                                                   filename_emplot, foldername+'/'+dirpath+'/'+filename_emplot]], columns=columns_datn_em)
                    datn_em_list = datn_em_list.append(temp_datn_list, ignore_index=True)
    pt3_list = pt3_list.sort(['Point number'], ascending=[1])
    FCS_list = FCS_list.sort(['Point number'], ascending=[1])
    datn_em_list = datn_em_list.sort(['Point number'], ascending=[1])
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
#---------get_point_specifics-----
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
                    pointnumbers=[1], x_lim_min=0, y_lim_min=0, x_lim_max=5, y_lim_max=6, bin=5, show_changepoint=True):
    """bin=1 in millisecond
    foldername should be given as r'D:\Research\...'
    """
    df_datn_emplot, df_FCS, folder = dir_mV_molNo(foldername)
    df_specific = df_datn_emplot[df_datn_emplot['Point number'].isin(pointnumbers)]#keep all the points that exist
    df_specific = df_specific[df_specific['Potential'].isin(input_potential)];
    df_specific=df_specific.sort(['Potential'], ascending=[1]);df_specific.reset_index(drop=True, inplace=True)
    fig, ax = plt.subplots(figsize = (10, 8),sharex=True, sharey=True)
    subplots_adjust(hspace=0.000);
    for i in range(len(df_specific)):
        given_potential = df_specific['Potential'][i]
        #print(given_potential)
        #df_specific_V = df_specific[df_specific['Potential'] == given_potential];df_specific_V.reset_index(drop=True, inplace=True)
        f_datn_path = df_specific['filepath[.datn]'][i]
        f_emplot_path = df_specific['filepath[.em.plot]'][i]
        f_datn = f_datn_path
        f_emplot = f_emplot_path

        ax = subplot(len(df_specific),1,i+1)
        df = pd.read_csv(f_datn, header=None)#Original data
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
#--------FCS fit functions .....AND......FCS plot for a molecule(s) at different potentials-----------------
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
#-----------------------------ON/OFF times from changepoint and FCS------------------------------------
def t_on_off_fromCP(f_datn, f_emplot):
    #expt data
    # df = pd.read_csv(f_datn, header=None)
    # binpts=5000; mi=min(df[0]); ma=mi+10;
    # df_hist = histogram(df[0], bins=binpts, range=(mi, ma))
    #change point
    df = pd.read_csv(f_emplot, header=None, sep='\t')
    df_diff= diff(df[0])
    #calculating Ton and Toff
    df_tag = df[[0, 1]];# df_ton = df_ton[1:]
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
    #remove NAN values:
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
#------------------HISTOGRAM ON/OFF: given folder name, potential and list of point number, plot histogram at certain potentialof ------------------
def histogram_on_off_1mol(foldername= foldername, input_potential=[100], pointnumbers=[1],
                          bins_on=50, range_on=[0, 0.2], bins_off=50, range_off=[0, 0.5], plotting=False):
    df_datn_emplot, df_FCS, folder = dir_mV_molNo(foldername)
    df_specific = df_datn_emplot[df_datn_emplot['Point number'].isin(pointnumbers)]#keep all the points that exist
    df_specific = df_specific[df_specific['Potential'].isin(input_potential)]; df_specific.reset_index(drop=True, inplace=True)
    f_datn_path = df_specific['filepath[.datn]'].values[0]
    f_emplot_path = df_specific['filepath[.em.plot]'].values[0]
    df_ton, df_toff, average_ton, average_toff = t_on_off_fromCP(f_datn_path, f_emplot_path)
    t_ons = np.array(df_ton);
    t_offs = np.array(df_toff)
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

def histogram_on_off_folder(foldername= foldername, input_potential=[100], pointnumbers=range(100),
                          bins_on=50, range_on=[0, 0.2], bins_off=50, range_off=[0, 0.5], plotting=False):
    df_datn_emplot, df_FCS, folder = dir_mV_molNo(foldername)
    df_specific = df_datn_emplot[df_datn_emplot['Point number'].isin(pointnumbers)]#keep all the points that exist
    df_specific = df_specific[df_specific['Potential'].isin(input_potential)]; df_specific.reset_index(drop=True, inplace=True)
    t_ons = []; t_offs = [];
    for i in range(len(df_specific)):
        f_datn_path = df_specific['filepath[.datn]'].values[i]
        f_emplot_path = df_specific['filepath[.em.plot]'].values[i]
        df_ton, df_toff, average_ton, average_toff = t_on_off_fromCP(f_datn_path, f_emplot_path);
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
#------------------TIME TRACE OUTPUT-------All parameters are calculaed from the time traces of molecule----t_on, t_off, t_ratio....also from FCS...........
pointnumbers = linspace(1, 40, 40);pointnumbers = pointnumbers.astype(int);
potentialist = linspace(-100, 200, 1+(200-(-100))/5);
def timetrace_outputs_folderwise(folderpath=foldername, pointnumbers=[1], potentialist=potentialist):
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
                        df_ton, df_toff, average_ton, average_toff, average_ton_err, average_toff_err = t_on_off_fromCP(df_datn_path, df_em_path)
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
