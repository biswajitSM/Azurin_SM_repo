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
foldername = r'D:\Research\Experimental\Analysis\2017analysis\201702\Analysis_Sebby_March_2017\S101d14Feb17_60.5_635_A2_CuAzu655'
def dir_mV_molNo(foldername=foldername):
    """input: Path of the Folder name as: foldername = r'D:\Research\Experimental\Analysis\2017analysis\201702\Analysis_Sebby_March_2017\S101d14Feb17_60.5_635_A2_CuAzu655'
	-----
	Output: df_datn_emplot, df_FCS, foldername
	e.g. df_datn_emplot, df_FCS, foldername = dir_mV_molNo(foldername)
	df_datn_emplot gives the list of points with their number, potential values and pathdirectory
	"""
#     maindir = os.getcwd()
    os.chdir(foldername)
#     folderdir = os.getcwd()
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
#                 print(pointnumber)
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
                    f_emplot = re.sub('.datn$','.datn.em.plot',f_datn)
                    temp_output = pd.DataFrame([[pointnumber, int(potentential), f_datn, foldername+dirpath[1:]+'\\'+f_datn, f_emplot, foldername+dirpath[1:]+'\\'+f_emplot]], columns=columns)
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
                temp_outputFCS = pd.DataFrame([[int(pointnumberFCS), int(potentential_FCS), f_FCS, foldername+dirpath[1:]+'\\'+f_FCS]], columns=columns_FCS)
                df_FCS = df_FCS.append(temp_outputFCS, ignore_index=True)

    os.chdir(foldername)    
    return(df_datn_emplot, df_FCS, foldername)
#---------given folder name, point number and list of potential, plot time traces at diff potentialof same molecule------
foldername = r'D:\Research\Experimental\Analysis\2017analysis\201702\Analysis_Sebby_March_2017\S101d14Feb17_60.5_635_A2_CuAzu655'
def time_trace_plot(foldername= foldername, input_potential=[0, 25, 50, 100],
                    input_number=1, x_lim_min=0, y_lim_min=0, x_lim_max=5, y_lim_max=6, bin=5, show_changepoint=True):
    """bin=1 in millisecond
    foldername should be given as r'D:\Research\...'
    """
    df_datn_emplot, df_FCS, folder = dir_mV_molNo(foldername)
    df_specific = df_datn_emplot[df_datn_emplot['Point number']==input_number]
    df_specific = df_specific[df_specific['Potential'].isin(input_potential)]; df_specific.reset_index(drop=True, inplace=True)
    fig, ax = plt.subplots(figsize = (10, 8),sharex=True, sharey=True)
    subplots_adjust(hspace=0.000);
    for i in range(len(df_specific)):
        given_potential = df_specific['Potential'][i]
        print(given_potential)
        df_specific_V = df_specific[df_specific['Potential'] == given_potential]
        f_datn_path = df_specific_V['filepath[.datn]'].values[0]
        f_emplot_path = df_specific_V['filepath[.em.plot]'].values[0]
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
        df = pd.read_csv(f_emplot, header=None, sep='\t') #change-point
        if show_changepoint == True:
            plot(df[0], df[1]*0.8/1000, 'r', linewidth=2, label='')#change-point analysis
        xlim(x_lim_min, x_lim_max)
        ylim(0, y_lim_max)# 1.5*max(df[1]/1000)
        xticks([])
        yticks(range(0, y_lim_max, 2), fontsize=16)
        if i == len(df_specific)-1:
            xticks(range(0, x_lim_max+1, 1), fontsize=16)
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

#------------------TIME TRACE OUTPUT-------All parameters are calculaed from the time traces of molecule----t_on, t_off, t_ratio....also from FCS...........

pointnumbers = linspace(1, 40, 40);pointnumbers = pointnumbers.astype(int);
potentialist = linspace(-100, 200, 1+(200-(-100))/5);
folderpath = r'D:\Research\Experimental\Analysis\2017analysis\201702\Analysis_Sebby_March_2017\S101d14Feb17_60.5_635_A2_CuAzu655'
def timetrace_outputs_folderwise(folderpath=folderpath, pointnumbers=[1], potentialist=potentialist):
    df_datn_emplot, df_FCS, folderpath = dir_mV_molNo(foldername=folderpath)
    df_datn_emplot = df_datn_emplot[df_datn_emplot['Point number'].isin(pointnumbers)]#keep all the points that exist
    df_datn_emplot = df_datn_emplot[df_datn_emplot['Potential'].isin(potentialist)]
    out_total = pd.DataFrame()
    for input_number in pointnumbers:
        df_datnem_specific = df_datn_emplot[df_datn_emplot['Point number']==input_number]
        
        if not df_datnem_specific.empty:
            #---------Create Pandas array to save outputs----------
            indices = np.ones(13); indices=indices.astype(str)
            Point_number = 'Point_'+str(input_number)
            indices[:]=Point_number
            group_1 = ['Potential']
            group_ind = np.ones(6);group_ind = group_ind.astype(str)
            group_2=group_ind.copy(); group_2[:]='t_ratio_timetrace'
            group_3 = group_ind.copy(); group_3[:]='t_ratio_FCS'
            group=concatenate((group_1, group_2, group_3))
            subgroup_1 = ['Potential']
            subgroup_2 = ['t_onav', 't_onaverr', 't_offav','t_offaverr','t_ratio', 't_ratioerr']
            subgroup_3 = subgroup_2;
            subgroup = concatenate((subgroup_1, subgroup_2, subgroup_3))
            arrays = [indices, group, subgroup]
            col = pd.MultiIndex.from_arrays(arrays)
            length=(len(df_datnem_specific))#for defining dimension of out_mat
            out_point = pd.DataFrame(zeros((length, len(subgroup))), columns=col)#create zeroes which will be replaced by proper values

            #---------Pandas array created to save outputs----
            out_point[Point_number]['Potential']=df_datnem_specific['Potential']
            t_onav=[];t_onaverr=[]; t_offav=[]; t_offaverr=[]; t_ratio=[]; t_ratioerr=[] #Empty forms for timetrace output
            t_onavfcs=[];t_onaverrfcs=[]; t_offavfcs=[]; t_offaverrfcs=[]; t_ratiofcs=[]; t_ratioerrfcs=[] #Empty forms for timetrace output
            for i in range(length):
                # .datn and em.plot data analysis
                t_onav.append(i)
                t_onaverr.append(i+1)
                t_offav.append(i+3)
                t_offaverr.append(i+4)
                t_ratio.append(i+10)
                t_ratioerr.append(i+20)
                potential = out_mat['Point number']['Potential']['Potential'][i]
                df_datnem_potential = df_datn_emplot[df_datn_emplot['Potential']==potential]
                df_datnem_potential = df_datn_emplot[df_datn_emplot['Potential']==potential]
                df_datnem_potential.reset_index(drop=True, inplace=True);
                df_datn_path = df_datnem_potential['filepath[.datn]'][0]
                df_em_path = df_datnem_potential['filepath[.em.plot]'][0]
                
                df_datn = pd.read_csv(df_datn_path, header=None)
                df_emplot = pd.read_csv(df_em_path, header=None, sep='\t')
                
                # FCS data analysis
                t_onavfcs.append(i*2)
                t_onaverrfcs.append(i*5)
                t_offavfcs.append(i*1+1)
                t_offaverrfcs.append(i*0.5)
                t_ratiofcs.append(i*6)
                t_ratioerrfcs.append(i*7)
            df_create=array([t_onav, t_onaverr, t_offav, t_offaverr, t_ratio, t_ratioerr])
            df_create=df_create.astype(float64)#;a=pd.DataFrame(a)
            df_create = pd.DataFrame(df_create.T, columns=['t_onav', 't_onaverr', 't_offav','t_offaverr','t_ratio', 't_ratioerr'])
            out_point[Point_number]['t_ratio_timetrace']=df_create
            
            df_create_fcs = array([t_onavfcs,t_onaverrfcs, t_offavfcs, t_offaverrfcs, t_ratiofcs, t_ratioerrfcs])
            df_create_fcs = df_create_fcs.astype(float64)
            df_create_fcs = pd.DataFrame(df_create_fcs.T, columns=subgroup_2)
            out_point[Point_number]['t_ratio_FCS']=df_create_fcs
            
            out_total=pd.concat([out_total, out_point], axis=1);
    return(out_total)
a=timetrace_outputs_folderwise()