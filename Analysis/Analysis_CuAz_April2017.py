#Import modules
import numpy as np
import pandas as pd
import os.path
from pylab import *
import glob
import os
import re
from xlwt import Workbook
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
global pointnumber


from numpy import sqrt, pi, exp, linspace, loadtxt
from lmfit import  Model, Parameter, Parameters

import matplotlib.pyplot as plt

def T_off_average(f_datn, f_emplot):

    #expt data

    df = pd.read_csv(f_datn, header=None)
    binpts=5000; mi=min(df[0]); ma=mi+10;
    df_hist = histogram(df[0], bins=binpts, range=(mi, ma))

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
    average_ton = np.average(df_ton);
    average_toff = np.average(df_toff);
    return(average_ton, average_toff, df_ton, df_toff)


def time_trace_plot(foldername='S101d14Feb17_60.5_635_A2_CuAzu655', input_potential=[0, 25],
                    input_number=1, x_lim_min=0, y_lim_min=0, x_lim_max=5, y_lim_max=5000, bin=1, show_changepoint=True):
    """bin=1 in millisecond"""
    maindir = os.getcwd()
    os.chdir(foldername)
    folderdir = os.getcwd()
    extensions = [".datn"] #file extensions we are interested in
    fig, ax = plt.subplots(figsize = (10, 8))
    subplots_adjust(hspace=0.000);
    
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
                
                #list_pointnumbers.append(pointnumber) #make a list will all the pointnumbers
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
                for i in range(len(input_potential)):
                    given_potential = input_potential[i]
                    if potentential == given_potential and pointnumber == input_number:
                        os.chdir(dirpath)
                        f_datn = filename
                        f_emplot = re.sub('.datn$','.datn.em.plot',f_datn)
                        ax = subplot(len(input_potential),1,i+1)
                        if os.path.isfile(f_emplot):
                            #raw data
                            df = pd.read_csv(f_datn, header=None)
                            tt_length=max(df[0])-min(df[0])
                            tt_length = round(tt_length, 0)
                            binpts=tt_length*1000/bin
                            df_hist = histogram(df[0], bins=binpts,range=(min(df[0]), max(df[0])))
                            #change point
                            df = pd.read_csv(f_emplot, header=None, sep='\t')

                            #----time trace overlapped with change-points
                            plt.plot()
                            plot(df_hist[1][:-1], df_hist[0]/bin, 'b', label=str(given_potential)+" mV")#original data
                            if show_changepoint == True:
                                plot(df[0], df[1]*0.8/1000, 'r', linewidth=2, label='')#change-point analysis
                            xticks([])
                            yticks(range(0, y_lim_max, 2), fontsize=16)
                            if i == len(input_potential)-1:
                                xticks(range(0, x_lim_max+1, 1), fontsize=16)
                                yticks(range(0, y_lim_max, 2), fontsize=16)
                                ax.set_xlabel('time/s', fontsize=16)
                            tick_params(
                                axis='y',          # changes apply to the x-axis
                                which='both',      # both major and minor ticks are affected
                                labelleft='on',      # ticks along the bottom edge are off
                                top='off',         # ticks along the top edge are off
                                labelbottom='off') # labels along the bottom edge are off
#                                 ax.set_ylabel('Fluorescence(kcps)', fontsize=16)
                            
                            xlim(x_lim_min, x_lim_max)
                            ylim(y_lim_min, y_lim_max)
                            
                            legend(fontsize=16, framealpha=0.5)
                        else:
                            print("The file %s does not exist" %f_emplot)
                        os.chdir(folderdir)
    ax.set_ylabel('Fluorescence(kcps)', fontsize=16)
    os.chdir(maindir)    
    return(fig)

def FCS_mono_fit(filename,tmin,tmax):
    df_fcs = pd.read_csv(filename, index_col=False, names=None, skiprows=1, header=None, sep='\ ', engine='python');
    df_fcs = df_fcs[df_fcs[0]>=tmin];
    df_fcs = df_fcs[df_fcs[0]<=tmax];
    df_fcs = df_fcs[np.isfinite(df_fcs[2])]
    np.isnan(df_fcs[0]);#removing nan file
    xdata=df_fcs[0];
    ydata=df_fcs[2];
    def mono_exp(x, A1, A2, t1):
        return((A1+A2*exp(-x/t1)))
    
    monofit, pcov_mono = curve_fit(mono_exp, xdata, ydata, p0 = [10, 1, 1], bounds=(0, np.inf))
    return(monofit)

def FCS_bi_fit(filename,tmin,tmax):
    df_fcs = pd.read_csv(filename, index_col=False, names=None, skiprows=1, header=None, sep='\ ', engine='python');
    df_fcs = df_fcs[df_fcs[0]>=tmin];
    df_fcs = df_fcs[df_fcs[0]<=tmax];
    df_fcs = df_fcs[np.isfinite(df_fcs[2])]
    np.isnan(df_fcs[0]);#removing nan file
    xdata=df_fcs[0];
    ydata=df_fcs[2];
    def biexp(x, A1, A2, t1, A3, t2):
        return(A1+A2*exp(-x/t1))+A3*exp(-x/t2)
    try:
        bifit, pcov_bi = curve_fit(biexp, xdata, ydata, p0 = [10, 1, 1, 0.5, 1], bounds=(0, np.inf))
    except RuntimeError:
        bifit = [NaN,NaN,NaN,NaN,NaN]
        print('Runtime Error %s' %filename)
    return(bifit)



def FCS_plot(foldername, tmin, tmax, pnt_numb, pnt_pot, kind):
    maindir = os.getcwd()
    os.chdir(foldername)
    folderdir = os.getcwd()
    extensions = [".dat"] #file extensions we are interested in
    
    for dirpath, dirnames, filenames in os.walk("."):
        for filename in [f for f in filenames if f.endswith(tuple(extensions))]: 
            #looking through all folders
            string_1 = 'mV'
            string_2 = 'FCS'
            position_FCS = filename.find(string_2)
            #determine whether or not it is FCS file
            if position_FCS in [-1]: #no FCS in name --> time trace file
                continue
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
                if pointnumberFCS == pnt_numb and potential_FCS == pnt_pot:
                    os.chdir(dirpath)
                    df_fcs = pd.read_csv(filename, index_col=False, names=None, skiprows=1, header=None, sep='\ ', engine='python');
                    df_fcs = df_fcs[df_fcs[0]>=tmin];
                    df_fcs = df_fcs[df_fcs[0]<=tmax];
                    df_fcs = df_fcs[np.isfinite(df_fcs[2])]
                    np.isnan(df_fcs[0]);#removing nan file
                    xdata=df_fcs[0];
                    ydata=df_fcs[2]; #reading in data
                    plt.plot(xdata,ydata, color = 'k') #plot raw data
                    plt.xscale('log')
                    def biexp(x, A1, A2, t1, A3, t2): #fit bi-exp
                        return(A1+A2*exp(-x/t1))+A3*exp(-x/t2)
                    def mono_exp(x, A1, A2, t1): #fit mono
                        return((A1+A2*exp(-x/t1)))
                    if kind in ['mono', 'Mono']: #determine what you want to plot
                        monofit = FCS_mono_fit(filename, tmin, tmax)
                        plt.plot(xdata, mono_exp(xdata, *monofit), color = 'b', linewidth=2.0)
                        print('g(t) = %s + %s * exp(-t/%s)' %(monofit[0], monofit[1], monofit[2]))
                    if kind in ['bi', 'Bi']:
                        bifit = FCS_bi_fit(filename, tmin, tmax)
                        plt.plot(xdata, biexp(xdata, *bifit), color = 'r', linewidth=2.0)
                        print('g(t) = %s + %s * exp(-t/%s) + %s * exp(-t/%s)' %(bifit[0], bifit[1], bifit[2], bifit[3], bifit[4]))
                    if kind in ['both', 'Both']:
                        monofit = FCS_mono_fit(filename, tmin, tmax)
                        plt.plot(xdata, mono_exp(xdata, *monofit), color = 'b', linewidth=2.0)
                        print('g(t) = %s + %s * exp(-t/%s)' %(monofit[0], monofit[1], monofit[2]))
                        bifit = FCS_bi_fit(filename, tmin, tmax)
                        plt.plot(xdata, biexp(xdata, *bifit), color = 'r', linewidth=2.0)
                        print('g(t) = %s + %s * exp(-t/%s) + %s * exp(-t/%s)' %(bifit[0], bifit[1], bifit[2], bifit[3], bifit[4]))
                    plt.xscale('log')
                    os.chdir(folderdir)
    os.chdir(maindir)
    return()


def midpoint_histograms(excel_name, excel_name_FCS, imp_pot, tminFCS, tmaxFCS, minimal_points, inp_bins, min_range, max_range, min_x1, max_x1, min_x2, max_x2):
    wb = Workbook()
    wb2 = Workbook()
    maindir = os.getcwd()
    directories_path = sorted([os.path.join(os.getcwd(),item) for item in next(os.walk(os.curdir))[1]])
    directories_name = next(os.walk('.'))[1] #listing all the top level directories in the main folder.
    #for i in directories_name:
     #   wb.add_sheet(i[5:24],cell_overwrite_ok=True) #adding sheets with part name directories

    #for i in range(len(directories_name)):
        #adding foldername on row 1 column 1
        #ws = wb.get_sheet(i)
        #ws.write(0, 0, 'Foldername: %s' %directories_name[i]) #write in sheet (row, col, 'information')
        
    extensions = [".datn",".dat"] #file extensions we are interested in
    list_direct = []
    w, h = 24, 300;
    t_ratio_TT = [[None for x in range(w)] for y in range(h)]
    t_ratio_FCS = [[None for x in range(w)] for y in range(h)]
    list_pointnumbers = []
    potential_array = [] #array with the potentials
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
                
                #list_pointnumbers.append(pointnumber) #make a list will all the pointnumbers
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
                if potentential not in potential_array:
                    potential_array.append(int(potentential)) #array with unique potentials 
                for i in range(len(potential_array)): 
                #put the on/off times in the right column with the corresponding potential
                    if int(potentential) == int(potential_array[i]):
                        if potentential >= imp_pot:
                        #max value of TT, below this value will be FCS.
                            if filename.endswith('.datn'):
                                os.chdir(dirpath)
                                f_datn = filename
                                f_emplot = re.sub('.datn$','.datn.em.plot',f_datn)
                                if os.path.isfile(f_emplot):    
                                    all_times = T_off_average(f_datn, f_emplot)
                                    #calculation of the ratios
                                    t_on = all_times[0]
                                    t_off = all_times[1]
                                    t_on_hist = all_times[2]
                                    t_off_hist = all_times[3]
                                    t_ratio_calc = t_off / t_on 
                                    #writing the calculations to the right place in the excell file
                                    for j in range(len(directories_name)):
                                        if directories_name[j] in dirpath:
                                            if directories_name[j] not in list_direct:
                                                list_direct.append(directories_name[j])
                                                wb.add_sheet(directories_name[j][5:24],cell_overwrite_ok=True)
                                                wb2.add_sheet(directories_name[j][5:24],cell_overwrite_ok=True)
                                                for x in range(len(list_direct)):
                                                    if directories_name[j] == list_direct[x]:
                                                        ws = wb.get_sheet(x)
                                                        ws.write(pointnumber+2,i+2,t_ratio_calc)
                                                        t_ratio_TT[x * 24 + pointnumber-1][i] = t_ratio_calc

                                            else:
                                                for x in range(len(list_direct)):
                                                    if directories_name[j] == list_direct[x]:
                                                        ws = wb.get_sheet(x)
                                                        ws.write(pointnumber+2,i+2,t_ratio_calc)
                                                        t_ratio_TT[x * 24 + pointnumber-1][i] = t_ratio_calc
                                    
                                else:
                                    print("The file %s does not exist" %f_emplot)
                                os.chdir(maindir)
                    
            
            
            else:
                #exactly same calculations as above but for the FCS files
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
                if potential_FCS not in potential_array:
                    potential_array.append(potential_FCS) #array with unique potentials
                for k in range(len(potential_array)):
                    if potential_FCS == potential_array[k]:
                        if potential_FCS < imp_pot:
                            os.chdir(dirpath)
                            variables_fit = FCS_bi_fit(filename,tminFCS,tmaxFCS)
                            var1 = variables_fit[0] #A1
                            var2 = variables_fit[1] #B1
                            var4 = variables_fit[3] #B2
                            var5 = variables_fit[2] #C1
                            var6 = variables_fit[4] #C2
                            
                            ton_1 = ((var1/var2) + 1) * var5 
                            ton_2 = ((var1/var4) + 1) * var6
                            ratio_1 = var2/var1
                            ratio_2 = var4/var1
                            if ton_1 > ton_2:
                                for h in range(len(directories_name)):
                                        if directories_name[h] in dirpath:
                                            if directories_name[h] not in list_direct:
                                                list_direct.append(directories_name[h])
                                                wb.add_sheet(directories_name[h][5:24],cell_overwrite_ok=True)
                                                wb2.add_sheet(directories_name[h][5:24],cell_overwrite_ok=True)
                                                for x in range(len(list_direct)):
                                                    if directories_name[h] == list_direct[x]:
                                                        ws = wb.get_sheet(x)
                                                        ws2 = wb2.get_sheet(x)
                                                        ws.write(pointnumberFCS+2,k+2,ratio_1)
                                                        ws2.write(pointnumberFCS+2,k+2,ratio_1)
                                                        t_ratio_TT[x * 24 + pointnumberFCS-1][k] = ratio_1
                                                        t_ratio_FCS[x * 24 + pointnumberFCS-1][k] = ratio_1
                                            else:
                                                for x in range(len(list_direct)):
                                                    if directories_name[h] == list_direct[x]:
                                                        ws = wb.get_sheet(x)
                                                        ws.write(pointnumberFCS+2,k+2,ratio_1)
                                                        ws2 = wb2.get_sheet(x)
                                                        ws2.write(pointnumberFCS+2,k+2,ratio_1)
                                                        t_ratio_TT[x * 24 + pointnumberFCS-1][k] = ratio_1
                                                        t_ratio_FCS[x * 24 + pointnumberFCS-1][k] = ratio_1

                            else:
                                for l in range(len(directories_name)):
                                        if directories_name[l] in dirpath:
                                            if directories_name[l] not in list_direct:
                                                list_direct.append(directories_name[l])
                                                wb.add_sheet(directories_name[l][5:24],cell_overwrite_ok=True)
                                                wb2.add_sheet(directories_name[l][5:24],cell_overwrite_ok=True)
                                                for x in range(len(list_direct)):
                                                    if directories_name[l] == list_direct[x]:
                                                        ws = wb.get_sheet(x)
                                                        ws2 = wb2.get_sheet(x)
                                                        ws.write(pointnumberFCS+2,k+2,ratio_2)
                                                        ws2.write(pointnumberFCS+2,k+2,ratio_2)
                                                        t_ratio_TT[x * 24 + pointnumberFCS-1][k] = ratio_2
                                                        t_ratio_FCS[x * 24 + pointnumberFCS-1][k] = ratio_2
                                            else:
                                                for x in range(len(list_direct)):
                                                    if directories_name[l] == list_direct[x]:
                                                        ws = wb.get_sheet(x)
                                                        ws2 = wb2.get_sheet(x)
                                                        ws.write(pointnumberFCS+2,k+2,ratio_2)
                                                        ws2.write(pointnumberFCS+2,k+2,ratio_2)
                                                        t_ratio_TT[x * 24 + pointnumberFCS-1][k] = ratio_2
                                                        t_ratio_FCS[x * 24 + pointnumberFCS-1][k] = ratio_2
                    
                    
                    
                    
                        else:
                            os.chdir(dirpath)
                            variables_fit_mono = FCS_mono_fit(filename,tminFCS,tmaxFCS)
                            var1 = variables_fit_mono[0] #A
                            var2 = variables_fit_mono[1] #B
                            ratio_mono = var2 / var1
                            for l in range(len(directories_name)):
                                        if directories_name[l] in dirpath:
                                            if directories_name[l] not in list_direct:
                                                list_direct.append(directories_name[l])
                                                wb.add_sheet(directories_name[l][5:24],cell_overwrite_ok=True)
                                                wb2.add_sheet(directories_name[l][5:24],cell_overwrite_ok=True)
                                                for x in range(len(list_direct)):
                                                    if directories_name[l] == list_direct[x]:
                                                        ws2 = wb2.get_sheet(x)
                                                        ws2.write(pointnumberFCS+2,k+2,ratio_mono)
                                                        t_ratio_FCS[x * 24 + pointnumberFCS-1][k] = ratio_mono
                                            else:
                                                for x in range(len(list_direct)):
                                                    if directories_name[l] == list_direct[x]:
                                                        ws2 = wb2.get_sheet(x)
                                                        ws2.write(pointnumberFCS+2,k+2,ratio_mono)
                                                        t_ratio_FCS[x * 24 + pointnumberFCS-1][k] = ratio_mono
            
            
            
            
            
            
            
            
            os.chdir(maindir)
            

    def nernst(x, a):
        return(10**((x - a) / 0.059))
    
    
    for i in range(len(list_direct)):
        ws = wb.get_sheet(i)
        ws2 = wb2.get_sheet(i)
        ws.write(0, 1, 'Potential(mV):')
        ws2.write(0, 1, 'Potential(mV):')
        ws2.write(1, 1, 'Pointnumber:')
        ws.write(1, 1, 'Pointnumber:')
        ws2.write(0, len(potential_array)+2, 'Midpoint Potential (in mV):')
        
        for i in range(len(potential_array)):
            ws.write(0,i+2,potential_array[i]) 
            ws2.write(0,i+2,potential_array[i])
            #writing all the potentials to the sheets 
        for i in range(1,25):
            ws.write(i+2,1,i)
            ws2.write(i+2,1,i)

    #making the values into mV        
    potential_array[:] = [x / 1000 for x in potential_array]
    
    #This part goes through the matrix with all the ratios 
    
    list_mp = []    
    midpoint_potential_array = []
    for j in range(len(list_direct) * 24):
        list_1 = []
        potential_1 = []
        all_midpoint_potential = []
        for i in range(len(potential_array)):
            

            if t_ratio_TT[j][i] is not None:
                #saving ratios with their potentials for each point
                list_1.append(t_ratio_TT[j][i])
                potential_1.append(potential_array[i])

        if len(list_1) >= minimal_points:
            fit_waardes, fit_variance = curve_fit(nernst, potential_1, list_1, p0 = 0.020)
            midpoint_potential_array.append(fit_waardes[0])
            list_mp.append(fit_waardes[0])
        else:
            list_mp.append(None)
            
        del list_1[:]
        del potential_1[:]
   
    midpoint_potential_array_FCS = []
    list_mp_FCS = []
    for j in range(len(list_direct) * 24):
        list_FCS = []
        potential_FCS = []
        for i in range(len(potential_array)):
            if t_ratio_FCS[j][i] is not None:
                list_FCS.append(t_ratio_FCS[j][i])
                potential_FCS.append(potential_array[i])                
        if len(list_FCS) >= minimal_points:
            fit_waardes, fit_variance = curve_fit(nernst, potential_FCS, list_FCS, p0 = 0.020)
            midpoint_potential_array_FCS.append(fit_waardes[0])
            list_mp_FCS.append(fit_waardes[0])
        else:
            list_mp_FCS.append(None)
            
            
        del list_FCS[:]
        del potential_FCS[:]
    for i in range(len(list_mp)):
        if list_mp[i] is not None:
            list_mp[i] = list_mp[i] * 1000
    for i in range(len(list_mp_FCS)):
        if list_mp_FCS[i] is not None:
            list_mp_FCS[i] = list_mp_FCS[i] * 1000
    
    
    for i in range(len(list_direct)):
        ws = wb.get_sheet(i)
        for j in range(24):
            ws.write(j+3, len(potential_array)+2, list_mp[i*24 + j])
        ws2 = wb2.get_sheet(i)
        for k in range(24):
            ws2.write(k+3, len(potential_array)+2, list_mp_FCS[i*24 + k])
        

        
            
    print('The average midpoint potential according to TT:')
    print(sum(midpoint_potential_array)/len(midpoint_potential_array))
    print('The average midpoint potential according to FCS:')
    print(sum(midpoint_potential_array_FCS)/len(midpoint_potential_array_FCS))
    
    #-------FIgure Parameters---------------
    fig, axes = subplots(figsize=(10, 10), ncols=1, nrows=2)
    # plt.figure()
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.size"] = "14"

    range_fit = [min_range, max_range]
    bins = inp_bins
    bin_centers_on = linspace(range_fit[0], range_fit[1], bins)
    x=linspace(range_fit[0], range_fit[1], 100)# for a smooth fitting plot include more points

    def gaus(x,M,sigma):
        return exp(-((x-M)**2)/(2*sigma**2))/(sigma*sqrt(2*pi))

    #Change point--------------------------
    n,bins_on1,patches = axes[0].hist(midpoint_potential_array, bins = bins, range=range_fit, color='b', label='Change Point')
    popt, pcov = curve_fit(gaus, bin_centers_on, n)
    E_mean = popt[0]*1000; #in mV
    FWHM = 2.3548*popt[1]*1000; #in mV
    axes[0].plot(x,gaus(x,*popt), color = 'k', linewidth = 3, label='$E^0_{avg}$ is %.2f mV\n FWHM is %.2f mV' %(E_mean, FWHM))
    axes[0].set_xlim(min_x1, max_x1)
    axes[0].set_ylabel('P', fontsize=16)
    axes[0].legend()

    #FCS plot------------------------------
    n,bins_on1,patches = axes[1].hist(midpoint_potential_array_FCS, bins = bins, range=range_fit, color='r', label='FCS')
    popt, pcov = curve_fit(gaus, bin_centers_on, n)
    E_mean = popt[0]*1000; #in mV
    FWHM = 2.3548*popt[1]*1000; #in mV
    axes[1].plot(x, gaus(x,*popt), color = 'k', linewidth = 3, label='$E^0_{avg}$ is %.2f mV\n FWHM is %.2f mV' %(E_mean, FWHM))

    axes[1].set_xlim(min_x2, max_x2)
    axes[1].set_xlabel('Potential[Volt]')
    axes[1].set_ylabel('P')
    axes[1].legend()

    fig.tight_layout()
    plt.show()
    
    
    wb.save(excel_name)
    wb2.save(excel_name_FCS)
    
    return(midpoint_potential_array, midpoint_potential_array_FCS)

def histograms(pot, pointnumbers, specific_potential, rnge_on, rnge_off, bins_on, bins_off, proteins, homedir, max_his_on, max_his_off, x_shift, plots=False):
    parentdir = os.getcwd()
    os.chdir(homedir)
    homedir1 = os.getcwd()
    potential, pntnumbers = pot, pointnumbers #creates an array with dimension potential (col) x pntnumbers (rows) 
    on_time = []
    off_time = []
    extensions = [".datn",".dat"]
    all_on_times_array = [[None] * potential for i in range(pntnumbers)]
    all_off_times_array = [[None] * potential for i in range(pntnumbers)]
    df3 = pd.DataFrame(columns=[specific_potential])
    df3_off = pd.DataFrame(columns=[specific_potential])
    for dirpath, dirnames, filenames in os.walk("."):
        for filename in [f for f in filenames if f.endswith(tuple(extensions))]:
            string_1 = 'mV'
            string_2 = 'FCS'
            position_FCS = filename.find(string_2)
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
                if potentential == int(specific_potential):
                    if filename.endswith('.datn'):
                        os.chdir(dirpath)
                        f_datn = filename
                        f_emplot = re.sub('.datn$','.datn.em.plot',f_datn)
                        if os.path.isfile(f_emplot):    
                            all_times = T_off_average(f_datn, f_emplot)
                            t_on = all_times[0]
                            t_off = all_times[1]
                            t_on_hist = all_times[2]
                            t_off_hist = all_times[3]
                            on_time = t_on
                            off_time = t_off
                            on_time_df = pd.DataFrame(t_on_hist,columns = [specific_potential])
                            off_time_df = pd.DataFrame(t_off_hist,columns = [specific_potential])
                            df3 = df3.append(on_time_df)
                            df3_off = df3_off.append(off_time_df)
                        else:
                            print("The file %s does not exist" %f_emplot)
                        os.chdir(homedir1)
            
            
    os.chdir(parentdir)       
 
    df_on_shifted = df3.shift(+1) ## shift up
    df_on_shifted.drop(df3.shape[0] - 1,inplace = True)
    df_off_shifted = df3_off.shift(+1) ## shift up
    df_off_shifted.drop(df3_off.shape[0] - 1,inplace = True)
    
    df_on_shifted_x = df3.shift(+x_shift) ## shift up
    df_on_shifted_x.drop(df3.shape[0] - x_shift,inplace = True)
    df_off_shifted_x = df3_off.shift(+x_shift) ## shift up
    df_off_shifted_x.drop(df3_off.shape[0] - x_shift,inplace = True)
  
    
    def single_exp(x_values, constant1,constant2):
        return(constant1*exp(-constant2*x_values))

    def double_exp(x_values, constant3, constant4, constant5, constant6):
        return(constant3*exp(-constant4*x_values) - constant5*exp(-constant6*x_values))
                     
    if plots==True:
        
        fig1, axes1 = plt.subplots(1, 2, figsize=(10,4))

        n,bins_on1,patches = axes1[0].hist(df3, range=(0,max_his_on),bins=bins_on)
        axes1[0].set_xlabel(r'$\tau_{on}$')
        axes1[0].set_ylabel('#')
        axes1[0].set_yscale('log')
        axes1[0].set_title("ON time %s-Azu %smV" %(proteins, specific_potential))
        '''
        bin_centers_on = bins_on1[:-1] + 0.5 * (bins_on1[1:] - bins_on1[:-1])
        try:
            popt_single, pcov_single = curve_fit(single_exp, bin_centers_on, n, p0 = [1, 1])
            plt.plot(bin_centers_on, single_exp(bin_centers_on, *popt_single))
            print(r'Fit ON time histogram: %s * e^{-%s t}' %(popt_single[0],popt_single[1]))

        except RuntimeError:
            print('Fit failed')
        '''
        n_off,bins_off1,patches_off = axes1[1].hist(df3_off, range=(0,max_his_off),bins=bins_off)
        axes1[1].set_xlabel(r'$\tau_{off}$')
        axes1[1].set_ylabel('#')
        axes1[1].set_yscale('log')
        axes1[1].set_title('OFF time %s-Azu %smV' %(proteins, specific_potential))

        '''
        bin_centers_off = bins_off1[:-1] + 0.5 * (bins_off1[1:] - bins_off1[:-1])    
        try:
            popt_double, pcov_double = curve_fit(double_exp, bin_centers_off, n_off, p0 = [1, 1, 1, 1])
            plt.plot(bin_centers_off, double_exp(bin_centers_off, *popt_double))
            print(r'Fit OFF time histogram: %s * e^{-%s t} - %s * e^{-%s t}' %(popt_double[0],popt_double[1],popt_double[2],popt_double[3]))


        except RuntimeError:
            print('Fit failed')
        '''

        fig2 = plt.figure(figsize=(12,10))
        ax3 = fig2.add_subplot(2,2,1)
        hist2d(df3[specific_potential],df_on_shifted[specific_potential], range=rnge_on ,bins=bins_on, norm=LogNorm());
        colorbar()
        ax3.set_title('ON time %s-Azu %smV' %(proteins, specific_potential))
        ax3.set_xlabel(r'$\tau_{on}/s$')
        ax3.set_ylabel(r'$\tau_{on}+1/s$')

        ax4 = fig2.add_subplot(2,2,2)
        hist2d(df3_off[specific_potential],df_off_shifted[specific_potential], range=rnge_off,bins=bins_off, norm=LogNorm());
        colorbar()
        ax4.set_title('OFF time %s-Azu %smV' %(proteins, specific_potential))
        ax4.set_xlabel(r'$\tau_{off}/s$')
        ax4.set_ylabel(r'$\tau_{off}+1/s$')
        plt.tight_layout()

        ax5 = fig2.add_subplot(2,2,3)
        hist2d(df3[specific_potential],df_on_shifted_x[specific_potential], range=rnge_on ,bins=bins_on, norm=LogNorm());
        colorbar()
        ax5.set_title('ON time %s-Azu %smV' %(proteins, specific_potential))
        ax5.set_xlabel(r'$\tau_{on}/s$')
        ax5.set_ylabel(r'$\tau_{off}+%s/s$' %x_shift)

        ax6 = fig2.add_subplot(2,2,4)
        hist2d(df3_off[specific_potential],df_off_shifted_x[specific_potential], range=rnge_off,bins=bins_off, norm=LogNorm());
        colorbar()
        ax6.set_title('OFF time %s-Azu %smV' %(proteins, specific_potential))
        ax6.set_xlabel(r'$\tau_{off}/s$')
        ax6.set_ylabel(r'$\tau_{off}+%s/s$' %x_shift)
        plt.tight_layout()    

    return(df3, df_on_shifted, df_on_shifted_x, df3_off, df_off_shifted, df_off_shifted_x)    