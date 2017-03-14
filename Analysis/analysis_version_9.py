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

import matplotlib.pyplot as plt

wb = Workbook()
sheet1 = wb.add_sheet('on_times',cell_overwrite_ok=True)
sheet2 = wb.add_sheet('off_times',cell_overwrite_ok=True)
sheet3 = wb.add_sheet('on_off_times_FCS',cell_overwrite_ok=True)
sheet4 = wb.add_sheet('FCS fit values',cell_overwrite_ok=True)


sheet1.write(0,0,'Potential (mV):')
sheet2.write(0,0,'Potential (mV):')
sheet3.write(0,0,'Potential (mV):')
sheet1.write(1,0,'Point #')
sheet2.write(1,0,'Point #')
sheet3.write(1,0,'Point #')
for i in list(range(1,32)):
    sheet1.write(i+2,0,i)
    sheet2.write(i+2,0,i)
    sheet3.write(i+2,0,i)
    sheet4.write(i+1,0,i)


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


def FCS_fit(filename,tmin,tmax):
    df_fcs = pd.read_csv(filename, index_col=False, names=None, skiprows=1, header=None, sep='\ ', engine='python');
    df_fcs = df_fcs[df_fcs[0]>=tmin];
    df_fcs = df_fcs[df_fcs[0]<=tmax];
    df_fcs = df_fcs[np.isfinite(df_fcs[2])]
    np.isnan(df_fcs[0]);#removing nan file
    xdata=df_fcs[0];
    ydata=df_fcs[2];
    def biexp(x, A1, t1, A2, t2):
        return((A1*exp(-x/t1))+(A2*exp(-x/t2)))
    def mono_exp(x, A1, t1):
        return(A1*exp(-x/t1))
    try:
        monofit, pcov_mono = curve_fit(mono_exp, xdata, ydata, p0 = [10, 1], bounds=(0.01, np.inf))
    except RuntimeError:
        print('Error - mono exponential curve_fit failed for file: %s' %filename)
        monofit = np.zeros(2)

    try:
        bifit, pcov_bi = curve_fit(biexp, xdata, ydata, p0 = [10, 1, 0.5, 1], bounds=(0.01, np.inf))
    except RuntimeError:
        print('Error - bi-exponential curve_fit failed for file: %s' %filename)
        bifit = np.zeros(4)
    if np.count_nonzero(monofit) == 2 and np.count_nonzero(bifit) == 4:
        for i in range(len(xdata)):
            fit_value_mono = mono_exp(xdata,monofit[0],monofit[1])
            fit_value_bi = biexp(xdata,bifit[0],bifit[1],bifit[2],bifit[3])
            av_diff_mono = np.mean(abs(np.subtract(fit_value_mono, ydata)))
            #av. distance from the fit to the real data.
            av_diff_bi = np.mean(abs(np.subtract(fit_value_bi, ydata)))
            if av_diff_mono > av_diff_bi:
                fitvalues = bifit
            else:
                fitvalues = np.append(monofit,[0,0])
    elif np.count_nonzero(monofit) == 2:
        fitvalues = np.append(monofit,[0,0])
    elif np.count_nonzero(bifit) == 2:
        fitvalues = bifit

    return(fitvalues)



def average_on_and_off_times(titel, pot, pointnumbers, proteins, homedir):
    os.chdir(homedir)
    homedir1 = os.getcwd()
    potential, pntnumbers = pot, pointnumbers #creates an array with dimension potential (col) x pntnumbers (rows)
    on_time = [[None] * potential for i in range(pntnumbers)]
    off_time = [[None] * potential for i in range(pntnumbers)]
    extensions = [".datn",".dat"]
    potential_array = []

    all_on_times_array = [[None] * potential for i in range(pntnumbers)]
    all_off_times_array = [[None] * potential for i in range(pntnumbers)]
    df3 = pd.DataFrame(columns=[potential_array])
    df3_off = pd.DataFrame(columns=[potential_array])
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
                if potentential not in potential_array:
                    potential_array.append(potentential) #array with unique potentials
                    all_on_times_array.append(potentential)
                    df3[potentential] = np.nan
                    df3_off[potentential] = np.nan
                for i in range(len(potential_array)): #put the on/off times in the right column with the corresponding potential
                    if potentential == potential_array[i]:
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
                                sheet1.write(pointnumber+2,i+1,t_on)
                                sheet2.write(pointnumber+2,i+1,t_off)
                                on_time[pointnumber-1][i] = t_on
                                off_time[pointnumber-1][i] = t_off

                                on_time_df = pd.DataFrame(t_on_hist,columns = [potential_array[i]])
                                off_time_df = pd.DataFrame(t_off_hist,columns = [potential_array[i]])
                                df3 = df3.append(on_time_df)
                                df3_off = df3_off.append(off_time_df)
                            else:
                                print("The file %s does not exist" %f_emplot)
                            os.chdir(homedir1)

    df_ontime = pd.DataFrame(data = on_time, index = range(1,pntnumbers+1), columns = potential_array)
    df_offtime = pd.DataFrame(data = off_time, index = range(1,pntnumbers+1), columns = potential_array)
    av_on = np.transpose(df_ontime.mean())
    av_off = np.transpose(df_offtime.mean())

    fig = plt.figure(figsize=(12,10))
    ax1 = fig.add_subplot(211)
    plt.title('On time')
    plt.xlabel('potential(mV)')
    plt.ylabel('ms')
    for i in range(len(df_ontime.index)):
        plt.scatter(list(df_ontime),df_ontime.iloc[i])

    ax2 = fig.add_subplot(212)
    plt.title('Off time')
    plt.xlabel('potential(mV)')
    plt.ylabel('ms')
    for i in range(len(df_offtime.index)):
        plt.scatter(list(df_offtime),df_offtime.iloc[i])
    plt.tight_layout()
    fig.savefig("Cualltimes")

    fig = plt.figure(figsize=(12,10))
    ax1 = fig.add_subplot(211)
    plt.title('Average On time')
    plt.xlabel('potential(mV)')
    plt.ylabel('ms')
    plt.scatter(list(df_ontime),av_on)

    ax1 = fig.add_subplot(212)
    plt.title('Average Off time')
    plt.xlabel('potential(mV)')
    plt.ylabel('ms')
    plt.scatter(list(df_ontime),av_off)
    plt.tight_layout()
    fig.savefig("Cuavertimes")

    wb.save(titel)
    os.chdir('..')
    return()

def time_trace_plot(f_datn, f_emplot, x_lim_min, x_lim_max, y_lim_min, y_lim_max,binpts=1500):


    # #expt data

    df = pd.read_csv(f_datn, header=None)
    mi=min(df[0]); ma=mi+10;
    df_hist = histogram(df[0], bins=binpts)

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

    df_onhist= histogram(df_ton[0], bins=100, range=(0, 0.5))
    df_offhist = histogram(df_toff[0], bins=100, range=(0, 0.5))

    figure(figsize=(12,10))
    #----time trace overlapped with change-points
    plt.plot()
    plot(df_hist[1][:-1], df_hist[0]*binpts/(ma-mi), 'b')#original data
    plot(df[0], df[1]*2, 'r', linewidth=2)#change-point analysis
    xlim(x_lim_min, x_lim_max)
    ylim(y_lim_min, y_lim_max)
    xlabel('time/s', fontsize=14, fontname='Times New Roman');
    xticks(fontsize=14, fontname='Times New Roman');
    ylabel('counts/s', fontsize=14, fontname='Times New Roman');
    yticks(fontsize=14, fontname='Times New Roman')
    legend(['expt data', 'Change-Point'], framealpha=0.5)

    return()

def t_ratio(pot, pointnumbers, homedir, x_prots):
    os.chdir(homedir)
    homedir1 = os.getcwd()
    potential, pntnumbers = pot, pointnumbers #creates an array with dimension potential (col) x pntnumbers (rows)
    on_time = [[None] * potential for i in range(pntnumbers)]
    off_time = [[None] * potential for i in range(pntnumbers)]
    t_ratio = [[None] * potential for i in range(pntnumbers)]
    extensions = [".datn",".dat"]
    potential_array = []

    all_on_times_array = [[None] * potential for i in range(pntnumbers)]
    all_off_times_array = [[None] * potential for i in range(pntnumbers)]
    df3 = pd.DataFrame(columns=[potential_array])
    df3_off = pd.DataFrame(columns=[potential_array])
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
                if potentential not in potential_array:
                    potential_array.append(potentential) #array with unique potentials
                    all_on_times_array.append(potentential)
                    df3[potentential] = np.nan
                    df3_off[potentential] = np.nan
                for i in range(len(potential_array)): #put the on/off times in the right column with the corresponding potential
                    if potentential == potential_array[i]:
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
                                t_ratio_calc = t_on / t_off
                                on_time[pointnumber-1][i] = t_on
                                off_time[pointnumber-1][i] = t_off
                                t_ratio[pointnumber-1][i] = t_ratio_calc
                                on_time_df = pd.DataFrame(t_on_hist,columns = [potential_array[i]])
                                off_time_df = pd.DataFrame(t_off_hist,columns = [potential_array[i]])

                                df3 = df3.append(on_time_df)
                                df3_off = df3_off.append(off_time_df)
                            else:
                                print("The file %s does not exist" %f_emplot)
                            os.chdir(homedir1)
    os.chdir('..')

    df_ontime = pd.DataFrame(data = on_time, index = range(1,pntnumbers+1), columns = potential_array)
    df_offtime = pd.DataFrame(data = off_time, index = range(1,pntnumbers+1), columns = potential_array)
    df_t_ratio = pd.DataFrame(data = t_ratio, index = range(1,pntnumbers+1), columns = potential_array)


    fig = plt.figure(figsize=(12,10))
    color=iter(cm.rainbow(np.linspace(0,1,x_prots)))
    plt.title(r'$ \tau_{on} \tau_{off}^{-1}$ vs potential')
    plt.xlabel('potential(mV)')
    plt.yscale('log')
    plt.ylabel(r'$\tau_{on}\tau_{off}^{-1}$')
    for i in range(x_prots):
        c=next(color)
        plt.scatter(list(df_t_ratio),df_t_ratio.iloc[i], c = c)



    return()
